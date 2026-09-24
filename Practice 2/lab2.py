"""
Название файла: lab2.py
Семантика файла: Программа предназначена для поиска частых наборов товаров
в транзакционных данных с использованием алгоритма Apriori, поиска ассоциативных правил
при различных порогах достоверности, сохранения результатов в текстовый файл и визуализации.
"""

import itertools
import time
import matplotlib.pyplot as plt
import pandas as pd

# 1. Загрузка данных

file_path = "baskets.csv"


def load_from_csv(file_path):
    """
    Загружает транзакции из CSV-файла.

    Входные параметры:
        file_path (str): Путь к CSV-файлу с транзакциями.

    Возвращаемое значение:
        transactions (list): Список транзакций, где каждая транзакция представлена списком товаров (список списков).
    """
    df = pd.read_csv(file_path, header=None, encoding='cp1251')

    transactions = []
    for _, row in df.iterrows():
        items = [str(val).strip() for val in row.dropna() if str(val).strip() and str(val).strip() != 'nan']
        if items:
            transactions.append(items)

    return transactions


# 2. Реализация алгоритма Apriori


def apriori(transactions, min_support=0.05, sort_by='support_desc'):
    """
    Реализует алгоритм Apriori для поиска частых наборов товаров в транзакциях.

    Входные параметры:
        transactions (list of list): Список транзакций (список списков).
        min_support (float): Порог минимальной поддержки (от 0.0 до 1.0).
        sort_by (str): Критерий сортировки результатов ('support_desc' или 'lexicographic').

    Возвращаемое значение:
        results (list of tuple): Список найденных частых наборов, где каждый элемент содержит
        множество товаров (set) и значение поддержки (float) (список кортежей).
        support_dict (dict): Словарь поддержки для всех частых наборов формата {tuple: float}.
    """
    trans_sets = [set(t) for t in transactions]
    n_trans = len(trans_sets)
    if n_trans == 0:
        return [], {}

    # Шаг 1: Нахождение частых 1-элементных наборов
    item_counts = {}
    for t in trans_sets:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    L1 = []
    support_dict = {}

    for item, count in item_counts.items():
        sup = count / n_trans
        if sup >= min_support:
            L1.append(([item], count))
            support_dict[(item,)] = sup

    L = [L1]
    k = 2

    L1.sort(key=lambda x: x[0])

    while True:
        prev_frequent = [itemset for itemset, count in L[k - 2]]
        candidates = []

        # Шаг 2: Генерация кандидатов C_k (Join)
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                l1 = prev_frequent[i]
                l2 = prev_frequent[j]

                if l1[: k - 2] == l2[: k - 2] and l1[k - 2] < l2[k - 2]:
                    candidate = l1 + [l2[k - 2]]

                    # Шаг 3: Отсечение (Prune) по принципу Apriori
                    have_infrequent_subset = False
                    for idx in range(len(candidate)):
                        subset = candidate[:idx] + candidate[idx + 1 :]
                        if subset not in prev_frequent:
                            have_infrequent_subset = True
                            break

                    if not have_infrequent_subset:
                        if candidate not in candidates:
                            candidates.append(candidate)

        if not candidates:
            break

        candidate_counts = {tuple(c): 0 for c in candidates}
        for t in trans_sets:
            for c_tuple in candidate_counts:
                if set(c_tuple).issubset(t):
                    candidate_counts[c_tuple] += 1

        # Шаг 4: Формирование Lk
        Lk = []
        for c_tuple, count in candidate_counts.items():
            sup = count / n_trans
            if sup >= min_support:
                Lk.append((list(c_tuple), count))
                support_dict[tuple(sorted(c_tuple))] = sup

        if not Lk:
            break

        L.append(Lk)
        k += 1

    results = []
    for level in L:
        for itemset, count in level:
            results.append((set(itemset), count / n_trans))

    if sort_by == 'support_desc':
        results.sort(key=lambda x: (-x[1], sorted(list(x[0]))))
    elif sort_by == 'lexicographic':
        results.sort(key=lambda x: sorted(list(x[0])))

    return results, support_dict


# 3. Поиск ассоциативных правил


def generate_association_rules(
    frequent_itemsets,
    support_dict,
    min_confidence=0.70,
    sort_by='support_desc',
):
    """
    Генерирует ассоциативные правила вида A -> B из ранее найденных частых наборов.

    Входные параметры:
        frequent_itemsets (list of tuple): Список частых наборов вида (множество_товаров, поддержка).
        support_dict (dict): Словарь значений поддержки формата {tuple: float}.
        min_confidence (float): Минимальный порог достоверности (от 0.0 до 1.0).
        sort_by (str): Критерий сортировки правил ('support_desc' или 'lexicographic').

    Возвращаемое значение:
        rules (list of dict): Список словарей с характеристиками правил (antecedent, consequent, support, confidence, total_size).
    """
    rules = []

    for itemset, sup_itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue

        itemset_set = set(itemset)
        itemset_list = sorted(list(itemset))

        for r in range(1, len(itemset_list)):
            for antecedent_tuple in itertools.combinations(itemset_list, r):
                antecedent_set = set(antecedent_tuple)
                consequent_set = itemset_set - antecedent_set

                ant_key = tuple(sorted(antecedent_tuple))
                sup_antecedent = support_dict.get(ant_key, 0.0)

                if sup_antecedent > 0:
                    confidence = sup_itemset / sup_antecedent
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': antecedent_set,
                            'consequent': consequent_set,
                            'support': sup_itemset,
                            'confidence': confidence,
                            'total_size': len(itemset),
                        })

    if sort_by == 'support_desc':
        rules.sort(
            key=lambda r: (
                -r['support'],
                -r['confidence'],
                sorted(list(r['antecedent'])),
            )
        )
    elif sort_by == 'lexicographic':
        rules.sort(
            key=lambda r: (
                sorted(list(r['antecedent'])),
                sorted(list(r['consequent'])),
            )
        )

    return rules


# 4. Эксперименты с правилами при фиксированной поддержке


def run_rule_experiments(
    file_path,
    fixed_support=0.01,
    conf_thresholds=None,
    sort_by='support_desc',
):
    """
    Запускает эксперименты по генерации ассоциативных правил при фиксированном пороге поддержки
    и различных порогах достоверности, сохраняет результаты в текстовый файл results.txt.

    Входные параметры:
        file_path (str): Путь к файлу с данными транзакций.
        fixed_support (float): Зафиксированный порог поддержки (от 0.0 до 1.0).
        conf_thresholds (list): Список порогов минимальной достоверности для тестирования.
        sort_by (str): Критерий сортировки результатов ('support_desc' или 'lexicographic').

    Возвращаемое значение:
        experiment_results (list of dict): Список словарей, содержащих результаты и метрики эксперимента.
    """
    if conf_thresholds is None:
        conf_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"Загрузка данных из файла {file_path}...")
    transactions = load_from_csv(file_path)
    print(f"Всего транзакций загружено: {len(transactions)}\n")

    print(
        f"--- Поиск частых наборов при фиксированном min_support = {fixed_support * 100}% ---"
    )
    frequent_itemsets, support_dict = apriori(
        transactions, min_support=fixed_support, sort_by=sort_by
    )
    print(f"Найдено частых наборов: {len(frequent_itemsets)}\n")

    experiment_results = []

    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("=== РЕЗУЛЬТАТЫ ПОИСКА АССОЦИАТИВНЫХ ПРАВИЛ ===\n")
        f.write(
            f"Фиксированный порог поддержки (min_support): {fixed_support * 100}%\n\n"
        )

        for conf in conf_thresholds:
            print(
                f"Запуск генерации правил для min_confidence = {conf * 100:.0f}%..."
            )
            start_time = time.time()

            rules = generate_association_rules(
                frequent_itemsets,
                support_dict,
                min_confidence=conf,
                sort_by=sort_by,
            )

            elapsed_time = time.time() - start_time

            experiment_results.append({
                'confidence_thresh': conf,
                'confidence_pct': f"{int(round(conf * 100))}%",
                'time_sec': elapsed_time,
                'total_rules': len(rules),
                'rules': rules,
            })

            f.write(
                f"Порог достоверности (min_confidence): {conf * 100:.0f}%\n"
            )
            f.write(f"Время генерации: {elapsed_time:.4f} сек.\n")
            f.write(f"Найдено правил: {len(rules)}\n")
            f.write("Найденные правила (антецедент -> консеквент):\n")

            for r in rules:
                ant_str = ", ".join(sorted(list(r['antecedent'])))
                cons_str = ", ".join(sorted(list(r['consequent'])))
                f.write(
                    f"  {{{ant_str}}} -> {{{cons_str}}} | поддержка: {r['support']:.4f}, достоверность: {r['confidence']:.4f}\n"
                )

            f.write("-" * 50 + "\n\n")

            print(
                f"  -> Найдено правил: {len(rules)}, Время: {elapsed_time:.4f} сек."
            )

    print("\nВсе результаты успешно сохранены в файл 'results.txt'.")
    return experiment_results


# 5. Визуализация результатов правил


def visualize_rule_results(experiment_results, fixed_support):
    """
    Визуализирует быстродействие генерации правил и общее количество найденных правил.

    Входные параметры:
        experiment_results (list of dict): Результаты экспериментов, полученные функцией run_rule_experiments.
        fixed_support (float): Зафиксированный порог поддержки.

    Возвращаемое значение:
        None (функция выводит графики на экран и сохраняет их на диск).
    """
    conf_labels = [res['confidence_pct'] for res in experiment_results]
    times = [res['time_sec'] for res in experiment_results]
    rule_counts = [res['total_rules'] for res in experiment_results]

    # График 1: Быстродействие генерации правил
    plt.figure(figsize=(8, 6))
    plt.plot(
        conf_labels, times, marker='s', color='g', linewidth=2, markersize=8
    )
    plt.title(
        f'Быстродействие поиска правил (min_support = {fixed_support * 100:.1f}%)',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Порог достоверности (min_confidence)', fontsize=10)
    plt.ylabel('Время выполнения (сек.)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(times):
        plt.annotate(
            f"{txt:.4f}с",
            (conf_labels[i], times[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
        )

    plt.tight_layout()
    plt.savefig('rules_performance.png', dpi=300)
    plt.show()
    print("График быстродействия правил сохранен в файл 'rules_performance.png'.")

    # График 2: Общее количество найденных правил
    plt.figure(figsize=(8, 6))
    plt.bar(
        conf_labels, rule_counts, color='orange', width=0.4, edgecolor='black'
    )
    plt.title(
        f'Количество правил при различных порогах достоверности\n(min_support = {fixed_support * 100:.1f}%)',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Порог достоверности (min_confidence)', fontsize=10)
    plt.ylabel('Количество правил', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    for i, count in enumerate(rule_counts):
        plt.text(
            i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig('rules_count.png', dpi=300)
    plt.show()
    print("График количества правил сохранен в файл 'rules_count.png'.")


# 6. Фильтрация разумных правил


def print_filtered_rules(experiment_results, target_conf=0.5, max_size=7):
    """
    Выводит на экран список правил с суммарным количеством объектов не превышающим max_size.

    Входные параметры:
        experiment_results (list of dict): Результаты экспериментов по правилам.
        target_conf (float): Выбранный порог достоверности для отображения.
        max_size (int): Максимально допустимое суммарное число объектов в правиле (|A U B|).

    Возвращаемое значение:
        None (функция выводит список правил в консоль).
    """
    print(
        f"\n=== Список разумных правил (Достоверность >= {target_conf * 100:.0f}%, Объектов <= {max_size}) ==="
    )

    target_exp = next(
        (
            res
            for res in experiment_results
            if abs(res['confidence_thresh'] - target_conf) < 1e-5
        ),
        None,
    )

    if not target_exp or not target_exp['rules']:
        print("Правила не найдены.")
        return

    filtered_rules = [
        r for r in target_exp['rules'] if r['total_size'] <= max_size
    ]

    for idx, r in enumerate(filtered_rules, 1):
        ant_str = ", ".join(sorted(list(r['antecedent'])))
        cons_str = ", ".join(sorted(list(r['consequent'])))
        print(
            f"{idx:2d}. {{{ant_str}}} -> {{{cons_str}}} | Поддержка: {r['support']:.4f} ({r['support']*100:.2f}%), Достоверность: {r['confidence']:.4f} ({r['confidence']*100:.2f}%)"
        )


try:
    fixed_supp = 0.01
    conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    rule_results = run_rule_experiments(
        file_path, fixed_support=fixed_supp, conf_thresholds=conf_thresholds
    )
    visualize_rule_results(rule_results, fixed_support=fixed_supp)
    print_filtered_rules(rule_results, target_conf=0.5, max_size=7)

except FileNotFoundError:
    print(f"Файл '{file_path}' не найден. Проверьте правильность пути.")