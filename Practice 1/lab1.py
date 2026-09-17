"""
Название файла: lab1.py
Семантика файла: Программа предназначена для поиска частых наборов товаров
в транзакционных данных с использованием алгоритма Apriori, проведения
вычислительных экспериментов при различных порогах поддержки, сохранения результатов
в текстовый файл и визуализации.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt

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
        results(list of tuple): Список найденных частых наборов, где каждый элемент содержит
        множество товаров (set) и значение поддержки (float) (список кортежей).
    """
    trans_sets = [set(t) for t in transactions]
    n_trans = len(trans_sets)
    if n_trans == 0:
        return []

    # Шаг 1: Нахождение частых 1-элементных наборов
    item_counts = {}
    for t in trans_sets:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    L1 = []
    for item, count in item_counts.items():
        if (count / n_trans) >= min_support:
            L1.append(([item], count))

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

                if l1[:k - 2] == l2[:k - 2] and l1[k - 2] < l2[k - 2]:
                    candidate = l1 + [l2[k - 2]]

                    # Шаг 3: Отсечение (Prune) по принципу Apriori
                    have_infrequent_subset = False
                    for idx in range(len(candidate)):
                        subset = candidate[:idx] + candidate[idx + 1:]
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

        # Шаг 4: Формирование Lk (list)
        Lk = []
        for c_tuple, count in candidate_counts.items():
            if (count / n_trans) >= min_support:
                Lk.append((list(c_tuple), count))

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

    return results


# 3. Эксперименты и итоговые результаты

def run_experiments(file_path, thresholds=None):
    """
    Запускает эксперименты алгоритма Apriori для различных порогов поддержки,
    выводит статистику и сохраняет все найденные наборы в текстовый файл results.txt.

    Входные параметры:
        file_path (str): Путь к файлу с данными транзакций.
        thresholds (list): Список порогов минимальной поддержки для тестирования.

    Возвращаемое значение:
        experiment_results (list of dict): Список словарей, содержащих метрики каждого
        эксперимента (время, число наборов, максимальная длина и прочее).
    """
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]

    print(f"Загрузка данных из файла {file_path}...")
    transactions = load_from_csv(file_path)
    print(f"Всего транзакций загружено: {len(transactions)}\n")

    experiment_results = []

    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ АЛГОРИТМА APRIORI ===\n\n")

        for thresh in thresholds:
            print(f"Запуск эксперимента для min_support = {thresh * 100}%...")
            start_time = time.time()

            frequent_itemsets = apriori(transactions, min_support=thresh, sort_by='support_desc')

            elapsed_time = time.time() - start_time

            length_counts = {}
            for itemset, sup in frequent_itemsets:
                length = len(itemset)
                length_counts[length] = length_counts.get(length, 0) + 1

            max_length = max(length_counts.keys()) if length_counts else 0

            experiment_results.append({
                'threshold': thresh,
                'threshold_pct': f"{int(thresh * 100)}%",
                'time_sec': elapsed_time,
                'total_itemsets': len(frequent_itemsets),
                'max_length': max_length,
                'length_counts': length_counts,
                'itemsets': frequent_itemsets
            })

            f.write(f"Порог поддержки (min_support): {thresh * 100}%\n")
            f.write(f"Время выполнения: {elapsed_time:.4f} сек.\n")
            f.write(f"Всего найдено частых наборов: {len(frequent_itemsets)}\n")
            f.write("Найденные наборы (набор товаров: поддержка):\n")

            for itemset, support in frequent_itemsets:
                items_str = ", ".join(itemset)
                f.write(f"  {{{items_str}}} -> поддержка: {support:.4f} ({support * 100:.2f}%)\n")

            f.write("-" * 50 + "\n\n")

            print(
                f"  -> Найдено наборов: {len(frequent_itemsets)}, Макс. длина: {max_length}, Время: {elapsed_time:.4f} сек.")

    print("\nВсе результаты успешно сохранены в файл 'results.txt'.")
    return experiment_results


# 4. Визуализация
def visualize_results(experiment_results):
    """
    Визуализирует результаты экспериментов и сохраняет графики в файлы формата .png.

    Входные параметры:
        experiment_results (list of dict): Результаты экспериментов, полученные функцией run_experiments (список словарей).

    Возвращаемое значение:
        None (функция выводит графики на экран и сохраняет их на диск).
    """
    thresholds_str = [res['threshold_pct'] for res in experiment_results]
    times = [res['time_sec'] for res in experiment_results]

    # График 1: Быстродействие алгоритма при изменяемом пороге поддержки
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_str, times, marker='o', color='b', linewidth=2, markersize=8)
    plt.title('Быстродействие алгоритма Apriori', fontsize=12, fontweight='bold')
    plt.xlabel('Порог поддержки (min_support)', fontsize=10)
    plt.ylabel('Время выполнения (сек.)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(times):
        plt.annotate(f"{txt:.2f}с", (thresholds_str[i], times[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.tight_layout()
    plt.savefig('apriori_performance.png', dpi=300)
    plt.show()
    print("График быстродействия сохранен в файл 'apriori_performance.png'.")

    # График 2: Количество наборов разной длины при изменяемом пороге поддержки
    all_lengths = sorted(list(set(l for res in experiment_results for l in res['length_counts'].keys())))

    bar_data = {length: [] for length in all_lengths}
    for res in experiment_results:
        for length in all_lengths:
            bar_data[length].append(res['length_counts'].get(length, 0))

    x = range(len(thresholds_str))
    width = 0.15
    multiplier = 0

    plt.figure(figsize=(8, 6))
    for length, counts in bar_data.items():
        offset = width * multiplier
        plt.bar([i + offset for i in x], counts, width, label=f'Длина {length}')
        multiplier += 1

    plt.title('Количество частых наборов различной длины', fontsize=12, fontweight='bold')
    plt.xlabel('Порог поддержки (min_support)', fontsize=10)
    plt.ylabel('Количество наборов', fontsize=10)
    plt.xticks([i + width * (len(all_lengths) - 1) / 2 for i in x], thresholds_str)
    plt.legend(title='Длина набора')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('apriori_itemsets_distribution.png', dpi=300)
    plt.show()
    print("График распределения длин сохранен в файл 'apriori_itemsets_distribution.png'.")


try:
    results = run_experiments(file_path, thresholds=[0.01, 0.03, 0.05, 0.10, 0.15])
    visualize_results(results)
except FileNotFoundError:
    print(f"Файл '{file_path}' не найден. Проверьте правильность пути.")