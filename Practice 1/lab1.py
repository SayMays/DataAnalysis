import time
import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных

file_path = "baskets.csv"

def load_from_csv(file_path):
    """
    Загрузка транзакций из CSV-файла с кодировкой Windows-1251,
    где каждая строка представляет собой список товаров, разделенных запятыми.
    """
    df = pd.read_csv(file_path, header=None, encoding='cp1251')

    transactions = []
    for _, row in df.iterrows():
        # Сбор всех непустых значений из всех колонок текущей строки в один список
        items = [str(val).strip() for val in row.dropna() if str(val).strip() and str(val).strip() != 'nan']
        if items:
            transactions.append(items)

    return transactions


# 2. Реализация алгоритма Apriori

def apriori(transactions, min_support=0.05, sort_by='support_desc'):
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

    while True:
        # Извлечение списков элементов из предыдущего уровня L[k-2]
        prev_frequent = [itemset for itemset, count in L[k - 2]]
        candidates = []

        # Шаг 2: Генерация кандидатов C_k (Join)
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                l1 = prev_frequent[i]
                l2 = prev_frequent[j]

                # Условие соединения: первые k-2 элементов равны
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

        # Подсчет поддержки кандидатов
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

    # Сбор результатов
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

def run_experiments(file_path, thresholds=[0.01, 0.03, 0.05, 0.10, 0.15]):
    print(f"Загрузка данных из файла {file_path}...")
    transactions = load_from_csv(file_path)
    print(f"Всего транзакций загружено: {len(transactions)}\n")

    experiment_results = []

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

        print(
            f"  -> Найдено наборов: {len(frequent_itemsets)}, Макс. длина: {max_length}, Время: {elapsed_time:.4f} сек.")

    return experiment_results

# 4. Визуализация
def visualize_results(experiment_results):
    thresholds_str = [res['threshold_pct'] for res in experiment_results]
    times = [res['time_sec'] for res in experiment_results]

    # График 1: Быстродействие алгоритма
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

    # График 2: Количество наборов разной длины
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