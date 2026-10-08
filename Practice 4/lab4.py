"""
Название файла: lab4.py
Семантика файла: Программа предназначена для классификации набора данных Census Income
с помощью ансамблевых техник: бэггинг, случайный лес и бустинг.
Базовый классификатор - дерево решений из lab3.py (критерии Information gain, Gain ratio, Gini index).
Для каждой из трёх техник проводится эксперимент с числом участников от 50 до 100 с шагом 10,
строится график показателей качества (accuracy, precision, recall, F-мера) в зависимости от числа участников,
на который нанесены значения для одиночного дерева решений из задания lab3.py.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

LAB3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Practice 3")
sys.path.append(LAB3_DIR)

import lab3

# Параметры программы
TRAIN_FILE = os.path.join(LAB3_DIR, "adult.data.txt")  # файл обучающей части набора данных
TEST_FILE = os.path.join(LAB3_DIR, "adult.test.txt")  # файл тестовой части набора данных
TECHNIQUES = ["bagging", "random_forest", "boosting"]  # ансамблевые техники для эксперимента
TECH_TITLES = {"bagging": "Бэггинг", "random_forest": "Случайный лес", "boosting": "Бустинг (AdaBoost)"}
N_ESTIMATORS_LIST = list(range(50, 101, 10))  # число участников ансамбля в эксперименте
CRITERION = "gini"  # критерий выбора атрибута: 'info_gain', 'gain_ratio' или 'gini'
TRAIN_RATIO = 0.8  # доля обучающей выборки
MAX_DEPTH = None  # глубина деревьев (None - по умолчанию: lab3.MAX_DEPTH, для бустинга BOOST_DEPTH)
BOOST_DEPTH = 3  # глубина слабых деревьев в бустинге
SAMPLE_RATIO = 1.0  # размер бутстрэп-выборки относительно обучающей (бэггинг, случайный лес)
MAX_FEATURES = None  # число случайных атрибутов в узле (случайный лес; None - sqrt(14))
SEED = lab3.SEED  # начальное значение генератора случайных чисел


# 1. Построение базового классификатора

def build_random_tree(data, idx, attrs, criterion, rng, max_features=None, depth=0,
                      max_depth=lab3.MAX_DEPTH, min_samples_split=2):
    """
    Строит дерево решений, совместимое с деревьями lab3.py (тот же формат узла, поэтому
    можно использовать lab3.predict и lab3.count_nodes). Отличие от lab3.build_tree -
    в каждом узле для поиска разбиения может использоваться случайное подмножество
    из max_features атрибутов (случайный лес). При max_features = None рассматриваются
    все доступные атрибуты (обычное дерево - для бэггинга и бустинга). Если в выбранном
    подмножестве полезного разбиения нет, поиск повторяется по всем доступным атрибутам.

    Входные параметры:
        data (dict): Закодированный набор данных.
        idx (numpy.ndarray): Индексы обучающих объектов узла (могут повторяться).
        attrs (list): Доступные для разбиения атрибуты.
        criterion (str): Критерий выбора атрибута.
        rng (numpy.random.RandomState): Генератор случайных чисел.
        max_features (int | None): Число случайно выбираемых атрибутов в узле.
        depth (int): Текущая глубина узла.
        max_depth (int | None): Максимальная глубина дерева.
        min_samples_split (int): Минимальное число объектов для разбиения узла.

    Возвращаемое значение:
        node (dict): Корень построенного (под)дерева.
    """

    ys = data["y"][idx]
    counts = np.bincount(ys, minlength=len(lab3.CLASSES))
    node = {"n": len(idx), "counts": counts.tolist(), "pred": int(np.argmax(counts)),
            "attr": None, "thr": None, "values": None, "children": []}
    if (counts.max() == len(idx) or len(idx) < min_samples_split or not attrs
            or (max_depth is not None and depth >= max_depth)):
        return node

    res = None
    if max_features is not None and max_features < len(attrs):
        subset = [int(a) for a in rng.choice(attrs, size=max_features, replace=False)]
        res = lab3.best_split(data, idx, subset, criterion)
    if res is None:
        res = lab3.best_split(data, idx, attrs, criterion)
    if res is None:
        return node

    j, thr, _ = res
    node["attr"], node["thr"] = j, thr
    x = data["cols"][j][idx]
    if thr is not None:
        mask = x <= thr
        for sub in (idx[mask], idx[~mask]):
            node["children"].append(build_random_tree(
                data, sub, attrs, criterion, rng, max_features, depth + 1, max_depth, min_samples_split))
    else:
        rest = [a for a in attrs if a != j]
        node["values"] = [int(v) for v in np.unique(x)]
        for v in node["values"]:
            node["children"].append(build_random_tree(
                data, idx[x == v], rest, criterion, rng, max_features, depth + 1, max_depth,
                min_samples_split))
    return node


# 2. Ансамблевые техники

def train_bagging(data, train_idx, n_estimators, criterion, max_depth, sample_ratio, rng,
                  max_features=None, verbose=True):
    """
    Обучает бэггинг (при max_features != None - случайный лес): каждое дерево строится на
    бутстрэп-выборке (выборка с возвращением) размера sample_ratio * |train|.

    Входные параметры:
        data (dict): Закодированный набор данных.
        train_idx (numpy.ndarray): Индексы обучающей выборки.
        n_estimators (int): Число деревьев в ансамбле.
        criterion (str): Критерий выбора атрибута разбиения.
        max_depth (int | None): Максимальная глубина деревьев.
        sample_ratio (float): Размер бутстрэп-выборки относительно обучающей выборки.
        rng (numpy.random.RandomState): Генератор случайных чисел.
        max_features (int | None): Число случайных атрибутов в узле (для случайного леса).
        verbose (bool): Печатать ли ход обучения.

    Возвращаемое значение:
        members (list): Список пар (дерево, вес голоса); у всех деревьев вес 1.
    """

    attrs = list(range(len(data["cols"])))
    size = max(1, int(round(len(train_idx) * sample_ratio)))
    members = []
    for t in range(n_estimators):
        sample = train_idx[rng.randint(0, len(train_idx), size)]
        tree = build_random_tree(data, sample, attrs, criterion, rng, max_features, 0, max_depth)
        members.append((tree, 1.0))
        if verbose and (t + 1) % 10 == 0:
            print("    обучено деревьев: %d/%d" % (t + 1, n_estimators), flush=True)
    return members


def train_boosting(data, train_idx, n_estimators, criterion, max_depth, rng, verbose=True):
    """
    Обучает бустинг AdaBoost (схема SAMME для двух классов) с взвешенной бутстрэп-выборкой:
    на каждом шаге обучающие объекты выбираются с возвращением с вероятностями, пропорциональными
    весам. Вес голоса дерева alpha = 0.5 * ln((1 - err) / err), где err - взвешенная ошибка
    дерева на всей обучающей выборке. Веса ошибочно классифицированных объектов увеличиваются,
    верно классифицированных - уменьшаются. Если err >= 0.5, дерево отбрасывается, а веса
    сбрасываются к равномерным.

    Входные параметры:
        data (dict): Закодированный набор данных.
        train_idx (numpy.ndarray): Индексы обучающей выборки.
        n_estimators (int): Число деревьев в ансамбле.
        criterion (str): Критерий выбора атрибута разбиения.
        max_depth (int | None): Максимальная глубина слабых деревьев.
        rng (numpy.random.RandomState): Генератор случайных чисел.
        verbose (bool): Печатать ли ход обучения.

    Возвращаемое значение:
        members (list): Список пар (дерево, вес голоса alpha).
    """

    attrs = list(range(len(data["cols"])))
    n = len(train_idx)
    y = data["y"][train_idx]
    w = np.full(n, 1.0 / n)
    members, attempts = [], 0
    while len(members) < n_estimators and attempts < 3 * n_estimators:
        attempts += 1
        sample = train_idx[rng.choice(n, size=n, p=w)]
        tree = build_random_tree(data, sample, attrs, criterion, rng, None, 0, max_depth)
        pred = lab3.predict(tree, data, train_idx)
        wrong = pred != y
        err = float(w[wrong].sum())
        if err >= 0.5:
            w = np.full(n, 1.0 / n)
            continue
        err = max(err, 1e-10)
        alpha = 0.5 * np.log((1.0 - err) / err)
        w = w * np.exp(np.where(wrong, alpha, -alpha))
        w /= w.sum()
        members.append((tree, float(alpha)))
        if verbose and len(members) % 10 == 0:
            print("    обучено деревьев: %d/%d" % (len(members), n_estimators), flush=True)
    return members


def train_ensemble(data, train_idx, technique, n_estimators, criterion, max_depth=None,
                   sample_ratio=1.0, max_features=None, seed=SEED, verbose=True):
    """
    Обучает ансамбль выбранной техники.

    Входные параметры:
        data (dict): Закодированный набор данных.
        train_idx (numpy.ndarray): Индексы обучающей выборки.
        technique (str): 'bagging', 'random_forest' или 'boosting'.
        n_estimators (int): Число участников ансамбля.
        criterion (str): Критерий выбора атрибута разбиения.
        max_depth (int | None): Глубина деревьев; None - значение по умолчанию для техники
            (lab3.MAX_DEPTH для бэггинга и случайного леса, BOOST_DEPTH для бустинга).
        sample_ratio (float): Размер бутстрэп-выборки (бэггинг, случайный лес).
        max_features (int | None): Число атрибутов в узле для случайного леса; None - sqrt(число атрибутов).
        seed (int): Начальное значение генератора случайных чисел.
        verbose (bool): Печатать ли ход обучения.

    Возвращаемое значение:
        members (list): Список пар (дерево, вес голоса).
    """

    rng = np.random.RandomState(seed)
    if technique == "boosting":
        depth = BOOST_DEPTH if max_depth is None else max_depth
        return train_boosting(data, train_idx, n_estimators, criterion, depth, rng, verbose)
    depth = lab3.MAX_DEPTH if max_depth is None else max_depth
    if technique == "random_forest":
        if max_features is None:
            max_features = max(1, int(round(np.sqrt(len(data["cols"])))))
        return train_bagging(data, train_idx, n_estimators, criterion, depth, sample_ratio, rng,
                             max_features, verbose)
    return train_bagging(data, train_idx, n_estimators, criterion, depth, sample_ratio, rng,
                         None, verbose)


# 3. Классификация и оценка качества

def ensemble_predict(members, data, idx, k=None):
    """
    Классифицирует объекты взвешенным голосованием первых k участников ансамбля.
    Для бэггинга и случайного леса веса равны 1 (обычное голосование большинством),
    для бустинга - alpha. Класс 1 выбирается, если сумма весов голосов за него больше
    суммы за класс 0 (при равенстве - класс 0).

    Входные параметры:
        members (list): Список пар (дерево, вес).
        data (dict): Закодированный набор данных.
        idx (numpy.ndarray): Индексы классифицируемых объектов.
        k (int | None): Число используемых участников (None - все).

    Возвращаемое значение:
        pred (numpy.ndarray): Предсказанные классы (0 или 1).
    """

    score = np.zeros(len(idx))
    for tree, alpha in members[:k]:
        score += alpha * (2 * lab3.predict(tree, data, idx) - 1)
    return (score > 0).astype(np.int64)


def evaluate_prefixes(members, data, idx, counts):
    """
    Считает показатели качества для ансамблей из первых k участников (k из counts).
    Предсказания каждого дерева вычисляются один раз, голоса накапливаются. Для всех
    рассматриваемых техник ансамбль из первых k деревьев совпадает с ансамблем,
    обученным с параметром n_estimators = k, поэтому достаточно обучить максимальный ансамбль.

    Входные параметры:
        members (list): Список пар (дерево, вес) максимального ансамбля.
        data (dict): Закодированный набор данных.
        idx (numpy.ndarray): Индексы тестовой выборки.
        counts (list): Размеры ансамблей для оценки.

    Возвращаемое значение:
        metrics (list): Список словарей показателей (accuracy, precision, recall, f1) для каждого k.
    """

    y_true = data["y"][idx]
    score = np.zeros(len(idx))
    result, wanted = [], set(counts)
    for i, (tree, alpha) in enumerate(members, start=1):
        score += alpha * (2 * lab3.predict(tree, data, idx) - 1)
        if i in wanted:
            result.append(lab3.compute_metrics(y_true, (score > 0).astype(np.int64)))
    return result


def run_ensemble(data, technique, counts, criterion, train_ratio=TRAIN_RATIO, max_depth=None,
                 sample_ratio=1.0, max_features=None, seed=SEED):
    """
    Выполняет эксперимент для одной техники: делит данные, обучает ансамбль максимального
    из требуемых размеров и оценивает качество для каждого размера из counts на тестовой части.

    Входные параметры:
        data (dict): Закодированный набор данных.
        technique (str): Ансамблевая техника.
        counts (list): Размеры ансамбля (число участников).
        criterion (str): Критерий выбора атрибута разбиения.
        train_ratio (float): Доля обучающей выборки.
        max_depth (int | None): Глубина деревьев.
        sample_ratio (float): Размер бутстрэп-выборки.
        max_features (int | None): Число атрибутов в узле (случайный лес).
        seed (int): Начальное значение генератора случайных чисел.

    Возвращаемое значение:
        metrics (list): Показатели качества для каждого размера из counts.
    """

    train_idx, test_idx = lab3.split_data(data, train_ratio, seed)
    members = train_ensemble(data, train_idx, technique, max(counts), criterion, max_depth,
                             sample_ratio, max_features, seed)
    return evaluate_prefixes(members, data, test_idx, sorted(counts))


# 4. Визуализация

def plot_ensemble_metrics(counts, results, baseline, criterion, train_ratio, path="ensemble_metrics.png"):
    """
    Строит график показателей качества (accuracy, precision, recall, F-мера; 2x2) в зависимости
    от числа участников ансамбля. Для каждой техники рисуется линия; пунктиром нанесено значение
    для одиночного дерева решений из задания lab3.py (то же разбиение и критерий).

    Входные параметры:
        counts (list): Числа участников ансамбля.
        results (dict): {техника: список словарей показателей по каждому числу участников}.
        baseline (dict): Показатели одиночного дерева решений (результат lab3.run_experiment).
        criterion (str): Критерий выбора атрибута.
        train_ratio (float): Доля обучающей выборки.
        path (str): Файл для сохранения рисунка.

    Возвращаемое значение:
        fig (matplotlib.figure.Figure): Построенная фигура.
    """

    names = [("accuracy", "Accuracy"), ("precision", "Precision"),
             ("recall", "Recall"), ("f1", "F-мера")]
    colors = {"bagging": "#4c78a8", "random_forest": "#54a24b", "boosting": "#f58518"}
    markers = {"bagging": "o", "random_forest": "s", "boosting": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.ravel(), names):
        for tech, res in results.items():
            ax.plot(counts, [m[key] for m in res], marker=markers[tech], color=colors[tech],
                    label=TECH_TITLES[tech])
        ax.axhline(baseline[key], color="red", linestyle="--",
                   label="Одно дерево (lab3): %.4f" % baseline[key])
        ax.set_title(title)
        ax.set_xlabel("Количество участников ансамбля")
        ax.set_ylabel("Значение показателя")
        ax.set_xticks(counts)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Показатели качества ансамблей (критерий: %s, обучающая выборка %d%%)"
                 % (lab3.CRITERIA_TITLES[criterion], round(train_ratio * 100)), fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


# 5. Точка входа в программу

def main():
    """
    Точка входа. Параметры берутся из констант в начале файла.
    Загружает и объединяет файлы набора данных, затем для каждой ансамблевой техники проводит
    эксперимент с числом участников 50..100 с шагом 10, печатает таблицы показателей и строит
    график с нанесёнными значениями одиночного дерева решений из lab3.py.

    Входные параметры:
        Нет.

    Возвращаемое значение:
        None.
    """

    data = lab3.encode_dataset(lab3.load_data(TRAIN_FILE) + lab3.load_data(TEST_FILE))
    print("Загружено записей: %d" % len(data["y"]))

    _, base = lab3.run_experiment(data, CRITERION, TRAIN_RATIO, seed=SEED)
    print("\nОдно дерево (lab3, %s): accuracy=%.4f precision=%.4f recall=%.4f F1=%.4f"
          % (lab3.CRITERIA_TITLES[CRITERION], base["accuracy"], base["precision"],
             base["recall"], base["f1"]))
    results = {}
    for tech in TECHNIQUES:
        print("\nТехника: %s" % TECH_TITLES[tech], flush=True)
        results[tech] = run_ensemble(data, tech, N_ESTIMATORS_LIST, CRITERION, TRAIN_RATIO,
                                     MAX_DEPTH, SAMPLE_RATIO, MAX_FEATURES, SEED)
        print("%8s %9s %10s %8s %8s" % ("\nдеревьев", "accuracy", "precision", "recall", "F1"))
        for n, m in zip(N_ESTIMATORS_LIST, results[tech]):
            print("%8d %9.4f %10.4f %8.4f %8.4f" % (n, m["accuracy"], m["precision"], m["recall"], m["f1"]))
    plot_ensemble_metrics(N_ESTIMATORS_LIST, results, base, CRITERION, TRAIN_RATIO)
    plt.show()


if __name__ == "__main__":
    main()