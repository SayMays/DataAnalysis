"""
Название файла: lab3.py
Семантика файла: Программа предназначена для классификации набора данных Census Income
с помощью дерева решений. Поддерживаются три критерия выбора атрибута разбиения: Information gain,
Gain ratio, Gini index. Числовые атрибуты разбиваются по порогу на две ветви (x <= t и x > t),
категориальные - многосторонне (по одной ветви на каждое значение атрибута).
Программа строит три дерева (по одному на критерий) на 100% данных и рисует
каждое на отдельном графике (на графике показаны не все узлы - только верхние уровни и
наиболее крупные ветви), а также для всех трёх критериев варьирует долю обучающей выборки
(60%, 70%, 80%, 90%), считает accuracy, precision, recall, F-меру и строит график зависимости
показателей от доли.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Параметры программы
TRAIN_FILE = "adult.data.txt"  # файл обучающей выборки
TEST_FILE = "adult.test.txt"  # файл тестовой выборки
MAX_DEPTH = 10  # максимальная глубина строящегося дерева
VIZ_DEPTH = 3  # глубина отображения деревьев на графиках
MAX_VIZ_CHILDREN = 3  # максимум ветвей у категориального узла, показываемых на графике
SEED = 42  # начальное значение генератора случайных чисел
RATIOS = [0.6, 0.7, 0.8, 0.9]  # доли обучающей выборки во втором эксперименте

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]

NUMERIC = {"age", "fnlwgt", "education-num", "capital-gain",
           "capital-loss", "hours-per-week"}
CLASSES = ["<=50K", ">50K"]

CRITERIA = ["info_gain", "gain_ratio", "gini"]

CRITERIA_TITLES = {"info_gain": "Information gain",
                   "gain_ratio": "Gain ratio",
                   "gini": "Gini index"}


# 1. Загрузка данных

def load_data(file_path):
    """
    Загружает записи набора Census Income из txt-файла (без заголовка, разделитель - запятая).
    Пропускает пустые строки, служебные строки, строки с неверным числом полей.
    Удаляет пробелы вокруг значений и завершающую точку в метке класса (в adult.test метки имеют вид '<=50K.').

    Входные параметры:
        file_path (str): Путь к CSV-файлу.

    Возвращаемое значение:
        rows (list): Список записей, каждая запись - список из 15 строк (14 атрибутов + класс).
    """

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for rec in csv.reader(f):
            if not rec or rec[0].strip().startswith("|"):
                continue
            rec = [v.strip() for v in rec]
            if len(rec) != len(COLUMNS):
                continue
            rec[-1] = rec[-1].rstrip(".")
            rows.append(rec)
    return rows


# 2. Предобработка данных

def encode_dataset(rows):
    """
    Преобразует строковые записи в числовые массивы NumPy. Числовые атрибуты переводятся
    в float, категориальные кодируются целыми числами (значение '?' рассматривается как
    обычное значение категории). Метка класса кодируется: '<=50K' -> 0, '>50K' -> 1.

    Входные параметры:
        rows (list): Список записей (результат load_data).

    Возвращаемое значение:
        data (dict): Словарь с ключами:
            'cols' - список массивов по столбцам-атрибутам (14 шт.),
            'y' - массив меток классов (int),
            'names' - названия атрибутов,
            'is_num' - список флагов "атрибут числовой",
            'cats' - для категориальных атрибутов список значений, для числовых None.
    """

    n_attr = len(COLUMNS) - 1
    cols, is_num, cats = [], [], []
    for j in range(n_attr):
        raw = [r[j] for r in rows]
        if COLUMNS[j] in NUMERIC:
            cols.append(np.array(raw, dtype=float))
            is_num.append(True)
            cats.append(None)
        else:
            values = sorted(set(raw))
            index = {v: i for i, v in enumerate(values)}
            cols.append(np.array([index[v] for v in raw], dtype=np.int64))
            is_num.append(False)
            cats.append(values)
    y = np.array([CLASSES.index(r[-1]) for r in rows], dtype=np.int64)
    return {"cols": cols, "y": y, "names": COLUMNS[:-1], "is_num": is_num, "cats": cats}


def split_data(data, train_ratio, seed=SEED):
    """
    Случайно разделяет набор данных на обучающую и тестовую выборки.

    Входные параметры:
        data (dict): Закодированный набор данных (результат encode_dataset).
        train_ratio (float): Доля обучающей выборки от общего размера набора (0 < r <= 1).
        seed (int): Начальное значение генератора случайных чисел.

    Возвращаемое значение:
        (train_idx, test_idx) (tuple): Массивы индексов обучающей и тестовой выборок.
        При train_ratio = 1.0 тестовая выборка пуста.
    """
    n = len(data["y"])
    perm = np.random.RandomState(seed).permutation(n)
    cut = int(round(n * train_ratio))
    return perm[:cut], perm[cut:]


# 3. Критерии выбора атрибута разбиения

def impurity(counts, criterion):
    """
    Вычисляет меру неоднородности (по строкам матрицы) для заданного критерия:
    энтропия (для Information gain и Gain ratio) или индекс Джини (для Gini index).

    Входные параметры:
        counts (numpy.ndarray): Матрица (m x k) количеств объектов k классов в m узлах.
        criterion (str): 'info_gain', 'gain_ratio' или 'gini'.

    Возвращаемое значение:
        imp (numpy.ndarray): Вектор длины m со значениями неоднородности.
    """

    n = counts.sum(axis=1)
    p = counts / np.maximum(n, 1)[:, None]
    if criterion == "gini":
        return 1.0 - (p ** 2).sum(axis=1)
    return -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1)


def split_info(sizes):
    """
    Вычисляет информацию о разбиении (Split Info) - энтропию распределения объектов по
    ветвям. Используется в знаменателе Gain ratio.

    Входные параметры:
        sizes (numpy.ndarray): Матрица (m x b) размеров b ветвей для m вариантов разбиения.

    Возвращаемое значение:
        si (numpy.ndarray): Вектор длины m со значениями Split Info.
    """

    p = sizes / sizes.sum(axis=1, keepdims=True)
    return -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1)


def best_split(data, idx, attrs, criterion):
    """
    Выбирает лучший атрибут для разбиения узла. Для числового атрибута перебираются все пороги
    между соседними различными значениями (бинарное разбиение x <= t / x > t, порог выбирается
    векторизованно через кумулятивные суммы). Для категориального атрибута разбиение
    многостороннее: по одной ветви на каждое значение, встретившееся в узле. Для Gain ratio
    применяется схема C4.5: среди атрибутов с приростом не ниже среднего выбирается максимум
    отношения прироста к Split Info.

    Входные параметры:
        data (dict): Закодированный набор данных.
        idx (numpy.ndarray): Индексы объектов, попавших в узел.
        attrs (list): Номера атрибутов, доступных для разбиения.
        criterion (str): Критерий выбора атрибута.

    Возвращаемое значение:
        best (tuple | None): (номер атрибута, порог для числового атрибута или None для
        категориального, прирост) либо None, если полезного разбиения не найдено.
    """

    ys = data["y"][idx]
    n = len(idx)
    k = len(CLASSES)
    total = np.bincount(ys, minlength=k).astype(float)
    parent = impurity(total[None, :], criterion)[0]
    cands = []  # (attr, threshold, gain, split_info)

    for j in attrs:
        x = data["cols"][j][idx]
        if data["is_num"][j]:
            order = np.argsort(x, kind="stable")
            xs, yo = x[order], ys[order]
            valid = xs[:-1] != xs[1:]
            if not valid.any():
                continue
            onehot = np.zeros((n, k))
            onehot[np.arange(n), yo] = 1.0
            left = np.cumsum(onehot, axis=0)[:-1][valid]
            right = total - left
            nl = left.sum(axis=1)
            nr = n - nl
            child = (nl * impurity(left, criterion) + nr * impurity(right, criterion)) / n
            gains = parent - child
            b = int(np.argmax(gains))
            si = split_info(np.stack([nl, nr], axis=1))[b]
            thr = (xs[:-1][valid][b] + xs[1:][valid][b]) / 2.0
            cands.append((j, thr, gains[b], si))
        else:
            c = len(data["cats"][j])
            counts = np.bincount(x * k + ys, minlength=c * k).reshape(c, k).astype(float)
            sizes = counts.sum(axis=1)
            present = sizes > 0
            if present.sum() < 2:
                continue
            counts, sizes = counts[present], sizes[present]
            child = (sizes * impurity(counts, criterion)).sum() / n
            si = split_info(sizes[None, :])[0]
            cands.append((j, None, parent - child, si))

    cands = [c for c in cands if c[2] > 1e-12]
    if not cands:
        return None
    if criterion == "gain_ratio":
        avg = np.mean([c[2] for c in cands])
        good = [c for c in cands if c[2] >= avg - 1e-12 and c[3] > 1e-12]
        if not good:
            good = [c for c in cands if c[3] > 1e-12]
        if not good:
            return None
        best = max(good, key=lambda c: c[2] / c[3])
    else:
        best = max(cands, key=lambda c: c[2])
    return best[0], best[1], best[2]


# 4. Построение дерева и классификация

def build_tree(data, idx, attrs, criterion, depth=0, max_depth=MAX_DEPTH, min_samples_split=2):
    """
    Рекурсивно строит дерево решений. Узел - словарь: 'n' (число объектов), 'counts'
    (распределение по классам), 'pred' (мажоритарный класс). Внутренний узел дополнительно имеет
    'attr' (атрибут), 'thr' (порог для числового атрибута, иначе None), 'values' (коды значений
    категориального атрибута, по одному на каждого потомка; для числового None) и 'children'
    (список потомков). У числового атрибута два потомка: [x <= thr, x > thr]; у категориального -
    по одному потомку на каждое значение, встретившееся в узле. Использованный категориальный
    атрибут исключается из доступных на данной ветви. Рост прекращается, если узел чистый,
    достигнута максимальная глубина, объектов слишком мало или нет полезного разбиения.

    Входные параметры:
        data (dict): Закодированный набор данных.
        idx (numpy.ndarray): Индексы обучающих объектов узла.
        attrs (list): Доступные для разбиения атрибуты.
        criterion (str): Критерий выбора атрибута.
        depth (int): Текущая глубина узла.
        max_depth (int | None): Максимальная глубина дерева (None - без ограничения).
        min_samples_split (int): Минимальное число объектов для разбиения узла.

    Возвращаемое значение:
        node (dict): Корень построенного (под)дерева.
    """

    ys = data["y"][idx]
    counts = np.bincount(ys, minlength=len(CLASSES))
    node = {"n": len(idx), "counts": counts.tolist(), "pred": int(np.argmax(counts)),
            "attr": None, "thr": None, "values": None, "children": []}
    if (counts.max() == len(idx) or len(idx) < min_samples_split or not attrs
            or (max_depth is not None and depth >= max_depth)):
        return node
    res = best_split(data, idx, attrs, criterion)
    if res is None:
        return node
    j, thr, _ = res
    node["attr"], node["thr"] = j, thr
    x = data["cols"][j][idx]
    if thr is not None:
        mask = x <= thr
        for sub in (idx[mask], idx[~mask]):
            node["children"].append(
                build_tree(data, sub, attrs, criterion, depth + 1, max_depth, min_samples_split))
    else:
        rest = [a for a in attrs if a != j]
        node["values"] = [int(v) for v in np.unique(x)]
        for v in node["values"]:
            node["children"].append(
                build_tree(data, idx[x == v], rest, criterion, depth + 1, max_depth, min_samples_split))
    return node


def predict(tree, data, idx):
    """
    Классифицирует объекты с помощью дерева решений. В числовом узле объект направляется
    в ветвь x <= thr или x > thr, в категориальном - в ветвь, соответствующую его значению.
    Если значение категориального атрибута не встречалось при обучении в данном узле,
    возвращается мажоритарный класс этого узла.

    Входные параметры:
        tree (dict): Корень дерева (результат build_tree).
        data (dict): Закодированный набор данных.
        idx (numpy.ndarray): Индексы классифицируемых объектов.

    Возвращаемое значение:
        pred (numpy.ndarray): Массив предсказанных классов (0 или 1) для объектов idx.
    """

    out = np.zeros(len(idx), dtype=np.int64)

    def rec(node, pos):
        if len(pos) == 0:
            return
        if node["attr"] is None:
            out[pos] = node["pred"]
            return
        x = data["cols"][node["attr"]][idx[pos]]
        if node["thr"] is not None:
            m = x <= node["thr"]
            rec(node["children"][0], pos[m])
            rec(node["children"][1], pos[~m])
        else:
            known = np.zeros(len(pos), dtype=bool)
            for v, child in zip(node["values"], node["children"]):
                m = x == v
                known |= m
                rec(child, pos[m])
            out[pos[~known]] = node["pred"]

    rec(tree, np.arange(len(idx)))
    return out


def count_nodes(tree):
    """
    Подсчитывает число узлов и листьев дерева.

    Входные параметры:
        tree (dict): Корень дерева.

    Возвращаемое значение:
        (nodes, leaves) (tuple): Общее число узлов и число листьев.
    """

    if not tree["children"]:
        return 1, 1
    nodes, leaves = 1, 0
    for ch in tree["children"]:
        a, b = count_nodes(ch)
        nodes += a
        leaves += b
    return nodes, leaves


# 5. Оценка качества классификации

def compute_metrics(y_true, y_pred):
    """
    Вычисляет показатели качества бинарной классификации (положительный класс - '>50K').

    Входные параметры:
        y_true (numpy.ndarray): Истинные метки классов.
        y_pred (numpy.ndarray): Предсказанные метки классов.

    Возвращаемое значение:
        metrics (dict): Ключи 'accuracy', 'precision', 'recall', 'f1'.
    """

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_experiment(data, criterion, train_ratio, max_depth=MAX_DEPTH, seed=SEED):
    """
    Выполняет один эксперимент: делит данные, строит дерево на обучающей части и считает
    показатели качества на тестовой части (при train_ratio = 1.0 - на обучающей).

    Входные параметры:
        data (dict): Закодированный набор данных.
        criterion (str): Критерий выбора атрибута разбиения.
        train_ratio (float): Доля обучающей выборки.
        max_depth (int | None): Максимальная глубина дерева.
        seed (int): Начальное значение генератора случайных чисел.

    Возвращаемое значение:
        (tree, metrics) (tuple): Построенное дерево и словарь показателей качества.
    """

    train_idx, test_idx = split_data(data, train_ratio, seed)
    tree = build_tree(data, train_idx, list(range(len(data["cols"]))), criterion, 0, max_depth)
    eval_idx = test_idx if len(test_idx) else train_idx
    pred = predict(tree, data, eval_idx)
    return tree, compute_metrics(data["y"][eval_idx], pred)


# 6. Визуализация

def build_view(node, data, depth, viz_depth, max_children=MAX_VIZ_CHILDREN):
    """
    Строит упрощённое представление дерева для построения графика. Выводятся не все узлы:
    дерево обрезается по глубине viz_depth, а у категориального узла показываются только
    max_children самых крупных ветвей (по числу объектов), остальные заменяются одним блоком
    "Прочие (k знач.)". Нижние узлы (листья и узлы на границе глубины) содержат подпись
    "Класс: ..." и цвет класса; если дерево в узле продолжается, но обрезано, добавляется
    пометка. Внутренний узел содержит название атрибута, ветви подписаны условием
    ("<= t" / "> t" для числового атрибута, значением для категориального).

    Входные параметры:
        node (dict): Узел дерева.
        data (dict): Закодированный набор данных (для названий атрибутов и категорий).
        depth (int): Глубина узла.
        viz_depth (int): Максимальная отображаемая глубина.
        max_children (int): Максимум отображаемых ветвей у категориального узла.

    Возвращаемое значение:
        view (dict): Узел представления: 'text', 'color', 'children' (список (подпись, view)).
    """

    c0, c1 = node["counts"]
    stat = "n=%d [%d / %d]" % (node["n"], c0, c1)
    cls = CLASSES[node["pred"]]
    color = "#f4a582" if node["pred"] == 1 else "#92c5de"
    if node["attr"] is None or depth >= viz_depth:
        mark = "" if node["attr"] is None else "\n(далее...)"
        return {"text": "Класс: %s\n%s%s" % (cls, stat, mark), "color": color, "children": []}

    j = node["attr"]
    view = {"text": "%s ?\n%s" % (data["names"][j], stat), "color": "#ffffff", "children": []}
    if node["thr"] is not None:
        labels = ["<= %.4g" % node["thr"], "> %.4g" % node["thr"]]
        for label, ch in zip(labels, node["children"]):
            view["children"].append((label, build_view(ch, data, depth + 1, viz_depth, max_children)))
        return view

    kids = [(data["cats"][j][v], ch) for v, ch in zip(node["values"], node["children"])]
    kids.sort(key=lambda t: -t[1]["n"])
    shown, hidden = kids[:max_children], kids[max_children:]
    for label, ch in shown:
        view["children"].append((label[:16], build_view(ch, data, depth + 1, viz_depth, max_children)))
    if hidden:
        hc = np.sum([ch["counts"] for _, ch in hidden], axis=0)
        hcls = int(np.argmax(hc))
        view["children"].append(("...", {
            "text": "Прочие (%d знач.)\nКласс: %s\nn=%d [%d / %d]"
                    % (len(hidden), CLASSES[hcls], int(hc.sum()), hc[0], hc[1]),
            "color": "#f4a582" if hcls == 1 else "#92c5de", "children": []}))
    return view


def layout_view(view, depth, counter):
    """
    Назначает узлам представления координаты для горизонтального дерева (слева направо):
    координата x равна глубине узла, листья располагаются сверху вниз по оси y,
    родитель - по центру напротив своих потомков.

    Входные параметры:
        view (dict): Узел представления (результат build_view).
        depth (int): Глубина узла.
        counter (list): Список из одного элемента - номер следующей свободной строки (y-позиции).

    Возвращаемое значение:
        None. Координаты записываются в поля 'x' и 'y' узлов.
    """

    view["x"] = depth
    if not view["children"]:
        view["y"] = -counter[0]
        counter[0] += 1
        return
    for _, ch in view["children"]:
        layout_view(ch, depth + 1, counter)
    ys = [ch["y"] for _, ch in view["children"]]
    view["y"] = sum(ys) / len(ys)


def draw_view(ax, view):
    """
    Рисует узел представления и рекурсивно его потомков: рёбра с подписями и блоки с текстом.

    Входные параметры:
        ax (matplotlib.axes.Axes): Область рисования.
        view (dict): Узел представления с рассчитанными координатами.

    Возвращаемое значение:
        None.
    """

    for label, ch in view["children"]:
        ax.plot([view["x"], ch["x"]], [view["y"], ch["y"]], color="gray", lw=0.8, zorder=1)
        ax.text((view["x"] + ch["x"]) / 2, (view["y"] + ch["y"]) / 2, label, fontsize=9,
                ha="center", va="center", color="darkgreen", zorder=2,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        draw_view(ax, ch)
    ax.text(view["x"], view["y"], view["text"], fontsize=8, ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.3", fc=view["color"], ec="black", lw=0.7))


def plot_tree(tree, data, criterion, viz_depth=VIZ_DEPTH, path=None):
    """
    Рисует одно дерево решений на отдельном графике, ориентированном слева направо
    (корень слева, листья справа). Отображается верхняя часть дерева
    (до глубины viz_depth) и только самые крупные ветви категориальных узлов, т.к. полное
    дерево содержит сотни узлов. Цвет блока: голубой - класс <=50K, оранжевый - класс >50K.
    В нижних узлах явно подписан класс. Подписи рёбер - условия ветвления.

    Входные параметры:
        tree (dict): Корень дерева.
        data (dict): Закодированный набор данных.
        criterion (str): Критерий, по которому построено дерево (для заголовка).
        viz_depth (int): Отображаемая глубина.
        path (str | None): Файл для сохранения рисунка (None - не сохранять).

    Возвращаемое значение:
        fig (matplotlib.figure.Figure): Построенная фигура.
    """

    view = build_view(tree, data, 0, viz_depth)
    counter = [0]
    layout_view(view, 0, counter)
    fig, ax = plt.subplots(figsize=(4.6 * (viz_depth + 1), max(7, counter[0] * 1.15)))
    draw_view(ax, view)
    nodes, leaves = count_nodes(tree)
    ax.set_title("Дерево решений (разбиение по %s): всего узлов: %d, листьев: %d, показано до глубины %d"
                 % (CRITERIA_TITLES[criterion], nodes, leaves, viz_depth), fontsize=13)
    ax.set_xlim(-0.6, viz_depth + 0.6)
    ax.set_ylim(-(counter[0] - 1) - 0.8, 0.8)
    ax.axis("off")
    ax.legend(handles=[Patch(fc="#92c5de", ec="black", label="класс <=50K"),
                       Patch(fc="#f4a582", ec="black", label="класс >50K")],
              loc="lower left", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_metrics(ratios, results, path="metrics.png"):
    """
    Строит столбчатую диаграмму показателей качества в зависимости от соотношения обучающей
    и тестовой выборок. Фигура состоит из четырёх графиков 2x2 (accuracy, precision, recall,
    F-мера). На каждом графике для каждого соотношения выборок рисуются три столбца - по одному
    на критерий выбора атрибута разбиения. Значения подписаны над столбцами.

    Входные параметры:
        ratios (list): Доли обучающей выборки (например, [0.6, 0.7, 0.8, 0.9]).
        results (dict): Словарь {критерий: список словарей показателей по каждой доле}.
        path (str): Файл для сохранения рисунка.

    Возвращаемое значение:
        fig (matplotlib.figure.Figure): Построенная фигура.
    """

    labels = ["%d%%:%d%%" % (round(r * 100), round((1 - r) * 100)) for r in ratios]
    names = [("accuracy", "Accuracy"), ("precision", "Precision"),
             ("recall", "Recall"), ("f1", "F-мера")]
    colors = {"info_gain": "#4c78a8", "gain_ratio": "#f58518", "gini": "#54a24b"}
    x = np.arange(len(ratios))
    w = 0.26
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.ravel(), names):
        all_vals = []
        for i, crit in enumerate(CRITERIA):
            vals = [m[key] for m in results[crit]]
            all_vals += vals
            bars = ax.bar(x + (i - 1) * w, vals, w, label=CRITERIA_TITLES[crit], color=colors[crit])
            for bar, v in zip(bars, vals):
                ax.annotate("%.3f" % v, (bar.get_x() + bar.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=7, rotation=90)
        lo, hi = min(all_vals), max(all_vals)
        ax.set_ylim(max(0.0, lo - 0.15 * (hi - lo) - 0.05), min(1.0, hi + 0.12 * (hi - lo) + 0.05))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("Соотношение обучающей и тестовой выборок")
        ax.set_ylabel("Значение показателя")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Показатели качества классификации по критериям выбора атрибута", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


# 7. Точка входа в программу

def main():
    """
    Точка входа. Параметры берутся из констант в начале файла.
    Загружает и объединяет файлы adult.data.txt и adult.test.txt, затем строит три дерева на 100% данных
    и рисует каждое на отдельном графике, после чего для каждого из трёх критериев варьирует
    долю обучающей выборки 60%..90% с шагом 10%, печатает таблицы показателей и рисует
    сводный столбчатый график.

    Входные параметры:
        Нет.

    Возвращаемое значение:
        None.
    """

    rows = load_data(TRAIN_FILE) + load_data(TEST_FILE)
    data = encode_dataset(rows)
    print("Загружено записей: %d\n" % len(data["y"]))

    # Эксперимент 1: Построение трёх деревьев на 100% данных
    for crit in CRITERIA:
        tree, m = run_experiment(data, crit, 1.0)
        plot_tree(tree, data, crit, VIZ_DEPTH, "tree_%s.png" % crit)
        print("%-16s узлов/листьев: %s, accuracy (на обучающей): %.4f"
              % (CRITERIA_TITLES[crit], count_nodes(tree), m["accuracy"]))

    # Эксперимент 2: Варьирование соотношения выборок для всех трёх критериев
    results = {}
    for crit in CRITERIA:
        results[crit] = []
        print("\nКритерий: %s" % CRITERIA_TITLES[crit])
        print("%-9s %9s %10s %8s %8s" % ("train:test", "accuracy", "precision", "recall", "F1"))
        for r in RATIOS:
            _, m = run_experiment(data, crit, r)
            results[crit].append(m)
            print("%3d%%:%2d%%  %9.4f %10.4f %8.4f %8.4f" % (round(r * 100), round((1 - r) * 100),
                                                             m["accuracy"], m["precision"],
                                                             m["recall"], m["f1"]))
    plot_metrics(RATIOS, results)
    plt.show()


if __name__ == "__main__":
    main()