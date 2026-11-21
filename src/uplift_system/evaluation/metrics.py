"""Ранжирующие метрики uplift — считаются на наблюдаемых данных.

Меряют качество РАНЖИРОВАНИЯ клиентов по эффекту, а не абсолютный бизнес-эффект;
последнее — evaluation/policy.py. Формулы повторяют определения scikit-uplift,
чтобы собственные реализации можно было сверить с эталоном (проверка №5).

Единицы: uplift_at_k и побакетный uplift — в долях отклика (0.01 = 1 п.п.);
кривые — в накопленных откликах (штуки клиентов).
"""

import numpy as np
import pandas as pd


def _thresholds(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    """Накопленные суммы в точках РАЗЛИЧНЫХ значений скора, по убыванию.

    Кривая считается не по одному клиенту на точку, а по одной точке на
    различное значение скора: внутри группы клиентов с одинаковым скором
    порядок произволен, и растягивать по ним ломаную значит подмешивать в
    метрику случайный порядок строк. Особенно это важно для кривой идеальной
    модели, где различных значений скора всего три. Так же устроено в
    scikit-uplift, поэтому значения совпадают побитово (проверка №5).

    Возвращает: n_all, n_treat, n_control, отклики в тесте, отклики в контроле.
    """
    y, t, score = np.asarray(y), np.asarray(t), np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")[::-1]
    y_sorted, t_sorted, score_sorted = y[order], t[order], score[order]

    y_treat = np.where(t_sorted == 1, y_sorted, 0)
    y_control = np.where(t_sorted == 0, y_sorted, 0)

    distinct = np.where(np.diff(score_sorted))[0]
    idx = np.r_[distinct, score_sorted.size - 1]

    n_treat = np.cumsum(t_sorted)[idx]
    n_all = idx + 1
    n_control = n_all - n_treat
    return n_all, n_treat, n_control, np.cumsum(y_treat)[idx], np.cumsum(y_control)[idx]


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float) -> float:
    """Разность долей отклика в тесте и контроле среди топ-k доли по скору.

    k — доля аудитории (0.3 = 30%). Результат в долях отклика. Если в топ-k не
    попал ни один клиент одной из групп, возвращается nan: метрика не определена,
    и подменять её нулём нельзя.
    """
    assert 0 < k <= 1, "k должен быть долей в (0, 1]"
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")[::-1]
    idx = order[: max(1, int(round(k * len(score))))]
    y_top, t_top = np.asarray(y)[idx], np.asarray(t)[idx]
    if t_top.sum() == 0 or (1 - t_top).sum() == 0:
        return float("nan")
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Кривая uplift: x — число клиентов, y — накопленный прирост откликов.

    y = (Y_t / N_t - Y_c / N_c) * (N_t + N_c) — определение scikit-uplift.
    """
    n_all, n_t, n_c, y_t, y_c = _thresholds(y, t, score)
    rate_t = np.divide(y_t, n_t, out=np.zeros(len(n_t), dtype=float), where=n_t != 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros(len(n_c), dtype=float), where=n_c != 0)
    curve = (rate_t - rate_c) * n_all
    return np.r_[0, n_all], np.r_[0.0, curve]


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Кривая Qini: y = Y_t - Y_c * N_t / N_c — определение scikit-uplift."""
    n_all, n_t, n_c, y_t, y_c = _thresholds(y, t, score)
    ratio = np.divide(n_t, n_c, out=np.zeros(len(n_t), dtype=float), where=n_c != 0)
    curve = y_t - y_c * ratio
    return np.r_[0, n_all], np.r_[0.0, curve]


def perfect_qini_curve(
    y: np.ndarray, t: np.ndarray, negative_effect: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Кривая Qini идеальной модели.

    Идеальный скор: +1 откликнувшимся в тесте, -1 откликнувшимся в контроле,
    0 остальным. Смысл: вперёд ставятся те, на кого воздействие подействовало,
    в конец — те, кому оно помешало (отклик был бы и без него). Определение
    повторяет scikit-uplift с negative_effect=True.
    """
    y, t = np.asarray(y), np.asarray(t)
    if negative_effect:
        ideal = (y * t - y * (1 - t)).astype(float)
        return qini_curve(y, t, ideal)
    n = len(y)
    ratio_random = y[t == 1].sum() - len(y[t == 1]) * y[t == 0].sum() / len(y[t == 0])
    return np.array([0.0, ratio_random, n]), np.array([0.0, ratio_random, ratio_random])


def perfect_uplift_curve(y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Кривая uplift идеальной модели — определение scikit-uplift.

    Идеальный скор ставит вперёд согласованные пары (отклик в тесте и его
    отсутствие в контроле), а порядок внутри задаётся тем, какая из групп
    «мешающих» наблюдений больше.
    """
    y, t = np.asarray(y), np.asarray(t)
    control_responders = int(np.sum((y == 1) & (t == 0)))
    treated_non_responders = int(np.sum((y == 0) & (t == 1)))
    summand = y if control_responders > treated_non_responders else t
    ideal = 2 * (y == t).astype(float) + summand
    return uplift_curve(y, t, ideal)


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """Площадь под ломаной методом трапеций."""
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def _normalized_auc(
    x_actual: np.ndarray,
    y_actual: np.ndarray,
    x_perfect: np.ndarray,
    y_perfect: np.ndarray,
    normalize: bool,
) -> float:
    """Площадь над случайной прямой, при normalize — в долях от идеальной.

    Случайная прямая проводится из нуля в конец идеальной кривой: конечная точка
    у всех кривых одна и та же, потому что при полном охвате порядок клиентов
    уже не важен.
    """
    baseline = _auc(np.array([0.0, x_perfect[-1]]), np.array([0.0, y_perfect[-1]]))
    actual = _auc(x_actual, y_actual) - baseline
    if not normalize:
        return actual
    perfect = _auc(x_perfect, y_perfect) - baseline
    if perfect == 0:
        return float("nan")
    return actual / perfect


def qini_auc(y: np.ndarray, t: np.ndarray, score: np.ndarray, normalize: bool = True) -> float:
    """Коэффициент Qini: площадь между кривой Qini и случайной прямой.

    normalize=True — в долях от той же площади у идеальной модели: 0 —
    случайное ранжирование, 1 — идеальное, отрицательное значение — хуже
    случайного.
    """
    x_actual, y_actual = qini_curve(y, t, score)
    x_perfect, y_perfect = perfect_qini_curve(y, t)
    return _normalized_auc(x_actual, y_actual, x_perfect, y_perfect, normalize)


def uplift_auc(y: np.ndarray, t: np.ndarray, score: np.ndarray, normalize: bool = True) -> float:
    """AUUC: то же, что qini_auc, но на кривой uplift."""
    x_actual, y_actual = uplift_curve(y, t, score)
    x_perfect, y_perfect = perfect_uplift_curve(y, t)
    return _normalized_auc(x_actual, y_actual, x_perfect, y_perfect, normalize)


def bucket_uplift(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_buckets: int = 10
) -> pd.DataFrame:
    """Побакетный uplift по децилям скора — диагностика монотонности, не метрика.

    Колонки: bucket (0 — лучший), n, n_treat, n_control, rate_treat,
    rate_control, uplift (доли отклика).
    """
    idx = np.argsort(np.asarray(score, dtype=float), kind="mergesort")[::-1]
    y_sorted, t_sorted = np.asarray(y)[idx], np.asarray(t)[idx]
    buckets = np.minimum((np.arange(len(y_sorted)) * n_buckets) // len(y_sorted), n_buckets - 1)
    rows = []
    for bucket in range(n_buckets):
        mask = buckets == bucket
        y_b, t_b = y_sorted[mask], t_sorted[mask]
        rate_t = y_b[t_b == 1].mean() if (t_b == 1).any() else np.nan
        rate_c = y_b[t_b == 0].mean() if (t_b == 0).any() else np.nan
        rows.append(
            {
                "bucket": bucket,
                "n": int(mask.sum()),
                "n_treat": int((t_b == 1).sum()),
                "n_control": int((t_b == 0).sum()),
                "rate_treat": rate_t,
                "rate_control": rate_c,
                "uplift": rate_t - rate_c,
            }
        )
    return pd.DataFrame(rows)


METRICS = {
    "qini_auc": lambda y, t, s: qini_auc(y, t, s),
    "uplift_auc": lambda y, t, s: uplift_auc(y, t, s),
    "uplift_at_10": lambda y, t, s: uplift_at_k(y, t, s, 0.1),
    "uplift_at_30": lambda y, t, s: uplift_at_k(y, t, s, 0.3),
}


def bootstrap_ci(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Бутстрап-интервал метрики ресэмплом ПО КЛИЕНТАМ.

    Ресэмпл по клиентам, а не по группам: клиент — единица наблюдения, и
    независимость выполняется только на этом уровне. Возвращает точечную оценку,
    границы интервала и стандартную ошибку.
    """
    assert metric in METRICS, f"неизвестная метрика {metric}, есть {sorted(METRICS)}"
    func = METRICS[metric]
    y, t, score = np.asarray(y), np.asarray(t), np.asarray(score)
    point = func(y, t, score)
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        values[i] = func(y[idx], t[idx], score[idx])
    values = values[np.isfinite(values)]
    return {
        "point": float(point),
        "ci_low": float(np.quantile(values, alpha / 2)),
        "ci_high": float(np.quantile(values, 1 - alpha / 2)),
        "se": float(values.std(ddof=1)),
        "n_boot_used": int(len(values)),
    }
