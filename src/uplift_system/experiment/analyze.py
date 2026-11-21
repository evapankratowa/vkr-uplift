"""Разбор проведённого эксперимента: ATE, интервалы, проверки валидности."""

import numpy as np
import pandas as pd
from scipy import stats

from ..datasets.base import UpliftDataset


def ate(dataset: UpliftDataset, outcome: str, alpha: float = 0.05) -> dict[str, float]:
    """ATE как разность долей отклика с доверительным интервалом.

    Величины — в долях отклика. Интервал по нормальному приближению для разности
    двух биномиальных долей.
    """
    y, t = dataset.y(outcome), dataset.t
    y_t, y_c = y[t == 1], y[t == 0]
    n_t, n_c = len(y_t), len(y_c)
    assert n_t > 0 and n_c > 0, "одна из групп пуста"
    p_t, p_c = y_t.mean(), y_c.mean()
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "ate": float(p_t - p_c),
        "ci_low": float(p_t - p_c - z * se),
        "ci_high": float(p_t - p_c + z * se),
        "se": float(se),
        "rate_treat": float(p_t),
        "rate_control": float(p_c),
        "n_treat": int(n_t),
        "n_control": int(n_c),
    }


def covariate_balance(dataset: UpliftDataset) -> pd.DataFrame:
    """Баланс ковариат между группами: стандартизованная разность средних (SMD).

    Порог |SMD| <= 0.1 — группы считаются сбалансированными. Возвращает таблицу,
    отсортированную по |SMD| убыванием.
    """
    t = dataset.t
    rows = []
    for column in dataset.features.columns:
        values = dataset.features[column].to_numpy()
        a, b = values[t == 1], values[t == 0]
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = 0.0 if pooled_sd == 0 else (a.mean() - b.mean()) / pooled_sd
        rows.append(
            {"feature": column, "mean_treat": a.mean(), "mean_control": b.mean(), "smd": smd}
        )
    table = pd.DataFrame(rows)
    table["abs_smd"] = table["smd"].abs()
    return table.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def propensity_by_strata(
    dataset: UpliftDataset, n_bins: int = 10, min_group: int = 30
) -> pd.DataFrame:
    """Проверка Допущения 1: доля назначенных в тест по группам каждого признака.

    При корректной рандомизации доля одинакова во всех группах с точностью до
    биномиального шума. Возвращает по признаку максимальное отклонение от общей
    доли в единицах его стандартной ошибки (z) и число проверенных групп.

    Признак разбивается на группы по-разному в зависимости от того, что он такое:
    непрерывный — на квантильные бины, а признак с небольшим числом значений
    (индикатор из one-hot, флаг пропуска) — по самим значениям. Делить бинарную
    колонку на децили нельзя: `qcut` отдаёт один бин или пропуски, и проверка
    молча превращается в NaN. Группы меньше min_group наблюдений отбрасываются:
    нормальное приближение на них не работает. Признак, у которого не осталось
    ни одной пригодной группы, помечается n_groups = 0 и в вердикт не идёт.
    """
    t = dataset.t
    overall = t.mean()
    rows = []
    for column in dataset.features.columns:
        values = dataset.features[column]
        if values.nunique(dropna=False) <= n_bins:
            groups = values.astype("category").cat.codes
        else:
            groups = pd.qcut(values, q=n_bins, duplicates="drop", labels=False)
        frame = pd.DataFrame({"group": groups, "t": t})
        agg = frame.groupby("group", dropna=False)["t"].agg(["mean", "size"])
        agg = agg[agg["size"] >= min_group]
        if agg.empty:
            rows.append(
                {
                    "feature": column,
                    "n_groups": 0,
                    "share_worst": np.nan,
                    "size_worst": 0,
                    "z_worst": np.nan,
                }
            )
            continue
        se = np.sqrt(overall * (1 - overall) / agg["size"])
        z = (agg["mean"] - overall) / se
        worst = z.abs().idxmax()
        rows.append(
            {
                "feature": column,
                "n_groups": int(len(agg)),
                "share_worst": float(agg.loc[worst, "mean"]),
                "size_worst": int(agg.loc[worst, "size"]),
                "z_worst": float(z.loc[worst]),
            }
        )
    table = pd.DataFrame(rows)
    table["abs_z"] = table["z_worst"].abs()
    return table.sort_values("abs_z", ascending=False, na_position="last").reset_index(drop=True)


def aa_test(
    dataset: UpliftDataset, outcome: str, n_runs: int = 1000, alpha: float = 0.05, seed: int = 42
) -> dict[str, float]:
    """A/A на неподверженной группе: доля ложных отклонений должна быть около alpha.

    Берутся только клиенты из контроля (воздействия не было), делятся случайно
    пополам, считается z-тест разности долей. Если доля отклонений заметно выше
    alpha — сломан генератор или сплит, а не «повезло».
    """
    y = dataset.y(outcome)[dataset.t == 0]
    n = len(y)
    assert n > 100, "в контроле слишком мало клиентов для A/A"
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_runs):
        mask = rng.binomial(1, 0.5, n).astype(bool)
        a, b = y[mask], y[~mask]
        if len(a) < 2 or len(b) < 2:
            continue
        p_a, p_b = a.mean(), b.mean()
        se = np.sqrt(p_a * (1 - p_a) / len(a) + p_b * (1 - p_b) / len(b))
        if se == 0:
            continue
        if abs((p_a - p_b) / se) > stats.norm.ppf(1 - alpha / 2):
            rejects += 1
    return {
        "false_positive_rate": rejects / n_runs,
        "expected": alpha,
        "n_runs": int(n_runs),
        "n_control": int(n),
    }
