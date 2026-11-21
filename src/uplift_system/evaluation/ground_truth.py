"""Сверка с истинным эффектом — доступна только на синтетике.

Ранжирующие метрики не отличают «модель поймала эффект» от «модель поймала шум,
который коррелирует с отбором»: на наблюдаемых данных истинный эффект клиента
ненаблюдаем в принципе. На симуляции он известен, и это проверяется прямо.
"""

import numpy as np
import pandas as pd
from scipy import stats


def pehe(ite_true: np.ndarray, ite_pred: np.ndarray) -> float:
    """Корень из средней квадратичной ошибки оценки CATE, в долях отклика."""
    ite_true, ite_pred = np.asarray(ite_true, float), np.asarray(ite_pred, float)
    assert ite_true.shape == ite_pred.shape, "разная длина истинного и предсказанного эффекта"
    return float(np.sqrt(np.mean((ite_true - ite_pred) ** 2)))


def rank_correlation(a: np.ndarray, b: np.ndarray, method: str = "kendall") -> float:
    """Ранговая корреляция двух векторов: kendall (tau-b) или spearman.

    Для uplift важно именно ранжирование: отбор идёт по порядку клиентов, а не по
    абсолютной величине оценки.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if method == "kendall":
        return float(stats.kendalltau(a, b).statistic)
    if method == "spearman":
        return float(stats.spearmanr(a, b).statistic)
    raise ValueError(f"неизвестный метод {method}")


def segment_recovery(segment: pd.Series, score: np.ndarray, k: float) -> pd.DataFrame:
    """Состав топ-k по скору в терминах истинных сегментов.

    Главный содержательный критерий: система должна набирать убеждаемых и не
    трогать спящих собак. Колонки: segment, n_in_top, share_in_top,
    share_in_population, lift (во сколько раз сегмент переоценён в отборе).
    """
    assert 0 < k <= 1, "k должен быть долей в (0, 1]"
    n_top = max(1, int(round(k * len(score))))
    top = np.argsort(-np.asarray(score, float), kind="stable")[:n_top]
    in_top = segment.iloc[top].value_counts()
    in_pop = segment.value_counts()
    table = pd.DataFrame(
        {
            "n_in_top": in_top.reindex(in_pop.index).fillna(0).astype(int),
            "share_in_top": (in_top.reindex(in_pop.index).fillna(0) / n_top),
            "share_in_population": in_pop / len(segment),
        }
    )
    table["lift"] = table["share_in_top"] / table["share_in_population"]
    return table.reset_index(names="segment")


def true_policy_value(ite_true: np.ndarray, selected: np.ndarray) -> float:
    """Истинная ценность политики: сумма истинных CATE отобранных клиентов.

    Единицы — ожидаемые дополнительные отклики (штуки), а не доли.
    """
    ite_true, selected = np.asarray(ite_true, float), np.asarray(selected).astype(bool)
    return float(ite_true[selected].sum())


def top_k_policy(score: np.ndarray, k: float) -> np.ndarray:
    """Булева маска топ-k доли по скору."""
    n_top = max(1, int(round(k * len(score))))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-np.asarray(score, float), kind="stable")[:n_top]] = True
    return mask


def policy_value_ratio(ite_true: np.ndarray, score: np.ndarray, k: float) -> float:
    """Доля от максимально возможной ценности при отборе топ-k по скору.

    Знаменатель — ценность отбора топ-k по ИСТИННОМУ эффекту. 1.0 означает, что
    модель отобрала ровно тех же клиентов по ценности, что и оракул.
    """
    actual = true_policy_value(ite_true, top_k_policy(score, k))
    oracle = true_policy_value(ite_true, top_k_policy(ite_true, k))
    if oracle == 0:
        return float("nan")
    return actual / oracle
