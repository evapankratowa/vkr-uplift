"""Оценка стратегии отбора против baseline — то, ради чего система существует.

Офлайн-оценки здесь — ориентир; финальный аргумент в проде даёт A/B-тест.
В учебной работе роль прода играет повторно смоделированный эксперимент.
"""

import numpy as np

from ..datasets.base import UpliftDataset


def ipw_policy_value(dataset: UpliftDataset, selected: np.ndarray) -> dict[str, float]:
    """IPW-оценка ценности стратегии по данным эксперимента, с 95% ДИ.

    Возвращает {'value', 'ci_low', 'ci_high'} в откликах на клиента.
    """
    raise NotImplementedError


def compare_strategies(
    dataset: UpliftDataset, strategies: dict[str, np.ndarray]
) -> "np.ndarray | object":
    """Попарное сравнение стратегий при равном бюджете контактов.

    Из сравнения исключаются клиенты, на которых стратегии совпадают: они дают
    ровно нулевой вклад в разницу и только добавляют шум.
    """
    raise NotImplementedError


def validation_experiment(
    dataset: UpliftDataset, strategies: dict[str, np.ndarray], seed: int
) -> dict:
    """Повторно смоделированный эксперимент для сравнения стратегий.

    Отдельный прогон DGP: клиенты, которым по стратегии положена коммуникация,
    получают её, и меряется фактический прирост отклика — независимая проверка
    офлайн-метрик.
    """
    raise NotImplementedError
