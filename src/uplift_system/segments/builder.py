"""Отбор сегмента с максимальным суммарным uplift при ограничении бюджета."""

from typing import Any

import pandas as pd


def score_clients(model_name: str, features: pd.DataFrame, models_dir: Any = None) -> pd.DataFrame:
    """Применяет сохранённую модель к клиентам: client_id, uplift_score, decile."""
    raise NotImplementedError


def select(scores: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Отбирает клиентов под бюджет контактов и отсечку min_uplift.

    Жадный отбор по убыванию uplift эквивалентен оптимуму при одинаковой стоимости
    контакта; при разной стоимости — задача о рюкзаке, см. DECISIONS.md.
    """
    raise NotImplementedError


def optimal_share(
    scores: pd.DataFrame, cost_per_contact: float, value_per_response: float
) -> float:
    """Доля аудитории, на которой предельный uplift равен стоимости контакта.

    Точка отсечения для главы 3 (экономическая эффективность).
    """
    raise NotImplementedError
