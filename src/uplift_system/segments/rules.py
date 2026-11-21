"""Бизнес-ограничения на сегменты: пересечения, частота, исключения."""

import pandas as pd


def check_overlap(segments: dict[str, pd.DataFrame], max_per_client: int = 1) -> pd.DataFrame:
    """Клиенты, попавшие больше чем в max_per_client кампаний. Ожидание — пустая таблица."""
    raise NotImplementedError


def apply_cooldown(
    selection: pd.DataFrame, history: pd.DataFrame, cooldown_days: int
) -> pd.DataFrame:
    """Убирает клиентов, получавших коммуникацию за последние cooldown_days."""
    raise NotImplementedError


def apply_exclusions(selection: pd.DataFrame, excluded_ids: pd.Series) -> pd.DataFrame:
    """Снимает клиентов из стоп-листа; печатает счётчик до и после."""
    raise NotImplementedError
