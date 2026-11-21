"""Выгрузка готового сегмента в формате системы рассылки."""

from pathlib import Path

import pandas as pd


def export_segment(selection: pd.DataFrame, campaign_id: str, fmt: str = "csv") -> Path:
    """Пишет сегмент в output/tables с префиксом папки проекта в имени файла.

    Возвращает путь. Имя: <PREFIX>_segment_<campaign_id>.<fmt>.
    """
    raise NotImplementedError
