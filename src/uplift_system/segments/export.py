"""Выгрузка готового сегмента в формате системы рассылки.

Контракт — docs/EXTERNAL_INTEGRATIONS.md, раздел 3: CSV/Parquet с client_id
и атрибутами отбора в двух форматах — standard (только целевой сегмент)
и ab (сегмент плюс удерживаемая контрольная группа для оценки эффекта кампании).
Контрольная группа случайно выбирается из оценённой аудитории вне сегмента,
поэтому пересечения с целевой выгрузкой нет по построению.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import OUT_TABLES, PROJECT_PREFIX

GROUP_COLUMN = "group"
GROUP_TARGET = "target"
GROUP_CONTROL = "control"


def _write(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    """Пишет кадр в csv или parquet без индекса; поднимает руки на прочие форматы."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        frame.to_csv(path, index=False)
    elif fmt == "parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"неизвестный формат файла {fmt!r}, есть csv | parquet")


def _order_columns(frame: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """Оставляет колонки из columns в заданном порядке; отсутствующие пропускает."""
    if columns is None:
        return frame
    existing = [c for c in columns if c in frame.columns]
    return frame[existing]


def export_segment(
    selection: pd.DataFrame,
    campaign_id: str,
    fmt: str = "csv",
    columns: list[str] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Пишет сегмент в формате standard: только целевые клиенты.

    Возвращает путь. Имя: <PREFIX>_segment_<campaign_id>.<fmt>, где PREFIX —
    имя папки проекта (см. paths.py). columns — порядок и состав колонок выгрузки
    (export.columns из configs/campaign.yaml); None — все колонки сегмента.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUT_TABLES
    path = out_dir / f"{PROJECT_PREFIX}_segment_{campaign_id}.{fmt}"
    _write(_order_columns(selection, columns), path, fmt)
    return path


def export_segment_ab(
    selection: pd.DataFrame,
    scores: pd.DataFrame,
    campaign_id: str,
    control_share: float,
    fmt: str = "csv",
    seed: int = 42,
    columns: list[str] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Пишет формат ab: целевой сегмент + удерживаемая контрольная группа.

    scores — оценённая аудитория (client_id + скоры всех клиентов); контроль
    выбирается из неё вне сегмента случайно и воспроизводимо (seed): размер
    round(len(selection) * control_share), доля — из configs/campaign.yaml
    (export.control_share). Скоры контрольникам не приписываются: их атрибуты
    отбора — NaN, отличие от целевых видно по колонке group (target | control).
    Имя: <PREFIX>_segment_ab_<campaign_id>.<fmt>.
    """
    if not 0.0 <= control_share <= 1.0:
        raise ValueError(f"control_share должен быть долей в [0, 1], получен {control_share}")
    out_dir = Path(out_dir) if out_dir is not None else OUT_TABLES
    target = selection.copy()
    target[GROUP_COLUMN] = GROUP_TARGET

    target_ids = set(selection["client_id"])
    pool = scores.loc[~scores["client_id"].isin(target_ids), ["client_id"]]
    n_control = int(round(len(selection) * float(control_share)))
    n_control = min(n_control, len(pool))
    if n_control > 0:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(pool), size=n_control, replace=False))
        control = pool.iloc[take].copy()
    else:
        control = pool.iloc[0:0].copy()
    control[GROUP_COLUMN] = GROUP_CONTROL

    merged = pd.concat([target, control], ignore_index=True, sort=False)
    if columns is not None and GROUP_COLUMN not in columns:
        columns = [*columns, GROUP_COLUMN]
    path = out_dir / f"{PROJECT_PREFIX}_segment_ab_{campaign_id}.{fmt}"
    _write(_order_columns(merged, columns), path, fmt)
    return path
