"""Загрузка конфигов YAML с проверкой обязательных ключей."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, required: tuple[str, ...] = ()) -> dict[str, Any]:
    """Читает YAML-конфиг и проверяет наличие ключей верхнего уровня из required."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"{path.name}: нет обязательных ключей {missing}")
    return cfg
