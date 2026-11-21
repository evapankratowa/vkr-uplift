"""Пути проекта и префикс файлов-результатов.

Префикс выводится из имени папки проекта, а не хардкодится: при переименовании
папки имена выгрузок едут вместе с ней.
"""

from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parents[2]
PROJECT_PREFIX: str = PROJECT_DIR.name

DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_INTERIM = PROJECT_DIR / "data" / "interim"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
CONFIGS = PROJECT_DIR / "configs"
CHECKS = PROJECT_DIR / "checks"
OUT_FIGURES = PROJECT_DIR / "output" / "figures"
OUT_TABLES = PROJECT_DIR / "output" / "tables"
OUT_REPORTS = PROJECT_DIR / "output" / "reports"


def out_path(kind: str, name: str, suffix: str) -> Path:
    """Путь файла-результата с префиксом папки проекта.

    kind: figures | tables | reports. Возвращает <output>/<kind>/<PREFIX>_<name>.<suffix>.
    """
    base = {"figures": OUT_FIGURES, "tables": OUT_TABLES, "reports": OUT_REPORTS}[kind]
    return base / f"{PROJECT_PREFIX}_{name}.{suffix}"
