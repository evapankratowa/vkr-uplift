"""Слой PostgreSQL: загрузка сырой выгрузки кэшбэка и чтение ковариат обратно.

Сырой CSV (источник 1 из DATA_PASSPORT, одна строка = клиент) раскладывается
в нормализованные таблицы схемы `uplift` идемпотентной вставкой
(ON CONFLICT DO NOTHING), поэтому повторный прогон загрузчика состояние
не меняет. Обратное чтение (`read_raw`) восстанавливает те же колонки, что
у файла, — это мост для флага `--source db`: DataFrame подаётся в тот же
адаптер, что и CSV-файл.

Экспорт пакета ленивый (PEP 562): импорт без установленных SQLAlchemy/psycopg
не падает, а `python -m uplift_system.db.load` не получает модуль дважды
в sys.modules (иначе runpy предупреждает о непредсказуемом поведении).
"""

from typing import Any

__all__ = [
    "DEFAULT_DATABASE_URL",
    "RawDataError",
    "database_url",
    "get_connection",
    "load_raw",
    "read_raw",
]


def __getattr__(name: str) -> Any:
    """Лениво отдаёт публичные имена из подмодулей пакета."""
    if name in ("RawDataError", "load_raw"):
        from . import load

        return getattr(load, name)
    if name in ("DEFAULT_DATABASE_URL", "database_url", "get_connection"):
        from . import connection

        return getattr(connection, name)
    if name == "read_raw":
        from . import read

        return read.read_raw
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
