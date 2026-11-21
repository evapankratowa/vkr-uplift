"""Загрузчик сырой выгрузки кэшбэка в операционную схему uplift.

Сырой CSV (источник 1 из DATA_PASSPORT: одна строка = клиент, 42 колонки —
client_id, 17 категорий трат в двух окнах, кэшбэк банка и партнёров в двух
окнах, три счётчика вовлечённости) раскладывается в нормализованные таблицы:

    client, spend_category, observation_window,
    client_spend, client_cashback, communication_stats.

Справочники категорий и окон собираются из самого файла (имена колонок),
номера справочников детерминированы, поэтому вставка идемпотентна
(ON CONFLICT DO NOTHING): повторный прогон не меняет состояние БД.

Нарушения CHECK-ограничений схемы (отрицательные суммы трат, отрицательные
счётчики, неуникальный client_id) отсекаются ДО вставки — бросается
RawDataError с физическим номером строки CSV. Отрицательный кэшбэк разрешён:
это сторно при возврате покупки (комментарий к client_cashback в DDL).

Счётчики витрины относятся к окну «последние 3 месяца» (DATA_PASSPORT),
поэтому в communication_stats они кладутся в окно last_3_month.

Команда для ручного прогона (в cli.py подключается позже, тикет T3):

    python -m uplift_system.db.load --csv data/raw/cashback_raw_2022-11-15.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.sql.schema import Table

ID_COLUMN = "client_id"

# Окна наблюдения: код и границы из DATA_PASSPORT («до» — семь месяцев
# 2022-01-01..2022-08-01; «последние 3 месяца» — по дате файла 2022-11-15).
# window_id детерминирован, порядок — как в файле: сначала «до», потом «после».
WINDOWS: tuple[tuple[int, str, str, str], ...] = (
    (1, "before", "2022-01-01", "2022-08-01"),
    (2, "last_3_month", "2022-08-01", "2022-11-15"),
)
COMMUNICATION_WINDOW_ID = 2  # счётчики витрины — окно «последние 3 месяца»

# Кэшбэк: колонка CSV -> (источник для ENUM cashback_source, код окна).
CASHBACK_COLUMNS: dict[str, tuple[str, str]] = {
    "cb_merch_before": ("partner", "before"),
    "cb_merch_last_3_month": ("partner", "last_3_month"),
    "cb_bank_before": ("bank", "before"),
    "cb_bank_last_3_month": ("bank", "last_3_month"),
}
CASHBACK_OUT_COLUMNS: dict[tuple[str, str], str] = {
    ("partner", "before"): "cb_merch_before",
    ("partner", "last_3_month"): "cb_merch_last_3_month",
    ("bank", "before"): "cb_bank_before",
    ("bank", "last_3_month"): "cb_bank_last_3_month",
}

COMMUNICATION_COLUMNS: tuple[str, ...] = ("cnt_communication", "cnt_view", "cnt_activation")

# Траты и счётчики ограничены CHECK-ом (>= 0) — валидация перед вставкой.
CHECKED_NON_NEGATIVE = "категории трат и счётчики вовлечённости"

INSERT_CHUNK = 20_000


class RawDataError(ValueError):
    """Сырая выгрузка не проходит ограничения схемы uplift.

    Атрибуты: номер строки CSV (физический, заголовок = строка 1) и колонка.
    """

    def __init__(self, row: int, column: str, value: object, reason: str) -> None:
        self.row = row
        self.column = column
        self.value = value
        super().__init__(f"строка {row}, колонка '{column}': {value} — {reason}")


def categories_of(columns: Sequence[str]) -> list[str]:
    """Категории трат — из колонок окна «до» (суффикс _before), в порядке файла.

    Кэшбэковые колонки с тем же суффиксом (cb_merch_before и пр.) не категории.
    """
    cashback_cols = set(CASHBACK_COLUMNS)
    categories = [
        c[: -len("_before")] for c in columns if c.endswith("_before") and c not in cashback_cols
    ]
    if not categories:
        raise ValueError("в файле нет колонок трат с суффиксом '_before'")
    return categories


def canonical_columns(categories: Sequence[str]) -> list[str]:
    """Порядок колонок сырой схемы: id, категории в двух окнах, кэшбэк, счётчики."""
    columns = [ID_COLUMN]
    for name in categories:
        columns += [f"{name}_before", name]
    columns += list(CASHBACK_COLUMNS)
    columns += list(COMMUNICATION_COLUMNS)
    return columns


def validate_raw(frame: pd.DataFrame) -> None:
    """Проверяет сырой набор перед вставкой; бросает RawDataError с номером строки.

    Контроль: обязательные колонки, уникальность client_id, отсутствие пропусков,
    неотрицательность трат по категориям и счётчиков (CHECK в DDL). Отрицательный
    кэшбэк — легальный сторно, его не трогаем.
    """
    categories = categories_of(list(frame.columns))
    required = [ID_COLUMN] + canonical_columns(categories)[1:]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"в файле нет обязательных колонок: {missing}")

    dup_mask = frame[ID_COLUMN].duplicated(keep=False)
    if dup_mask.any():
        pos = int(dup_mask.to_numpy().nonzero()[0][0])
        raise RawDataError(
            row=pos + 2,
            column=ID_COLUMN,
            value=frame[ID_COLUMN].iloc[pos],
            reason="client_id повторяется, ключ клиента должен быть уникален",
        )

    payload = frame[required].drop(columns=ID_COLUMN)
    nan_pos = payload.isna().to_numpy().nonzero()
    if len(nan_pos[0]):
        pos = int(nan_pos[0][0])
        raise RawDataError(
            row=pos + 2,
            column=str(payload.columns[int(nan_pos[1][0])]),
            value=None,
            reason="пропуск в данных: сырые нули содержательны, NaN недопустимы",
        )

    non_negative = [f"{c}_before" for c in categories] + list(categories)
    non_negative += list(COMMUNICATION_COLUMNS)
    for column in non_negative:
        bad = (frame[column] < 0).to_numpy().nonzero()[0]
        if len(bad):
            pos = int(bad[0])
            raise RawDataError(
                row=pos + 2,
                column=column,
                value=frame[column].iloc[pos],
                reason="отрицательное значение запрещено CHECK-ограничением схемы",
            )


def prepare_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Превращает сырой набор в длинные таблицы под INSERT — чистый pandas, без БД.

    Возвращает словарь 'таблица -> DataFrame' с колонками, совпадающими
    с DDL по именам: client, spend_category, observation_window, client_spend,
    client_cashback, communication_stats.
    """
    validate_raw(frame)
    categories = categories_of(list(frame.columns))

    window_ids = {code: wid for wid, code, _, _ in WINDOWS}
    category_ids = {name: i + 1 for i, name in enumerate(categories)}

    spend_parts = []
    cashback_parts = []
    for name in categories:
        for suffix, window_code in (("_before", "before"), ("", "last_3_month")):
            column = f"{name}{suffix}"
            part = frame[[ID_COLUMN, column]].rename(columns={column: "amount"})
            part["category_id"] = category_ids[name]
            part["window_id"] = window_ids[window_code]
            spend_parts.append(part)
    for column, (source, window_code) in CASHBACK_COLUMNS.items():
        part = frame[[ID_COLUMN, column]].rename(columns={column: "amount"})
        part["source"] = source
        part["window_id"] = window_ids[window_code]
        cashback_parts.append(part)

    client = frame[[ID_COLUMN]].rename_axis(None).reset_index(drop=True)
    spend_category = pd.DataFrame(
        {"category_id": list(category_ids.values()), "name": list(category_ids)}
    )
    observation_window = pd.DataFrame(
        {
            "window_id": [w[0] for w in WINDOWS],
            "code": [w[1] for w in WINDOWS],
            "date_from": [w[2] for w in WINDOWS],
            "date_to": [w[3] for w in WINDOWS],
        }
    )
    communication = frame[[ID_COLUMN, *COMMUNICATION_COLUMNS]].copy()
    communication["window_id"] = COMMUNICATION_WINDOW_ID

    return {
        "client": client,
        "spend_category": spend_category,
        "observation_window": observation_window,
        "client_spend": pd.concat(spend_parts, ignore_index=True),
        "client_cashback": pd.concat(cashback_parts, ignore_index=True),
        "communication_stats": communication,
    }


def _insert_frames(prepared: dict[str, pd.DataFrame], conn: Connection) -> dict[str, int]:
    """Вставляет подготовленные таблицы идемпотентно и возвращает число строк по таблицам.

    Колонки и типы повторяют db/001_init.sql (артефакт главы 2, не изменять);
    схема создаётся только если её нет — рабочий путь это docker-compose,
    где DDL применён при первом старте контейнера.
    """
    from sqlalchemy import (
        BigInteger,
        Column,
        Date,
        Integer,
        MetaData,
        Numeric,
        SmallInteger,
        String,
        Table,
        func,
        select,
    )
    from sqlalchemy.dialects.postgresql import ENUM
    from sqlalchemy.schema import CreateSchema

    metadata = MetaData()
    conn.execute(CreateSchema("uplift", if_not_exists=True))
    tables = {
        "client": Table(
            "client", metadata, Column("client_id", BigInteger, primary_key=True), schema="uplift"
        ),
        "spend_category": Table(
            "spend_category",
            metadata,
            Column("category_id", SmallInteger, primary_key=True),
            Column("name", String(32), nullable=False),
            schema="uplift",
        ),
        "observation_window": Table(
            "observation_window",
            metadata,
            Column("window_id", SmallInteger, primary_key=True),
            Column("code", String(16), nullable=False),
            Column("date_from", Date),
            Column("date_to", Date),
            schema="uplift",
        ),
        "client_spend": Table(
            "client_spend",
            metadata,
            Column("client_id", BigInteger, nullable=False),
            Column("category_id", SmallInteger, nullable=False),
            Column("window_id", SmallInteger, nullable=False),
            Column("amount", Numeric(14, 2), nullable=False),
            schema="uplift",
        ),
        "client_cashback": Table(
            "client_cashback",
            metadata,
            Column("client_id", BigInteger, nullable=False),
            Column("window_id", SmallInteger, nullable=False),
            # ENUM cashback_source из DDL: тип нужен, чтобы параметр биндился
            # как cashback_source, а не varchar (Postgres varchar -> enum не кастит)
            Column(
                "source",
                ENUM("partner", "bank", name="cashback_source", create_type=False),
                nullable=False,
            ),
            Column("amount", Numeric(14, 2), nullable=False),
            schema="uplift",
        ),
        "communication_stats": Table(
            "communication_stats",
            metadata,
            Column("client_id", BigInteger, primary_key=True),
            Column("window_id", SmallInteger, nullable=False),
            Column("cnt_communication", Integer, nullable=False),
            Column("cnt_view", Integer, nullable=False),
            Column("cnt_activation", Integer, nullable=False),
            schema="uplift",
        ),
    }

    for name in ("client", "spend_category", "observation_window"):
        _upsert(conn, tables[name], prepared[name])
    for name in ("client_spend", "client_cashback", "communication_stats"):
        _upsert(conn, tables[name], prepared[name])

    counts = {}
    for name, table in tables.items():
        counts[name] = int(conn.execute(select(func.count()).select_from(table)).scalar_one())
    return counts


def _upsert(conn: Connection, table: Table, frame: pd.DataFrame) -> None:
    """Идемпотентная вставка чанками: ON CONFLICT DO NOTHING повторяема."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    if frame.empty:
        return
    statement = pg_insert(table).on_conflict_do_nothing()
    records = frame.to_dict("records")
    for start in range(0, len(records), INSERT_CHUNK):
        conn.execute(statement, records[start : start + INSERT_CHUNK])


def load_raw(csv_path: str | Path, conn: Engine | Connection) -> dict[str, int]:
    """Читает CSV, валидирует и идемпотентно вставляет в схему uplift.

    Возвращает контрольные числа строк по таблицам (агрегаты, не сырые данные).
    Принимает Engine (своя транзакция) или Connection (транзакция на вызывающей стороне).
    """
    from sqlalchemy.engine import Engine

    frame = pd.read_csv(csv_path)
    prepared = prepare_frames(frame)
    if isinstance(conn, Engine):
        with conn.begin() as transaction:
            return _insert_frames(prepared, transaction)
    return _insert_frames(prepared, conn)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа `python -m uplift_system.db.load --csv <path>`.

    Код возврата: 0 — загружено, 1 — данные не прошли валидацию или БД недоступна.
    """
    parser = argparse.ArgumentParser(
        prog="uplift db-load",
        description="Загружает сырую выгрузку кэшбэка (CSV) в схему uplift PostgreSQL. "
        "Вставка идемпотентна: повторный прогон не меняет состояние.",
    )
    parser.add_argument("--csv", required=True, type=Path, help="путь к сырой выгрузке CSV")
    parser.add_argument(
        "--database-url",
        default=None,
        help="строка подключения (дефолт — DATABASE_URL или учебный стенд localhost)",
    )
    args = parser.parse_args(argv)

    try:
        from sqlalchemy.exc import SQLAlchemyError
    except ImportError:
        SQLAlchemyError = None  # type: ignore[assignment]

    from .connection import get_connection

    engine = get_connection(args.database_url)
    try:
        counts = load_raw(args.csv, engine)
    except RawDataError as exc:
        print(f"ошибка данных: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — любая ошибка драйвера/БД: короткое сообщение, не traceback
        if SQLAlchemyError is not None and isinstance(exc, SQLAlchemyError):
            print(f"ошибка базы данных: {exc}", file=sys.stderr)
            return 1
        raise
    for table, n_rows in counts.items():
        print(f"{table}: {n_rows} строк")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
