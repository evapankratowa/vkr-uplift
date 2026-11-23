"""Тесты слоя БД (тикет T3): загрузчик сырой выгрузки в схему uplift.

Сырого cashback-файла в репозитории нет (лицензия источника 1), поэтому
фикстура — синтетический CSV тех же 42 колонок (DATA_PASSPORT, db/001_init.sql),
1000 строк, seed фиксирован. Проверки — агрегатами: число строк по таблицам,
контрольные суммы, целостность кругового чтения; сырые строки в тестах
не сравниваются поштучно.

Тесты с пометкой `docker` прогоняются против postgres из docker-compose
(БД поднимается с DDL db/001_init.sql); без БД или без драйверов они
пропускаются. Быстрые проверки валидации и разборки CSV в pandas не требуют
ни БД, ни SQLAlchemy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_system.db import database_url
from uplift_system.db.load import (
    CASHBACK_COLUMNS,
    ID_COLUMN,
    WINDOWS,
    RawDataError,
    canonical_columns,
    categories_of,
    prepare_frames,
    validate_raw,
)

SEED = 42
N_ROWS = 1000

# 17 категорий трат источника 1 (DATA_PASSPORT); Taxi/Taxi_before в реальной
# выгрузке нулевые, здесь генерируются как обычные — на схему не влияет.
CATEGORIES = [
    "Air",
    "Beauty shop",
    "Book store",
    "Children",
    "Cinema",
    "Clothes",
    "Electronics",
    "Fastfood",
    "Fuel",
    "House",
    "Jewelry",
    "Other",
    "Pet shop",
    "Pharmacies",
    "Supermarkets",
    "Taxi",
    "Transport",
]


def make_raw_cashback_frame(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Синтетический сырой набор: те же колонки, что у выгрузки кэшбэка.

    Траты скошенные и с нулями («не покупал»), в кэшбэке ~2% отрицательных
    строк — сторно при возврате покупки (легальны по DDL). Деньги округлены
    до копеек, как numeric(14, 2). Никакого отношения к реальным данным нет.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {ID_COLUMN: np.arange(1_000_000, 1_000_000 + n)}
    for name in CATEGORIES:
        spend_before = rng.lognormal(5.0, 1.4, n) * rng.binomial(1, 0.55, n)
        spend_after = rng.lognormal(4.8, 1.4, n) * rng.binomial(1, 0.50, n)
        data[f"{name}_before"] = np.round(spend_before, 2)
        data[name] = np.round(spend_after, 2)
    for column in CASHBACK_COLUMNS:
        amount = rng.lognormal(5.5, 1.1, n) * rng.binomial(1, 0.60, n)
        storno = rng.random(n) < 0.02
        data[column] = np.where(
            storno, -np.round(rng.lognormal(5.0, 0.8, n), 2), np.round(amount, 2)
        )
    data["cnt_communication"] = rng.poisson(1.7, n)
    data["cnt_view"] = np.round(rng.lognormal(4.5, 1.0, n)).astype(int)
    data["cnt_activation"] = np.round(rng.lognormal(4.0, 0.9, n)).astype(int)
    return pd.DataFrame(data)[canonical_columns(CATEGORIES)]


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> Path:
    """Фикстурный CSV на 1000 строк — вход загрузчика во всех docker-тестах."""
    path = tmp_path_factory.mktemp("raw") / "cashback_fixture.csv"
    make_raw_cashback_frame().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def db_engine():
    """Движок SQLAlchemy к postgres из docker-compose; скип без БД/драйверов."""
    pytest.importorskip("sqlalchemy", reason="не установлена SQLAlchemy")
    pytest.importorskip("psycopg", reason="не установлен драйвер psycopg")
    from sqlalchemy import text

    from uplift_system.db import get_connection

    engine = get_connection(connect_timeout=3)
    try:
        with engine.connect() as conn:
            schema_ok = conn.execute(
                text("select 1 from information_schema.schemata where schema_name = 'uplift'")
            ).scalar()
    except Exception as exc:  # noqa: BLE001 — любая ошибка подключения: БД недоступна
        pytest.skip(f"postgres недоступен: {exc}")
    if not schema_ok:
        pytest.skip("в БД нет схемы uplift: примени db/001_init.sql (docker compose up -d db)")
    return engine


# ---------------------------------------------------------------------------
# Быстрые проверки без БД: разборка CSV и валидация ограничений схемы
# ---------------------------------------------------------------------------


def test_categories_and_windows_come_from_data():
    """Справочники собираются из самого файла: категории из колонок, окна фиксированы."""
    frame = make_raw_cashback_frame(n=20)
    assert categories_of(list(frame.columns)) == CATEGORIES
    prepared = prepare_frames(frame)
    assert prepared["spend_category"]["name"].tolist() == CATEGORIES
    assert prepared["spend_category"]["category_id"].tolist() == list(range(1, 18))
    assert prepared["observation_window"]["window_id"].tolist() == [w[0] for w in WINDOWS]
    assert prepared["observation_window"]["code"].tolist() == [w[1] for w in WINDOWS]


def test_prepare_frames_row_counts():
    """Разборка в длинные таблицы: по строке на факт; контрольные числа для 1000 клиентов."""
    prepared = prepare_frames(make_raw_cashback_frame())
    assert len(prepared["client"]) == N_ROWS
    assert len(prepared["client_spend"]) == N_ROWS * len(CATEGORIES) * 2
    assert len(prepared["client_cashback"]) == N_ROWS * len(CASHBACK_COLUMNS)
    assert len(prepared["communication_stats"]) == N_ROWS
    assert prepared["client_spend"]["amount"].ge(0).all()


def test_negative_spend_rejected_with_row_number():
    """Отрицательные суммы трат нарушают CHECK amount >= 0 — отказ с номером строки."""
    frame = make_raw_cashback_frame(n=50)
    bad_pos = 13
    frame.loc[bad_pos, "Fuel_before"] = -0.01
    with pytest.raises(RawDataError, match=rf"строка {bad_pos + 2}.*Fuel_before"):
        validate_raw(frame)


def test_negative_counter_rejected():
    """Отрицательные счётчики нарушают CHECK cnt_* >= 0 — отказ с номером строки."""
    frame = make_raw_cashback_frame(n=50)
    frame.loc[3, "cnt_view"] = -1
    with pytest.raises(RawDataError, match=r"строка 5.*cnt_view"):
        validate_raw(frame)


def test_duplicate_client_rejected():
    frame = make_raw_cashback_frame(n=50)
    frame.loc[7, ID_COLUMN] = frame.loc[0, ID_COLUMN]
    with pytest.raises(RawDataError, match=rf"строка 2.*{ID_COLUMN}"):
        validate_raw(frame)


def test_missing_column_rejected():
    frame = make_raw_cashback_frame(n=10).drop(columns=["cb_bank_last_3_month"])
    with pytest.raises(ValueError, match="cb_bank_last_3_month"):
        validate_raw(frame)


def test_negative_cashback_allowed():
    """Сторно кэшбэка при возврате — отрицательные значения легальны (комментарий DDL)."""
    prepared = prepare_frames(make_raw_cashback_frame(n=100))
    assert prepared["client_cashback"]["amount"].lt(0).any()


def test_database_url_from_env(monkeypatch):
    """Строка подключения: DATABASE_URL, иначе дефолт учебного стенда."""
    from uplift_system.db import DEFAULT_DATABASE_URL

    monkeypatch.delenv("DATABASE_URL", raising=False)  # compose передаёт переменную в контейнер
    assert database_url() == DEFAULT_DATABASE_URL
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db:5432/base")
    assert database_url() == "postgresql://user:pass@db:5432/base"
    assert database_url("postgresql://x@y/z") == "postgresql://x@y/z"


# ---------------------------------------------------------------------------
# Проверки против postgres (docker compose up -d db); без БД — пропуск
# ---------------------------------------------------------------------------


@pytest.mark.docker
class TestWithPostgres:
    """Идемпотентная загрузка фикстурного CSV и круговое чтение ковариат."""

    EXPECTED_COUNTS = {
        "client": N_ROWS,
        "spend_category": len(CATEGORIES),
        "observation_window": len(WINDOWS),
        "client_spend": N_ROWS * len(CATEGORIES) * 2,
        "client_cashback": N_ROWS * len(CASHBACK_COLUMNS),
        "communication_stats": N_ROWS,
    }

    def test_load_raw_control_counts(self, db_engine, csv_path):
        """Контрольное число строк в таблицах сходится с фикстурой (1000 клиентов)."""
        from uplift_system.db import load_raw

        counts = load_raw(csv_path, db_engine)
        assert counts == self.EXPECTED_COUNTS

    def test_load_raw_idempotent(self, db_engine, csv_path):
        """Повторный прогон не меняет состояние: ON CONFLICT DO NOTHING."""
        from uplift_system.db import load_raw

        load_raw(csv_path, db_engine)
        counts = load_raw(csv_path, db_engine)
        assert counts == self.EXPECTED_COUNTS

    def test_aggregates_match_fixture(self, db_engine, csv_path):
        """Контрольные суммы в БД равны суммам фикстуры (агрегаты, не сырые строки)."""
        from sqlalchemy import text

        from uplift_system.db import load_raw

        load_raw(csv_path, db_engine)
        frame = pd.read_csv(csv_path)
        with db_engine.connect() as conn:
            total_spend = conn.execute(
                text("select coalesce(sum(amount), 0) from uplift.client_spend")
            ).scalar()
            total_cashback = conn.execute(
                text("select coalesce(sum(amount), 0) from uplift.client_cashback")
            ).scalar()
        spend_cols = [f"{c}_before" for c in CATEGORIES] + CATEGORIES
        assert float(total_spend) == pytest.approx(frame[spend_cols].to_numpy().sum())
        assert float(total_cashback) == pytest.approx(frame[list(CASHBACK_COLUMNS)].sum().sum())

    def test_read_raw_roundtrip(self, db_engine, csv_path):
        """Круговое чтение: read_raw собирает те же ковариаты, что были в CSV."""
        from uplift_system.db import load_raw, read_raw

        load_raw(csv_path, db_engine)
        restored = read_raw(db_engine).sort_values(ID_COLUMN).reset_index(drop=True)
        expected = pd.read_csv(csv_path).sort_values(ID_COLUMN).reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected)

    def test_cli_main_happy_path(self, db_engine, csv_path, capsys):
        """python -m uplift_system.db.load --csv ...: код 0 и сводка строк по таблицам."""
        from uplift_system.db.load import main

        assert main(["--csv", str(csv_path)]) == 0
        out = capsys.readouterr().out
        for table, n_rows in self.EXPECTED_COUNTS.items():
            assert f"{table}: {n_rows} строк" in out

    def test_cli_main_bad_csv(self, db_engine, tmp_path, capsys):
        """Битая строка — код 1 и понятное сообщение с номером строки, в БД ничего не пишется."""
        from uplift_system.db.load import main

        frame = make_raw_cashback_frame(n=10)
        frame.loc[4, "Supermarkets"] = -5.0
        bad_csv = tmp_path / "bad.csv"
        frame.to_csv(bad_csv, index=False)
        assert main(["--csv", str(bad_csv)]) == 1
        assert "строка 6" in capsys.readouterr().err
