"""Разбор открытых наборов: схемы из документации, без обращения к сети.

Файлы подделываются РОВНО по опубликованным схемам Hillstrom и X5 RetailHero.
Тест проверяет разбор и приведение к контракту UpliftDataset, а не сами данные:
скачивание требует сети и в тестах не участвует.
"""

import gzip

import numpy as np
import pandas as pd
import pytest

from uplift_system.datasets.open_data import load_hillstrom, load_x5


@pytest.fixture
def hillstrom_file(tmp_path):
    """Hillstrom: колонки по описанию MineThatData, три ветки рассылки."""
    rng = np.random.default_rng(0)
    n = 3000
    frame = pd.DataFrame(
        {
            "Recency": rng.integers(1, 13, n),
            "History_Segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
            "History": rng.lognormal(4.5, 1.0, n).round(2),
            "Mens": rng.binomial(1, 0.55, n),
            "Womens": rng.binomial(1, 0.55, n),
            "Zip_Code": rng.choice(["Urban", "Suburban", "Rural"], n),
            "Newbie": rng.binomial(1, 0.5, n),
            "Channel": rng.choice(["Web", "Phone", "Multichannel"], n),
            "Segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
        }
    )
    frame["Visit"] = rng.binomial(1, 0.15, n)
    frame["Conversion"] = frame["Visit"] * rng.binomial(1, 0.06, n)
    frame["Spend"] = frame["Conversion"] * 100.0
    path = tmp_path / "hillstrom_raw.csv.gz"
    with gzip.open(path, "wt", encoding="utf8") as fh:
        frame.to_csv(fh, index=False)
    return tmp_path


@pytest.fixture
def x5_files(tmp_path):
    """X5: uplift_train.csv и clients.csv по описанию соревнования."""
    rng = np.random.default_rng(0)
    n = 2000
    ids = np.array([f"c{i:06d}" for i in range(n)])
    train = pd.DataFrame({"client_id": ids, "treatment_flg": rng.binomial(1, 0.5, n)})
    train["target"] = rng.binomial(1, 0.62, n)
    clients = pd.DataFrame(
        {
            "client_id": ids,
            "first_issue_date": pd.to_datetime("2017-01-01")
            + pd.to_timedelta(rng.integers(0, 900, n), "D"),
            "first_redeem_date": pd.to_datetime("2017-06-01")
            + pd.to_timedelta(rng.integers(0, 700, n), "D"),
            "age": rng.integers(18, 80, n),
            "gender": rng.choice(["F", "M", "U"], n),
        }
    )
    clients.loc[rng.random(n) < 0.15, "first_redeem_date"] = pd.NaT
    for name, frame in (("x5_uplift_train_raw.csv.gz", train), ("x5_clients_raw.csv.gz", clients)):
        with gzip.open(tmp_path / name, "wt", encoding="utf8") as fh:
            frame.to_csv(fh, index=False)
    return tmp_path


def test_hillstrom_keeps_two_arms(hillstrom_file):
    """Берутся только выбранная ветка и «No E-Mail»; третья выбрасывается целиком."""
    data = load_hillstrom(data_dir=hillstrom_file, download_if_missing=False)
    assert set(np.unique(data.t)) == {0, 1}
    assert data.n < 3000, "третья ветка не выброшена"
    assert data.meta["treat_prob"] == pytest.approx(0.5, abs=0.05)
    assert data.ite_true is None, "у реального эксперимента истинного CATE быть не может"


def test_hillstrom_features_are_numeric(hillstrom_file):
    """Категориальные колонки развёрнуты в индикаторы, матрица числовая."""
    data = load_hillstrom(data_dir=hillstrom_file, download_if_missing=False)
    assert all(pd.api.types.is_numeric_dtype(data.features[c]) for c in data.features.columns)
    assert any(c.startswith("zip_code_") for c in data.features.columns)
    assert not data.features.isna().to_numpy().any()


def test_hillstrom_target_switch(hillstrom_file):
    """Целевая колонка выбирается параметром: visit или conversion."""
    visit = load_hillstrom(data_dir=hillstrom_file, download_if_missing=False, target_col="visit")
    conv = load_hillstrom(
        data_dir=hillstrom_file, download_if_missing=False, target_col="conversion"
    )
    assert visit.y("purchase").mean() > conv.y("purchase").mean()


def test_x5_merges_one_to_one(x5_files):
    """Обучающая часть и клиенты соединяются один к одному, ключ уникален."""
    data = load_x5(data_dir=x5_files, download_if_missing=False)
    assert data.n == 2000
    assert data.id.is_unique
    assert data.meta["purchases_used"] is False


def test_x5_dates_become_recency(x5_files):
    """Даты превращаются в давность и флаг пропуска, абсолютных дат в признаках нет."""
    data = load_x5(data_dir=x5_files, download_if_missing=False)
    assert any(c.startswith("days_since_") for c in data.features.columns)
    assert any(c.endswith("_missing") for c in data.features.columns)
    assert not any(
        "date" in c and not c.startswith("days_since_") and not c.endswith("_missing")
        for c in data.features.columns
    )
    assert all(pd.api.types.is_numeric_dtype(data.features[c]) for c in data.features.columns)


def test_missing_file_raises_without_download(tmp_path):
    """Без сети и без файла — внятная ошибка, а не попытка молча скачать."""
    with pytest.raises(FileNotFoundError):
        load_hillstrom(data_dir=tmp_path, download_if_missing=False)


def test_x5_drops_post_treatment_features(x5_files):
    """Дата первого списания бонусов выброшена как пост-воздействие (решение №32)."""
    data = load_x5(data_dir=x5_files, download_if_missing=False)
    assert "days_since_first_redeem_date" not in data.features.columns
    assert "first_redeem_date_missing" not in data.features.columns
    assert set(data.meta["post_treatment_features_dropped"]) == {
        "days_since_first_redeem_date",
        "first_redeem_date_missing",
    }
    naive = load_x5(data_dir=x5_files, download_if_missing=False, include_post_treatment=True)
    assert "days_since_first_redeem_date" in naive.features.columns
    assert naive.features.shape[1] == data.features.shape[1] + 2


def test_x5_single_reference_date_for_all_recencies(x5_files):
    """Опорная дата одна на все колонки с датами, а не нарастающий максимум.

    Регрессия на найденный дефект: при нарастающем максимуме первая колонка
    отсчитывалась от своего максимума, вторая — от общего, и давности
    оказывались на разных базах, зависящих от порядка колонок в файле.
    """
    import gzip

    naive = load_x5(data_dir=x5_files, download_if_missing=False, include_post_treatment=True)
    reference = pd.Timestamp(naive.meta["reference_date"])

    with gzip.open(x5_files / "x5_clients_raw.csv.gz", "rt", encoding="utf8") as fh:
        clients = pd.read_csv(fh)
    issue = pd.to_datetime(clients["first_issue_date"])
    redeem = pd.to_datetime(clients["first_redeem_date"])
    assert reference == max(issue.max(), redeem.max()), "опорная дата не глобальный максимум"

    # обе давности отсчитаны от одной и той же даты: минимум каждой равен
    # расстоянию от опорной даты до максимума своей колонки. Пропуски в матрице
    # признаков заполнены -1, поэтому считаем минимум только по заполненным.
    for column, parsed in (("first_issue_date", issue), ("first_redeem_date", redeem)):
        expected_min = (reference - parsed.max()).days
        present = naive.features[f"{column}_missing"] == 0
        assert naive.features.loc[present, f"days_since_{column}"].min() == expected_min
