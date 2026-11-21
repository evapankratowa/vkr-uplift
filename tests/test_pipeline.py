"""Пайплайн обучения и оценки: команды train и evaluate (тикет T2).

Прогоны идут на малом смоделированном эксперименте (4000 клиентов, быстрый
CatBoost из фикстуры models_cfg). Мост --source db проверяется моком: read_raw
подменяется фикстурным DataFrame с колонками сырого CSV, подключение — заглушкой
без SQLAlchemy. Живых тестов против PostgreSQL здесь нет — они тикета T3.
"""

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from uplift_system import cli, pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Эталон — output/tables/VKR_uplift_architectures_2026-10-01.csv внутреннего
# репозитория: набор и порядок колонок evaluate обязаны совпадать с ним.
REFERENCE_COLUMNS = [
    "архитектура",
    "qini обучение",
    "qini отложенная",
    "uplift@30%, п.п.",
    "PEHE",
    "tau против истины",
    "ценность@30% от оракула",
    "сек",
]

ENABLED_MODELS = [
    "s_learner",
    "t_learner",
    "x_learner",
    "class_transformation",
    "random",
    "response_model",
]

METRIC_COLUMNS = [c for c in REFERENCE_COLUMNS if c != "сек"]


def _fast_models_cfg(models_cfg: dict) -> dict:
    """Урезанный конфиг моделей: короткий бустинг, полный набор архитектур."""
    cfg = yaml.safe_load(yaml.safe_dump(models_cfg))  # глубокая копия без import copy
    cfg["base_learner"]["params"]["iterations"] = 25
    return cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory, experiment, models_cfg) -> dict:
    """Один прогон train во временный каталог: joblib и манифест для всех тестов."""
    store = tmp_path_factory.mktemp("models_store")
    cfg = _fast_models_cfg(models_cfg)
    config_path = store / "models.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    train, _holdout = pipeline.split_for_evaluation(experiment, cfg, {})
    stats = pipeline.train_models(cfg, train, models_dir=store)
    manifest_path = pipeline.write_manifest(cfg, train, config_path, stats, models_dir=store)
    return {
        "store": store,
        "cfg": cfg,
        "config_path": config_path,
        "stats": stats,
        "manifest_path": manifest_path,
        "dataset": experiment,
    }


def test_train_writes_joblib_per_architecture(trained):
    """Каждая включённая архитектура и baseline сериализуются в <имя>.joblib."""
    for name in ENABLED_MODELS:
        path = trained["store"] / f"{name}.joblib"
        assert path.exists(), f"нет файла {path.name}"
        model = joblib.load(path)
        assert callable(model.predict_uplift)
    assert trained["stats"]["архитектура"].tolist() == ENABLED_MODELS
    assert trained["stats"]["сек"].ge(0).all()


def test_manifest_holds_model_params_config_hash_and_date(trained):
    """Манифест: модель, параметры, хеш конфига, дата."""
    manifest = json.loads(trained["manifest_path"].read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(trained["config_path"].read_bytes()).hexdigest()
    assert manifest["config_hash"] == expected_hash
    assert manifest["date"] == date.today().isoformat()
    assert manifest["seed"] == trained["cfg"]["seed"]
    assert manifest["dataset"]["n_train"] > 0
    names = [item["name"] for item in manifest["models"]]
    assert names == ENABLED_MODELS
    for item in manifest["models"]:
        assert item["file"] == f"{item['name']}.joblib"
        assert item["params"] == trained["cfg"]["base_learner"]["params"]
        assert item["seconds"] >= 0


def test_evaluate_table_matches_reference_columns(trained):
    """Таблица evaluate: колонки эталона, PEHE у baseline пуст, сортировка по ценности."""
    result = pipeline.run_evaluation(trained["cfg"], trained["dataset"], trained["store"], {})
    table = result["table"]
    assert list(table.columns) == REFERENCE_COLUMNS
    assert sorted(table["архитектура"].tolist()) == sorted(ENABLED_MODELS)
    assert table["ценность@30% от оракула"].is_monotonic_decreasing  # сортировка по ценности
    baseline = table["архитектура"].isin(["random", "response_model"])
    assert table.loc[baseline, "PEHE"].isna().all()
    assert table.loc[~baseline, "PEHE"].notna().all()
    assert table.loc[~baseline, "tau против истины"].between(-1, 1).all()
    assert table["сек"].notna().all()  # секунды взяты из манифеста


def test_evaluate_is_repeatable(trained):
    """Повторный evaluate по тем же моделям даёт идентичные метрики."""
    first = pipeline.run_evaluation(trained["cfg"], trained["dataset"], trained["store"], {})[
        "table"
    ]
    second = pipeline.run_evaluation(trained["cfg"], trained["dataset"], trained["store"], {})[
        "table"
    ]
    pd.testing.assert_frame_equal(first[METRIC_COLUMNS], second[METRIC_COLUMNS])


def test_evaluate_without_models_raises_with_hint(tmp_path, trained):
    """Без joblib оценка отказывает с понятной подсказкой про uplift train."""
    cfg = trained["cfg"]
    with pytest.raises(FileNotFoundError, match="uplift train"):
        pipeline.run_evaluation(cfg, trained["dataset"], tmp_path, {})


def _write_interim_parquet(path: Path, experiment, n_rows: int) -> None:
    """Parquet в формате ячейки simulate (cli._write_cell) на n первых клиентов."""
    data = experiment
    frame = data.features.head(n_rows).copy()
    frame["treatment"] = data.treatment.head(n_rows)
    for outcome in ("purchase", "activation"):
        frame[f"y_{outcome}"] = data.outcomes[outcome].head(n_rows)
        frame[f"ite_true_{outcome}"] = data.ite_true[outcome].head(n_rows)
    frame["segment"] = data.segment.head(n_rows)
    frame.to_parquet(path)


def test_train_reads_latest_interim_parquet(tmp_path, monkeypatch, experiment):
    """Без --dataset train берёт самый свежий parquet из data/interim."""
    interim = tmp_path / "interim"
    interim.mkdir()
    _write_interim_parquet(interim / "experiment_a_2026-01-01.parquet", experiment, 400)
    newer = interim / "experiment_b_2026-01-02.parquet"
    _write_interim_parquet(newer, experiment, 700)
    newer.touch()  # гарантированно свежее по mtime
    monkeypatch.setattr(pipeline, "DATA_INTERIM", interim)

    dataset = pipeline.load_training_dataset(None, "file", {}, {})
    assert dataset.n == 700
    assert dataset.name == newer.stem
    assert set(np.unique(dataset.t)) == {0, 1}
    assert dataset.has_ground_truth
    assert dataset.segment is not None
    reserved = {
        "treatment",
        "segment",
        "y_purchase",
        "y_activation",
        "ite_true_purchase",
        "ite_true_activation",
    }
    assert not reserved.intersection(dataset.features.columns)
    expected_y = experiment.outcomes["purchase"].head(700).to_numpy()
    np.testing.assert_array_equal(dataset.y("purchase"), expected_y)


def test_train_without_parquet_falls_back_to_synthetic(tmp_path, monkeypatch, sim_cfg, caplog):
    """Пустой data/interim: честный fallback на синтетику с предупреждением."""
    monkeypatch.setattr(pipeline, "DATA_INTERIM", tmp_path / "empty_interim")
    exp_cfg = yaml.safe_load(
        (PROJECT_DIR / "configs" / "experiment.yaml").read_text(encoding="utf-8")
    )
    with caplog.at_level(logging.WARNING, logger="uplift"):
        dataset = pipeline.load_training_dataset(None, "file", sim_cfg, exp_cfg)
    assert dataset.treatment is not None
    assert dataset.has_ground_truth
    assert "синтетически" in " ".join(r.getMessage() for r in caplog.records)


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeEngine:
    """Заглушка SQLAlchemy Engine: драйвер в тестовом окружении не установлен."""

    def connect(self):
        return _FakeConnection()


def _raw_cashback_frame(n: int = 2000, seed: int = 3) -> pd.DataFrame:
    """Сырой набор колонок CSV как из read_raw: траты до окна, кэшбэк, счётчики."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "client_id": np.arange(1_000_000, 1_000_000 + n),
            "Supermarkets_before": rng.lognormal(5.0, 1.4, n),
            "cb_merch_before": rng.lognormal(5.5, 1.1, n),
            "cb_bank_before": rng.lognormal(5.0, 1.2, n),
            "cnt_communication": rng.poisson(1.7, n),
            "cnt_view": np.round(rng.lognormal(4.5, 1.0, n)).astype(int),
            "cnt_activation": np.round(rng.lognormal(4.0, 0.9, n)).astype(int),
        }
    )
    return frame


def test_covariates_from_frame_mirrors_file_transform(sim_cfg):
    """Мост read_raw → признаки: окно «до» + счётчики, log1p, индекс client_id."""
    raw = _raw_cashback_frame(n=50)
    cov_cfg = sim_cfg["covariates"]
    features = pipeline.covariates_from_frame(raw, cov_cfg)
    assert features.index.name == cov_cfg["id_column"]
    assert list(features.columns) == [
        "Supermarkets_before",
        "cb_merch_before",
        "cb_bank_before",
        "cnt_view",
        "cnt_activation",
        "cnt_communication",
    ]
    expected = np.log1p(raw[features.columns].astype(float).clip(lower=0))
    np.testing.assert_allclose(features.to_numpy(), expected.to_numpy())


def test_source_db_uses_read_raw_bridge(monkeypatch, sim_cfg):
    """--source db: ковариаты идут через db.read_raw, DGP — тот же, что у файла."""
    raw = _raw_cashback_frame()
    calls = {"n": 0}

    def fake_read_raw(conn):
        calls["n"] += 1
        return raw

    monkeypatch.setattr("uplift_system.db.read.read_raw", fake_read_raw)
    monkeypatch.setattr("uplift_system.db.connection.get_connection", lambda *a, **k: _FakeEngine())

    exp_cfg = yaml.safe_load(
        (PROJECT_DIR / "configs" / "experiment.yaml").read_text(encoding="utf-8")
    )
    dataset = pipeline.load_training_dataset("synthetic", "db", sim_cfg, exp_cfg)
    assert calls["n"] == 1
    assert dataset.n == len(raw)
    assert dataset.treatment is not None
    assert dataset.has_ground_truth
    control_share = float(exp_cfg["grid_proxy_mix"]["control_share"])
    assert dataset.meta["treat_prob"] == pytest.approx(1.0 - control_share)


def test_cli_train_and_evaluate_end_to_end(tmp_path, monkeypatch, capsys, caplog, experiment):
    """CLI: train пишет joblib + манифест, evaluate — CSV эталонных колонок и RGB-график."""
    monkeypatch.setattr(pipeline, "MODELS_STORE", tmp_path / "models_store")
    monkeypatch.setattr(pipeline, "DATA_INTERIM", tmp_path / "empty_interim")
    monkeypatch.setattr(pipeline, "load_training_dataset", lambda *a, **k: experiment)
    monkeypatch.setattr("uplift_system.paths.OUT_TABLES", tmp_path / "tables")
    monkeypatch.setattr("uplift_system.paths.OUT_FIGURES", tmp_path / "figures")

    cfg = _fast_models_cfg(
        yaml.safe_load((PROJECT_DIR / "configs" / "models.yaml").read_text(encoding="utf-8"))
    )
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="uplift"):
        assert cli.main(["train", "--config", str(config_path)]) == 0
        store = tmp_path / "models_store"
        for name in ENABLED_MODELS:
            assert (store / f"{name}.joblib").exists()
        manifest = json.loads((store / pipeline.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["config_hash"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert "обученные модели" in capsys.readouterr().out

        assert cli.main(["evaluate", "--config", str(config_path)]) == 0
    csv_files = sorted((tmp_path / "tables").glob("*_architectures_*.csv"))
    png_files = sorted((tmp_path / "figures").glob("*_qini_curves_*.png"))
    assert len(csv_files) == 1
    assert len(png_files) == 1

    table = pd.read_csv(csv_files[0])
    assert list(table.columns) == REFERENCE_COLUMNS

    from PIL import Image

    with Image.open(png_files[0]) as img:
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 255, 255)  # белый фон

    out = capsys.readouterr().out
    assert "метрики архитектур" in out
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "таблица" in messages
    assert "график" in messages


def test_train_evaluate_help_mentions_dataset_and_source(capsys):
    """--help train и evaluate описывает --dataset и --source."""
    for command in ("train", "evaluate"):
        with pytest.raises(SystemExit) as exc:
            cli.main([command, "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--dataset" in out
        assert "--source" in out
