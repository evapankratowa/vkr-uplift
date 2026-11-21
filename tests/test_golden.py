"""Золотые эталоны end-to-end (тикет T8).

Прогон пайплайна на малой конфигурации (2000 клиентов, одна ячейка сетки,
repeats: 1 — конфиги tests/golden/) и сверка ключевых чисел с закоммиченными
эталонами tests/golden/golden_metrics.json, допуск 1e-6 (seed зафиксирован,
CatBoost обучается в один поток — см. thread_count в configs/models.yaml).

Сверяются:
  - qini отложенной выборки и ценность@30% от оракула по каждой включённой
    архитектуре configs/models.yaml (результат evaluate);
  - итоговые числа economics: порог uplift, число контактов, число
    инкрементальных откликов и эффект в рублях по каждому каналу/сценарию.

Прогон идёт во временных каталогах: проектные data/, output/ и models_store/
тест не трогает, сырые данные и сеть не нужны. Переснять эталон после честного
изменения методики: UPLIFT_GOLDEN_UPDATE=1 pytest tests/test_golden.py.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from uplift_system import cli, paths, pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).parent / "golden"
SIM_CFG = GOLDEN_DIR / "sim_golden.yaml"
EXP_CFG = GOLDEN_DIR / "experiment_golden.yaml"
MODELS_CFG = PROJECT_DIR / "configs" / "models.yaml"
CAMPAIGN_CFG = PROJECT_DIR / "configs" / "campaign.yaml"
GOLDEN_JSON = GOLDEN_DIR / "golden_metrics.json"

TOL = 1e-6  # допуск сверки с эталоном (тикет T8)
UPDATE_ENV = "UPLIFT_GOLDEN_UPDATE"  # режим пересъёмки эталона

# Ключи эталона по архитектурам и каналам — английские, чтобы JSON было удобно
# дифать; порядок и состав сверяются с configs/models.yaml и campaign.yaml.
ARCH_METRICS = ("qini_holdout", "value_at_30pct")
ECON_METRICS = ("threshold", "n_contacts", "n_incremental", "effect_rub", "ci_low", "ci_high")


def _enabled_models() -> list[str]:
    """Имена включённых архитектур и baseline в порядке configs/models.yaml."""
    import yaml

    cfg = yaml.safe_load(MODELS_CFG.read_text(encoding="utf-8"))
    names = [
        item["name"]
        for item in cfg.get("architectures", []) + cfg.get("baselines", [])
        if item.get("enabled", True)
    ]
    assert names, "в configs/models.yaml нет включённых архитектур"
    return names


def _architecture_payload(table: pd.DataFrame) -> dict:
    """Метрики эталона по архитектурам: qini отложенной и ценность@30% от оракула."""
    payload = {}
    for row in table.to_dict("records"):
        payload[row["архитектура"]] = {
            "qini_holdout": float(row["qini отложенная"]),
            "value_at_30pct": float(row["ценность@30% от оракула"]),
        }
    return payload


def _economics_payload(table: pd.DataFrame) -> dict:
    """Итоговые числа эталона по каналам: порог, контакты, отклики, эффект с ДИ."""
    payload = {}
    for row in table.to_dict("records"):
        key = f"{row['канал']}|{row['сценарий']}"
        payload[key] = {
            "threshold": float(row["порог_uplift"]),
            "n_contacts": int(row["n_контактов"]),
            "n_incremental": float(row["n_инкр_откликов"]),
            "effect_rub": float(row["эффект_руб"]),
            "ci_low": float(row["эффект_ci_low_руб"]),
            "ci_high": float(row["эффект_ci_high_руб"]),
        }
    return payload


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory) -> dict:
    """Полный малый прогон simulate → train → evaluate → economics во временных каталогах.

    Пути проекта (data/interim, output, models_store) подменяются на временные,
    чтобы тест не зависел от прогонов make all и не писал в репозиторий; выборка
    для train/evaluate — parquet ячейки simulate, как в боевой цепочке make all.
    """
    tmp = tmp_path_factory.mktemp("golden")
    interim = tmp / "interim"
    tables = tmp / "tables"
    figures = tmp / "figures"
    store = tmp / "models_store"
    econ_csv = tmp / "economics.csv"

    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "DATA_INTERIM", interim)
    mp.setattr(pipeline, "DATA_INTERIM", interim)
    mp.setattr(pipeline, "MODELS_STORE", store)
    mp.setattr(paths, "OUT_TABLES", tables)
    mp.setattr(paths, "OUT_FIGURES", figures)
    try:
        assert cli.main(["simulate", "--config", str(SIM_CFG), "--experiment", str(EXP_CFG)]) == 0
        assert cli.main(["train", "--config", str(MODELS_CFG)]) == 0
        assert cli.main(["evaluate", "--config", str(MODELS_CFG)]) == 0
        code = cli.main(
            [
                "economics",
                "--config",
                str(CAMPAIGN_CFG),
                "--sim-config",
                str(SIM_CFG),
                "--n-boot",
                "1000",
                "--out",
                str(econ_csv),
            ]
        )
        assert code == 0
    finally:
        mp.undo()

    arch_csvs = sorted(tables.glob("*_architectures_*.csv"))
    pngs = sorted(figures.glob("*_qini_curves_*.png"))
    assert len(arch_csvs) == 1, f"ожидалась одна таблица архитектур, нашлось: {arch_csvs}"
    assert len(pngs) == 1, f"ожидался один график Qini, нашлось: {pngs}"
    arch_table = pd.read_csv(arch_csvs[0])
    econ_table = pd.read_csv(econ_csv)

    return {
        "interim": interim,
        "tables": tables,
        "figures": figures,
        "store": store,
        "arch_table": arch_table,
        "econ_table": econ_table,
        "payload": {
            "n_clients": 2000,
            "repeats": 1,
            "architectures": _architecture_payload(arch_table),
            "economics": _economics_payload(econ_table),
        },
    }


@pytest.fixture(scope="module")
def golden_reference(golden_run) -> dict:
    """Эталон из tests/golden/golden_metrics.json; в режиме UPDATE_ENV — перезапись."""
    if os.environ.get(UPDATE_ENV) == "1":
        GOLDEN_JSON.write_text(
            json.dumps(golden_run["payload"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"эталон переснят прогоном и записан в {GOLDEN_JSON}")
    if not GOLDEN_JSON.exists():
        pytest.fail(
            f"эталон {GOLDEN_JSON} отсутствует: переснимите прогоном "
            f"{UPDATE_ENV}=1 pytest tests/test_golden.py"
        )
    return json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))


def test_golden_pipeline_writes_artifacts(golden_run):
    """Малый прогон создаёт артефакты всех стадий: parquet, модели, таблицы, график."""
    assert len(list(golden_run["interim"].glob("experiment_*.parquet"))) == 1
    for name in _enabled_models():
        assert (golden_run["store"] / f"{name}.joblib").exists(), f"нет модели {name}"
    assert (golden_run["store"] / "manifest.json").exists()
    assert len(list(golden_run["tables"].glob("*_architectures_*.csv"))) == 1
    assert len(list(golden_run["figures"].glob("*_qini_curves_*.png"))) == 1
    assert golden_run["econ_table"].columns is not None  # таблица экономики прочитана


def test_golden_architecture_metrics(golden_run, golden_reference):
    """Qini отложенной и ценность@30% по каждой архитектуре — как в эталоне (1e-6)."""
    expected = golden_reference["architectures"]
    got = golden_run["payload"]["architectures"]
    # таблица evaluate отсортирована по ценности@30%, поэтому порядок отличается
    # от configs/models.yaml — сверяем состав, а не порядок
    assert set(got) == set(_enabled_models()), "состав архитектур разошёлся с configs/models.yaml"
    assert set(got) == set(expected), (
        f"состав архитектур прогона ({sorted(got)}) разошёлся с эталоном ({sorted(expected)})"
    )
    for name in got:
        for metric in ARCH_METRICS:
            assert got[name][metric] == pytest.approx(expected[name][metric], abs=TOL), (
                f"{name}.{metric}: {got[name][metric]} против эталона {expected[name][metric]}"
            )


def test_golden_economics_numbers(golden_run, golden_reference):
    """Итоговые числа economics по каналам/сценариям — как в эталоне (1e-6)."""
    expected = golden_reference["economics"]
    got = golden_run["payload"]["economics"]
    assert set(got) == set(expected), (
        f"состав строк economics ({sorted(got)}) разошёлся с эталоном ({sorted(expected)})"
    )
    for key in got:
        for metric in ECON_METRICS:
            assert got[key][metric] == pytest.approx(expected[key][metric], abs=TOL), (
                f"{key}.{metric}: {got[key][metric]} против эталона {expected[key][metric]}"
            )
