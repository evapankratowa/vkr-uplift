"""Тесты командного интерфейса: команда simulate (тикет T1).

Прогоны идут на малой сетке (1200 клиентов, 2 доли контроля, 2 повтора), чтобы
набор оставался быстрым. Реальная выгрузка кэшбэка не нужна: в публичном
репозитории её нет, и команда обязана работать через fallback на синтетические
ковариаты (проверяется предупреждением в логе).
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from uplift_system import cli

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _small_configs(tmp_path: Path, n_clients: int = 1200) -> tuple[Path, Path]:
    """Уменьшенные копии боевых конфигов: быстрый детерминированный прогон сетки."""
    sim = yaml.safe_load((PROJECT_DIR / "configs" / "sim_base.yaml").read_text(encoding="utf-8"))
    sim["sample"]["n_clients"] = n_clients
    exp = yaml.safe_load((PROJECT_DIR / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    exp["seed"] = 7
    exp["grid"] = {
        "n_total": [n_clients],
        "control_share": [0.5, 0.3],
        "effect_scale": [1.0],
        "repeats": 2,
    }
    exp["validation"] = {"aa_test": True, "aa_runs": 40}
    sim_path = tmp_path / "sim_small.yaml"
    exp_path = tmp_path / "exp_small.yaml"
    sim_path.write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")
    exp_path.write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    return sim_path, exp_path


def _run_simulate(sim_path: Path, exp_path: Path) -> int:
    """Запуск команды simulate на переданных конфигах."""
    return cli.main(["simulate", "--config", str(sim_path), "--experiment", str(exp_path)])


def test_top_level_help_lists_commands(capsys):
    """--help верхнего уровня показывает все пять команд."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("simulate", "train", "evaluate", "select-segment", "check"):
        assert command in out


def test_simulate_help_informative(capsys):
    """--help команды simulate описывает оба конфига."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["simulate", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--experiment" in out


def test_simulate_writes_parquet_and_prints_summary(tmp_path, monkeypatch, capsys, caplog):
    """Команда завершается нулём, пишет parquet на ячейку сетки и печатает сводку."""
    monkeypatch.setattr(cli, "DATA_INTERIM", tmp_path / "interim")
    sim_path, exp_path = _small_configs(tmp_path)
    with caplog.at_level(logging.INFO, logger="uplift"):
        code = _run_simulate(sim_path, exp_path)
    assert code == 0

    files = sorted((tmp_path / "interim").glob("experiment_*.parquet"))
    assert len(files) == 4  # 2 доли контроля × 2 повтора

    frame = pd.read_parquet(files[0])
    for column in ("treatment", "y_purchase", "y_activation", "ite_true_purchase", "segment"):
        assert column in frame.columns
    assert len(frame) == 1200

    out = capsys.readouterr().out
    assert "сводка" in out
    assert "ate" in out
    assert "aa_fpr" in out

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "конфиги загружены" in messages
    assert "старт сетки" in messages
    assert "готово" in messages
    # выгрузки кэшбэка в репозитории нет: сработал fallback с предупреждением
    assert "синтетическими" in messages


def test_simulate_repeatable_by_hash(tmp_path, monkeypatch):
    """Повторный запуск с тем же seed даёт побайтово идентичные parquet."""
    monkeypatch.setattr(cli, "DATA_INTERIM", tmp_path / "interim")
    sim_path, exp_path = _small_configs(tmp_path)
    hashes: dict[str, str] = {}
    for _ in range(2):
        assert _run_simulate(sim_path, exp_path) == 0
        current = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((tmp_path / "interim").glob("experiment_*.parquet"))
        }
        if not hashes:
            hashes = current
        else:
            assert current == hashes
    assert len(hashes) == 4


def test_simulate_reads_covariates_from_csv(tmp_path, monkeypatch, caplog):
    """При наличии CSV-выгрузки ковариаты читаются из неё, fallback не срабатывает."""
    monkeypatch.setattr(cli, "DATA_INTERIM", tmp_path / "interim")
    sim_path, exp_path = _small_configs(tmp_path)
    csv_path = tmp_path / "cashback_raw.csv"
    rng = pd.Series(range(1201, 2401))  # детерминированные «траты» без numpy-зависимости
    frame = pd.DataFrame(
        {
            "client_id": range(1, 1201),
            "cb_merch_before": rng % 997,
            "Supermarkets_before": (rng * 7) % 613,
            "cnt_view": (rng * 3) % 41,
            "cnt_activation": (rng * 5) % 17,
            "cnt_communication": (rng * 11) % 23,
        }
    )
    frame.to_csv(csv_path, index=False)
    sim = yaml.safe_load(sim_path.read_text(encoding="utf-8"))
    sim["covariates"]["source"] = str(csv_path)
    sim_path.write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="uplift"):
        assert _run_simulate(sim_path, exp_path) == 0
    assert not any("синтетическими" in record.getMessage() for record in caplog.records)
    assert len(list((tmp_path / "interim").glob("experiment_*.parquet"))) == 4
