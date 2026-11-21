"""Команды `select-segment` и `economics` в CLI (тикеты T6 и T4-интеграция).

select-segment прогоняется на маленькой синтетической аудитории и фикстурной
модели-заглушке из временного models_store: реальная выгрузка и обучение не
нужны — набор обязан проходить на чистой машине и в CI. Проверяются критерии
приёмки T6: оба файловых формата (csv и parquet), оба формата выгрузки
(standard и ab), бюджет контактов и непересечение контрольной группы с целевым
сегментом. economics проверяется тонким вызовом economics.run.main на малой
конфигурации: сводная таблица пишется и печатается, три сценария SMS строками.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from uplift_system import cli
from uplift_system.paths import PROJECT_PREFIX
from uplift_system.segments import builder, export

PROJECT_DIR = Path(__file__).resolve().parents[1]
AUDIENCE_N = 800  # аудитория больше бюджета 200: лимит достижим


class _ToyModel:
    """Детерминированная модель-заглушка: predict_uplift без обучения."""

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.linspace(0.0005, 0.05, len(features))


@pytest.fixture()
def campaign_path(tmp_path: Path) -> Path:
    """Уменьшенная копия configs/campaign.yaml: бюджет 200 контактов."""
    cfg = yaml.safe_load((PROJECT_DIR / "configs" / "campaign.yaml").read_text(encoding="utf-8"))
    cfg["constraints"]["budget_contacts"] = 200
    path = tmp_path / "campaign_small.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def sim_path(tmp_path: Path) -> Path:
    """Уменьшенная копия configs/sim_base.yaml: аудитория 800 клиентов."""
    cfg = yaml.safe_load((PROJECT_DIR / "configs" / "sim_base.yaml").read_text(encoding="utf-8"))
    cfg["sample"]["n_clients"] = AUDIENCE_N
    path = tmp_path / "sim_small.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def models_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Временный models_store с фикстурной моделью campaign.model (x_learner)."""
    directory = tmp_path / "models_store"
    directory.mkdir()
    joblib.dump(_ToyModel(), directory / "x_learner.joblib")
    monkeypatch.setattr(builder, "MODELS_DIR", directory)
    return directory


@pytest.fixture()
def out_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Выгрузки сегмента — во временный output/tables, проектный каталог не трогаем."""
    directory = tmp_path / "out_tables"
    monkeypatch.setattr(export, "OUT_TABLES", directory)
    return directory


def _run_select_segment(campaign: Path, sim: Path) -> int:
    """Запуск команды select-segment на переданных конфигах."""
    return cli.main(["select-segment", "--config", str(campaign), "--sim-config", str(sim)])


def test_select_segment_help_informative(capsys):
    """--help команды select-segment описывает оба конфига."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["select-segment", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--sim-config" in out


def test_select_segment_writes_standard_and_ab_in_both_formats(
    models_store, out_tables, campaign_path, sim_path, capsys
):
    """Команда создаёт standard и ab в csv и parquet; сегмент не больше бюджета."""
    assert _run_select_segment(campaign_path, sim_path) == 0

    cfg = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign_id = cfg["campaign"]["id"]
    budget = cfg["constraints"]["budget_contacts"]
    files = {p.name for p in out_tables.iterdir()}
    assert files == {
        f"{PROJECT_PREFIX}_segment_{campaign_id}.csv",
        f"{PROJECT_PREFIX}_segment_{campaign_id}.parquet",
        f"{PROJECT_PREFIX}_segment_ab_{campaign_id}.csv",
        f"{PROJECT_PREFIX}_segment_ab_{campaign_id}.parquet",
    }

    # standard: ровно под бюджет, лучшие по uplift первыми, колонки из конфига
    standard = pd.read_csv(out_tables / f"{PROJECT_PREFIX}_segment_{campaign_id}.csv")
    assert len(standard) == budget <= AUDIENCE_N
    assert standard["uplift_score"].is_monotonic_decreasing
    expected_columns = [c for c in cfg["export"]["columns"] if c in standard.columns]
    assert list(standard.columns) == expected_columns
    assert "campaign_id" in standard.columns
    standard_pq = pd.read_parquet(out_tables / f"{PROJECT_PREFIX}_segment_{campaign_id}.parquet")
    assert len(standard_pq) == len(standard)
    assert set(standard_pq["client_id"]) == set(standard["client_id"])

    # ab: целевая часть совпадает с standard, контроль не пересекается с ней
    ab = pd.read_csv(out_tables / f"{PROJECT_PREFIX}_segment_ab_{campaign_id}.csv")
    assert list(ab.columns) == [*expected_columns, export.GROUP_COLUMN]
    target = ab[ab[export.GROUP_COLUMN] == export.GROUP_TARGET]
    control = ab[ab[export.GROUP_COLUMN] == export.GROUP_CONTROL]
    assert set(target["client_id"]) == set(standard["client_id"])
    assert set(control["client_id"]).isdisjoint(set(target["client_id"]))
    share = cfg["export"]["control_share"]
    assert len(control) == round(len(target) * share) > 0
    assert control["uplift_score"].isna().all()  # контроль не про отбор: скоров у него нет
    ab_pq = pd.read_parquet(out_tables / f"{PROJECT_PREFIX}_segment_ab_{campaign_id}.parquet")
    assert len(ab_pq) == len(ab)

    out = capsys.readouterr().out
    assert "сегмент кампании" in out
    assert str(out_tables) in out


def test_select_segment_not_larger_than_audience(
    models_store, out_tables, campaign_path, sim_path, tmp_path
):
    """Бюджет больше аудитории: сегмент равен аудитории и не превышает бюджет."""
    cfg = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    cfg["constraints"]["budget_contacts"] = 10**9
    big_budget = tmp_path / "campaign_big_budget.yaml"
    big_budget.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    assert _run_select_segment(big_budget, sim_path) == 0
    standard = pd.read_csv(out_tables / f"{PROJECT_PREFIX}_segment_{cfg['campaign']['id']}.csv")
    assert len(standard) == AUDIENCE_N
    assert len(standard) <= cfg["constraints"]["budget_contacts"]


def test_select_segment_missing_model_raises(tmp_path, monkeypatch, campaign_path, sim_path):
    """Без обученной модели — понятная ошибка, а не молчаливый провал."""
    monkeypatch.setattr(builder, "MODELS_DIR", tmp_path / "empty_models_store")
    with pytest.raises(FileNotFoundError, match="x_learner"):
        _run_select_segment(campaign_path, sim_path)


def test_economics_help_informative(capsys):
    """--help команды economics описывает конфиг и бутстрап."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["economics", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--n-boot" in out


def test_economics_command_writes_summary_table(tmp_path, sim_path, capsys):
    """Тонкий вызов: команда пишет сводную таблицу и дублирует её в stdout."""
    out_csv = tmp_path / "economics.csv"
    code = cli.main(
        [
            "economics",
            "--config",
            str(PROJECT_DIR / "configs" / "campaign.yaml"),
            "--sim-config",
            str(sim_path),
            "--n-boot",
            "20",
            "--out",
            str(out_csv),
        ]
    )
    assert code == 0
    assert out_csv.exists()

    table = pd.read_csv(out_csv)
    sms = table[table["канал"] == "sms"].sort_values("цена_контакта_руб")
    assert len(sms) == 3  # три сценария цены SMS отдельными строками
    assert sms["цена_контакта_руб"].tolist() == [3.2, 5.75, 30.0]

    out = capsys.readouterr().out
    assert "экономика каналов" in out
    assert "push" in out
