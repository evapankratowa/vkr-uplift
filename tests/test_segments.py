"""Сборщик сегментов: бюджет, ограничения, выгрузка в форматах standard/ab.

Тесты идут на маленьких синтетических скорах — реальная выгрузка и обучение
моделей не нужны: набор обязан проходить на чистой машине и в CI.
Контракт выгрузки — docs/EXTERNAL_INTEGRATIONS.md, раздел 3.
"""

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from uplift_system.config import load_config
from uplift_system.paths import CONFIGS, PROJECT_PREFIX
from uplift_system.segments import builder, export, rules


class _ToyModel:
    """Детерминированная модель-заглушка для score_clients без обучения."""

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.linspace(-0.01, 0.05, len(features))


@pytest.fixture()
def campaign_cfg() -> dict:
    """Конфиг кампании из configs/; тесты меняют копию, файл не трогают."""
    return load_config(CONFIGS / "campaign.yaml")


def _make_scores(
    n: int = 2000, seed: int = 1, negative_share: float = 0.0, with_risk: bool = False
) -> pd.DataFrame:
    """Скоры сегмента: client_id, uplift_score; опционально risk_ci_high."""
    rng = np.random.default_rng(seed)
    uplift = rng.uniform(0.0005, 0.05, n)
    if negative_share > 0:
        negative = rng.random(n) < negative_share
        uplift[negative] = -rng.uniform(0.0005, 0.02, int(negative.sum()))
    frame = pd.DataFrame({"client_id": np.arange(1_000_000, 1_000_000 + n), "uplift_score": uplift})
    if with_risk:
        frame["risk_ci_high"] = rng.uniform(0.0001, 0.01, n)
    return frame


def test_budget_respected(campaign_cfg: dict) -> None:
    """Размер отобранного сегмента не превышает budget_contacts."""
    cfg = copy.deepcopy(campaign_cfg)
    budget = cfg["constraints"]["budget_contacts"]
    scores = _make_scores(n=budget + 5000)  # аудитория больше бюджета: лимит достижим
    selected = builder.select(scores, cfg)
    assert len(selected) <= budget
    assert len(selected) == budget
    assert selected["uplift_score"].is_monotonic_decreasing  # лучшие по uplift вошли первыми


def test_no_client_in_two_campaigns(campaign_cfg: dict) -> None:
    """При max_campaigns_per_client=1 пересечение сегментов пустое."""
    cfg_a = copy.deepcopy(campaign_cfg)
    cfg_a["constraints"]["budget_contacts"] = 300
    segment_a = builder.select(_make_scores(n=2000), cfg_a)

    cfg_b = copy.deepcopy(campaign_cfg)
    cfg_b["constraints"]["budget_contacts"] = 300
    cfg_b["constraints"]["exclude_segments"] = [segment_a]  # уже выбранные не занимаем
    segment_b = builder.select(_make_scores(n=2000), cfg_b)

    assert rules.check_overlap({"a": segment_a, "b": segment_b}).empty

    cfg_bare = copy.deepcopy(campaign_cfg)  # без исключений те же места заняли бы те же клиенты
    cfg_bare["constraints"]["budget_contacts"] = 300
    segment_bare = builder.select(_make_scores(n=2000), cfg_bare)
    assert not rules.check_overlap({"a": segment_a, "b": segment_bare}).empty


def test_negative_uplift_excluded(campaign_cfg: dict, tmp_path: Path) -> None:
    """Клиенты с uplift ниже min_uplift не попадают в выгрузку."""
    cfg = copy.deepcopy(campaign_cfg)
    cfg["constraints"]["budget_contacts"] = 10**9  # связывает отсечка, не бюджет
    min_uplift = cfg["constraints"]["min_uplift"]
    scores = _make_scores(n=2000, negative_share=0.3)
    assert (scores["uplift_score"] < min_uplift).any()  # в данных есть кого отсекать

    selected = builder.select(scores, cfg)
    path = export.export_segment(selected, cfg["campaign"]["id"], out_dir=tmp_path)
    back = pd.read_csv(path)
    assert len(back) == int((scores["uplift_score"] >= min_uplift).sum())
    assert back["uplift_score"].min() >= min_uplift


def test_risk_threshold_respected(campaign_cfg: dict) -> None:
    """Верхняя граница ДИ uplift по выплатам банка не превышает risk_uplift_max."""
    cfg = copy.deepcopy(campaign_cfg)
    cfg["constraints"]["budget_contacts"] = 10**9
    risk_max = cfg["constraints"]["risk_uplift_max"]
    scores = _make_scores(n=2000, with_risk=True)
    scores["risk_ci_high"] = np.where(np.arange(len(scores)) % 2 == 0, risk_max / 2, risk_max * 5)
    assert (scores["risk_ci_high"] > risk_max).any()

    selected = builder.select(scores, cfg)
    assert selected["risk_ci_high"].max() <= risk_max
    assert len(selected) == int((scores["risk_ci_high"] <= risk_max).sum())


def test_export_filename_has_project_prefix(campaign_cfg: dict, tmp_path: Path) -> None:
    """Имя файла выгрузки начинается с имени папки проекта."""
    cfg = copy.deepcopy(campaign_cfg)
    selected = builder.select(_make_scores(n=100), cfg)

    path = export.export_segment(selected, cfg["campaign"]["id"], out_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith(PROJECT_PREFIX)
    assert path.name.endswith(".csv")
    back = pd.read_csv(path)
    assert set(back["client_id"]) == set(selected["client_id"])

    path_pq = export.export_segment(
        selected, cfg["campaign"]["id"], fmt="parquet", out_dir=tmp_path
    )
    assert path_pq.name.startswith(PROJECT_PREFIX)
    assert len(pd.read_parquet(path_pq)) == len(selected)


def test_export_ab_holds_out_control_group(campaign_cfg: dict, tmp_path: Path) -> None:
    """Формат ab: контрольная группа доли из конфига, не пересекается с сегментом."""
    cfg = copy.deepcopy(campaign_cfg)
    cfg["constraints"]["budget_contacts"] = 400
    scores = _make_scores(n=2000)
    selected = builder.select(scores, cfg)
    share = cfg["export"]["control_share"]

    path = export.export_segment_ab(
        selected,
        scores,
        cfg["campaign"]["id"],
        control_share=share,
        seed=cfg["seed"],
        columns=cfg["export"]["columns"],
        out_dir=tmp_path,
    )
    back = pd.read_csv(path)
    present = [c for c in cfg["export"]["columns"] if c in back.columns]
    assert list(back.columns) == [*present, export.GROUP_COLUMN]  # порядок конфига + group

    target = back[back[export.GROUP_COLUMN] == export.GROUP_TARGET]
    control = back[back[export.GROUP_COLUMN] == export.GROUP_CONTROL]
    assert set(target["client_id"]) == set(selected["client_id"])
    assert set(control["client_id"]).isdisjoint(set(target["client_id"]))
    assert len(control) == round(len(selected) * share) > 0
    assert control["uplift_score"].isna().all()  # контроль не про отбор, скоров у него нет

    path_again = export.export_segment_ab(  # тот же seed — та же контрольная группа
        selected,
        scores,
        cfg["campaign"]["id"],
        control_share=share,
        seed=cfg["seed"],
        columns=cfg["export"]["columns"],
        out_dir=tmp_path,
    )
    again = pd.read_csv(path_again)
    assert set(again.loc[again[export.GROUP_COLUMN] == export.GROUP_CONTROL, "client_id"]) == set(
        control["client_id"]
    )


def test_cooldown_removes_recent_contacts() -> None:
    """Контакт в пределах cooldown_days запрещён; давние и новые клиенты проходят."""
    selection = pd.DataFrame({"client_id": [1, 2, 3, 4]})
    history = pd.DataFrame(
        {
            "client_id": [1, 2],
            "sent_at": pd.to_datetime(["2026-11-20", "2026-10-20"]),  # опорная — 2026-11-20
        }
    )
    kept = rules.apply_cooldown(selection, history, cooldown_days=14)
    assert set(kept["client_id"]) == {2, 3, 4}


def test_optimal_share_matches_threshold() -> None:
    """Доля выше порога окупаемости: ровно клиенты со скором не ниже cost/value."""
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"uplift_score": rng.uniform(0.0, 0.01, 10_000)})
    share = builder.optimal_share(scores, cost_per_contact=0.6, value_per_response=120.0)
    expected = float((scores["uplift_score"] >= 0.6 / 120.0).mean())
    assert share == pytest.approx(expected, abs=1e-12)
    assert 0.4 < share < 0.6  # равномерный скор, порог посередине — около половины


def test_score_clients_reads_joblib_model(tmp_path: Path) -> None:
    """score_clients применяет модель из models_store и размечает децилы."""
    joblib.dump(_ToyModel(), tmp_path / "toy.joblib")
    features = pd.DataFrame(
        {"x": np.arange(100.0)},
        index=pd.Index(np.arange(1_000_000, 1_000_100), name="client_id"),
    )
    scored = builder.score_clients("toy", features, models_dir=tmp_path)
    assert list(scored.columns) == ["client_id", "uplift_score", "decile"]
    assert scored["client_id"].tolist() == features.index.tolist()
    assert scored.loc[scored["uplift_score"].idxmax(), "decile"] == 1
    assert scored.loc[scored["uplift_score"].idxmin(), "decile"] == 10


def test_score_clients_missing_model_raises(tmp_path: Path) -> None:
    """Без файла модели — понятная ошибка, а не молчаливый провал."""
    features = pd.DataFrame({"x": [1.0]}, index=pd.Index([1], name="client_id"))
    with pytest.raises(FileNotFoundError, match="toy"):
        builder.score_clients("toy", features, models_dir=tmp_path)
