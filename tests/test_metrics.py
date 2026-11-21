"""Метрики против счёта руками и против scikit-uplift (проверка №5)."""

import numpy as np
import pytest

from uplift_system.evaluation import ground_truth as G
from uplift_system.evaluation import metrics as M


def test_uplift_at_k_matches_manual(toy_experiment):
    """uplift@50% на игрушечном наборе равен посчитанному руками."""
    y, t, score = toy_experiment["y"], toy_experiment["t"], toy_experiment["score"]
    # топ-4 по скору: клиенты 0..3, все из теста -> контроля нет, метрика не определена
    assert np.isnan(M.uplift_at_k(y, t, score, 0.5))
    # весь набор: тест 2/4, контроль 2/4 -> uplift 0
    assert M.uplift_at_k(y, t, score, 1.0) == pytest.approx(0.0)


def test_random_score_gives_near_zero_qini(experiment):
    """Случайный скор даёт нормированный Qini около нуля."""
    y, t = experiment.y("purchase"), experiment.t
    rng = np.random.default_rng(0)
    values = [M.qini_auc(y, t, rng.normal(0, 1, len(y))) for _ in range(20)]
    assert abs(float(np.mean(values))) < 0.05


def test_true_ite_beats_random(experiment):
    """Истинный эффект как скор ранжирует лучше случайного."""
    y, t = experiment.y("purchase"), experiment.t
    rng = np.random.default_rng(0)
    assert M.qini_auc(y, t, experiment.ite_true["purchase"].to_numpy()) > M.qini_auc(
        y, t, rng.normal(0, 1, len(y))
    )


def test_matches_sklift(experiment):
    """Собственные метрики совпадают с scikit-uplift побитово (проверка №5)."""
    sklift_metrics = pytest.importorskip("sklift.metrics")
    y, t = experiment.y("purchase"), experiment.t
    rng = np.random.default_rng(0)
    scores = {
        "модельный": experiment.ite_true["purchase"].to_numpy() + rng.normal(0, 0.05, len(y)),
        "случайный": rng.normal(0, 1, len(y)),
        "с тиками": np.round(experiment.ite_true["purchase"].to_numpy(), 2),
    }
    for score in scores.values():
        assert M.uplift_at_k(y, t, score, 0.3) == pytest.approx(
            sklift_metrics.uplift_at_k(y, score, t, strategy="overall", k=0.3), abs=1e-12
        )
        assert M.qini_auc(y, t, score) == pytest.approx(
            sklift_metrics.qini_auc_score(y, score, t), abs=1e-12
        )
        assert M.uplift_auc(y, t, score) == pytest.approx(
            sklift_metrics.uplift_auc_score(y, score, t), abs=1e-12
        )


def test_bucket_uplift_covers_everyone(experiment):
    """Побакетный uplift не теряет и не размножает клиентов."""
    y, t = experiment.y("purchase"), experiment.t
    table = M.bucket_uplift(y, t, experiment.ite_true["purchase"].to_numpy(), n_buckets=10)
    assert table["n"].sum() == len(y)
    assert len(table) == 10


def test_bootstrap_ci_contains_point(experiment):
    """Точечная оценка лежит внутри бутстрап-интервала."""
    y, t = experiment.y("purchase"), experiment.t
    ci = M.bootstrap_ci(y, t, experiment.ite_true["purchase"].to_numpy(), "qini_auc", n_boot=200)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_pehe_zero_on_exact_prediction(experiment):
    """PEHE равен нулю, если предсказание совпало с истиной."""
    ite = experiment.ite_true["purchase"].to_numpy()
    assert G.pehe(ite, ite) == pytest.approx(0.0)


def test_policy_value_ratio_of_oracle_is_one(experiment):
    """Отбор по истинному эффекту даёт ровно ценность оракула."""
    ite = experiment.ite_true["purchase"].to_numpy()
    assert G.policy_value_ratio(ite, ite, 0.3) == pytest.approx(1.0)


def test_segment_recovery_picks_persuadables(experiment):
    """Отбор по истинному эффекту переоценивает убеждаемых и обходит спящих собак."""
    table = G.segment_recovery(
        experiment.segment, experiment.ite_true["purchase"].to_numpy(), 0.3
    ).set_index("segment")
    assert table.loc["persuadable", "lift"] > 1.0
    assert table.loc["sleeping_dog", "lift"] < 0.5
