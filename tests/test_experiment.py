"""Рандомизация и проверки валидности эксперимента."""

import pytest

from uplift_system.experiment import analyze
from uplift_system.experiment.simulate import randomize, realize_outcome


def test_randomization_hits_target_share(experiment):
    """Фактическая доля теста близка к заданной."""
    assert experiment.t.mean() == pytest.approx(0.5, abs=0.02)


def test_covariate_balance_within_threshold(experiment):
    """Баланс ковариат: |SMD| не превышает 0.1 ни по одному признаку."""
    assert analyze.covariate_balance(experiment)["abs_smd"].max() < 0.1


def test_propensity_does_not_depend_on_covariates(experiment):
    """Допущение 1: доля назначений не зависит от децилей признаков.

    При 21 признаке и 10 децилях сравнений около двухсот, поэтому граница на
    максимум |z| взята с запасом на множественность.
    """
    assert analyze.propensity_by_strata(experiment)["abs_z"].max() < 4.5


def test_aa_false_positive_rate_near_alpha(experiment):
    """A/A на контрольной группе даёт долю отклонений около alpha."""
    result = analyze.aa_test(experiment, "purchase", n_runs=400, alpha=0.05, seed=1)
    assert result["false_positive_rate"] == pytest.approx(0.05, abs=0.03)


def test_ate_estimate_unbiased_across_splits(experiment, sim_cfg):
    """Средняя оценка ATE по сплитам сходится к внутривыборочной истине."""
    truth = float(
        (experiment.potential[("purchase", 1)] - experiment.potential[("purchase", 0)]).mean()
    )
    estimates = []
    for i in range(20):
        e = realize_outcome(randomize(experiment, control_share=0.5, seed=500 + i))
        estimates.append(analyze.ate(e, "purchase")["ate"])
    mean = sum(estimates) / len(estimates)
    assert mean == pytest.approx(truth, abs=0.01)


def test_realize_outcome_picks_correct_branch(experiment):
    """Наблюдаемый исход равен потенциальному при назначенном воздействии."""
    for outcome in ("purchase", "activation"):
        observed = experiment.outcomes[outcome].to_numpy()
        y0 = experiment.potential[(outcome, 0)].to_numpy()
        y1 = experiment.potential[(outcome, 1)].to_numpy()
        expected = (experiment.t == 1) * y1 + (experiment.t == 0) * y0
        assert (observed == expected).all()
