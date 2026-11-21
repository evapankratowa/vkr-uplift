"""Генератор данных: если DGP врёт, врёт и всё остальное."""

import numpy as np
import pytest

from uplift_system.datasets.synthetic import generate


def test_base_rates_and_ate_calibrated(sim_cfg, covariates):
    """Базовые уровни и ATE воспроизводят конфиг с точностью 1%."""
    data = generate(sim_cfg, seed=42, features=covariates)
    meta = data.meta
    assert meta["base_rate_purchase"] == pytest.approx(sim_cfg["target"]["base_rate"], rel=0.01)
    assert meta["base_rate_activation"] == pytest.approx(
        sim_cfg["target"]["proxy_base_rate"], rel=0.01
    )
    assert meta["ate_target_realized"] == pytest.approx(sim_cfg["effect"]["ate_target"], rel=0.01)
    assert meta["ate_proxy_realized"] == pytest.approx(sim_cfg["effect"]["ate_proxy"], rel=0.01)


def test_segment_shares_and_effect_signs(sim_cfg, covariates):
    """Доли сегментов из конфига; у спящих собак эффект отрицательный."""
    data = generate(sim_cfg, seed=42, features=covariates)
    shares = data.segment.value_counts(normalize=True)
    for name, expected in sim_cfg["effect"]["segment_shares"].items():
        assert shares[name] == pytest.approx(expected, abs=0.01)
    by_segment = data.ite_true["purchase"].groupby(data.segment).mean()
    assert by_segment["persuadable"] > 0
    assert by_segment["sleeping_dog"] < 0
    assert abs(by_segment["lost_cause"]) < abs(by_segment["persuadable"])


def test_identical_parameters_give_identical_ranking(sim_cfg, covariates):
    """Совпали все параметры двух исходов — совпали и ранжирования по истинным CATE.

    Главная валидация генератора: если эта проверка падает, расхождение метрик
    создаёт сама параметризация, а не задача.
    """
    cfg = {
        **sim_cfg,
        "target": {**sim_cfg["target"], "proxy_base_rate": sim_cfg["target"]["base_rate"]},
        "effect": {
            **sim_cfg["effect"],
            "rho_base": 1.0,
            "proxy_mix": 1.0,
            "ate_proxy": sim_cfg["effect"]["ate_target"],
        },
    }
    data = generate(cfg, seed=42, features=covariates)
    assert data.meta["kendall_true_ite"] > 0.999


def test_proxy_mix_is_monotone(sim_cfg, covariates):
    """Чем выше вес общего механизма, тем выше согласие истинных CATE."""
    taus = []
    for mix in (0.0, 0.4, 0.7, 1.0):
        cfg = {**sim_cfg, "effect": {**sim_cfg["effect"], "proxy_mix": mix}}
        taus.append(generate(cfg, seed=42, features=covariates).meta["kendall_true_ite"])
    assert taus == sorted(taus), f"немонотонно: {taus}"


def test_negative_share_stable_across_mix(sim_cfg, covariates):
    """Доля отрицательных эффектов не зависит от веса смеси.

    Иначе proxy_mix менял бы сразу две вещи — связь и распределение эффекта, —
    и результат нельзя было бы приписать связи.
    """
    shares = []
    for mix in (0.0, 0.5, 1.0):
        cfg = {**sim_cfg, "effect": {**sim_cfg["effect"], "proxy_mix": mix}}
        shares.append(generate(cfg, seed=42, features=covariates).meta["share_negative_activation"])
    assert max(shares) - min(shares) < 0.08, f"доля отрицательных плывёт: {shares}"


def test_analytic_ite_matches_potential_outcomes(sim_cfg, covariates):
    """Аналитический CATE совпадает со средней разностью потенциальных исходов."""
    data = generate(sim_cfg, seed=42, features=covariates)
    for outcome in ("purchase", "activation"):
        realized = float((data.potential[(outcome, 1)] - data.potential[(outcome, 0)]).mean())
        analytic = float(data.ite_true[outcome].mean())
        se = np.sqrt(0.25 / data.n) * 3  # грубая граница на шум реализации
        assert abs(realized - analytic) < max(se, 0.01)


def test_seed_reproducibility(sim_cfg, covariates):
    """Два прогона с одним сидом дают одинаковые данные."""
    a = generate(sim_cfg, seed=7, features=covariates)
    b = generate(sim_cfg, seed=7, features=covariates)
    assert np.array_equal(a.ite_true["purchase"].to_numpy(), b.ite_true["purchase"].to_numpy())
    assert (a.segment == b.segment).all()


def test_unreachable_ate_raises(sim_cfg, covariates):
    """Недостижимый ATE падает с внятным сообщением, а не молча упирается в потолок."""
    cfg = {**sim_cfg, "effect": {**sim_cfg["effect"], "ate_target": 0.9}}
    with pytest.raises(AssertionError, match="недостижим"):
        generate(cfg, seed=42, features=covariates)
