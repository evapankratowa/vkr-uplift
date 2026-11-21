"""Дизайн эксперимента: формулы мощности и MDE."""

import pytest

from uplift_system.experiment import design


def test_sample_size_matches_hand_calculation():
    """При p=0.48 и MDE 1 п.п. нужно около 39 200 клиентов в группе."""
    assert design.sample_size(0.48, 0.01) == pytest.approx(39182, rel=0.001)


def test_mde_and_sample_size_are_inverse():
    """Подставив размер из sample_size, получаем обратно тот же MDE."""
    n = design.sample_size(0.48, 0.0125)
    assert design.mde(n, n, 0.48) == pytest.approx(0.0125, rel=0.01)


def test_mde_grows_as_control_shrinks():
    """Чем меньше контроль, тем крупнее минимальный ловимый эффект."""
    table = design.mde_table(50_000, 0.48, (0.5, 0.3, 0.1))
    assert table["mde_abs"].is_monotonic_increasing
    assert table.loc[0, "mde_pp"] == pytest.approx(1.25, abs=0.02)
    assert table.loc[2, "mde_pp"] == pytest.approx(2.09, abs=0.02)


def test_mde_is_maximal_at_half():
    """Дисперсия доли максимальна при p=0.5, значит и MDE там наибольший."""
    values = [design.mde(25_000, 25_000, p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values[2] == max(values)
