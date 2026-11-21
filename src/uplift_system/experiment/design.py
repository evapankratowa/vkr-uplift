"""Дизайн эксперимента: размер выборки, MDE, цена контроля.

Все величины — в долях отклика (0.01 = 1 процентный пункт), не в процентах.
Нормальное приближение для разности долей, двусторонняя альтернатива.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _z_sum(alpha: float, power: float) -> float:
    """Сумма квантилей: z_{1-alpha/2} + z_{power}."""
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def sample_size(base_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Размер ОДНОЙ группы для разницы долей mde при базовой конверсии base_rate.

    n = z_sum^2 * 2 * p * (1 - p) / mde^2, округление вверх. Возвращает число
    клиентов в группе, не суммарно по двум.
    """
    assert 0 < base_rate < 1, "base_rate вне (0, 1)"
    assert mde > 0, "mde должен быть положительным"
    n = _z_sum(alpha, power) ** 2 * 2 * base_rate * (1 - base_rate) / mde**2
    return int(np.ceil(n))


def mde(
    n_treat: int, n_control: int, base_rate: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Минимальный ловимый эффект при заданных размерах групп, абсолютная разность долей.

    d = z_sum * sqrt(p * (1 - p) * (1 / n_t + 1 / n_c)).
    """
    assert n_treat > 0 and n_control > 0, "размеры групп должны быть положительными"
    assert 0 < base_rate < 1, "base_rate вне (0, 1)"
    var = base_rate * (1 - base_rate) * (1 / n_treat + 1 / n_control)
    return float(_z_sum(alpha, power) * np.sqrt(var))


def mde_table(
    n_total: int,
    base_rate: float,
    control_shares: tuple[float, ...] = (0.5, 0.3, 0.1),
    alpha: float = 0.05,
    power: float = 0.8,
) -> pd.DataFrame:
    """Цена контроля: MDE по долям контроля при фиксированном общем размере.

    Колонки: control_share, n_control, n_treat, mde_abs (доли), mde_pp
    (процентные пункты), mde_rel (доля от base_rate).
    """
    rows = []
    for share in control_shares:
        n_c = int(round(n_total * share))
        n_t = n_total - n_c
        d = mde(n_t, n_c, base_rate, alpha, power)
        rows.append(
            {
                "control_share": share,
                "n_control": n_c,
                "n_treat": n_t,
                "mde_abs": d,
                "mde_pp": d * 100,
                "mde_rel": d / base_rate,
            }
        )
    return pd.DataFrame(rows)
