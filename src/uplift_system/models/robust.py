"""DR- и R-learner: устойчивость к смещению отбора, доверительные интервалы.

Этап 5. Обёртки над econml с приведением к BaseUpliftModel; собственная
реализация не нужна и увеличивает поверхность ошибок.
"""

import numpy as np
import pandas as pd

from .base import BaseUpliftModel


class DRLearner(BaseUpliftModel):
    """Doubly robust: состоятелен, если верна модель отклика ИЛИ модель propensity."""

    name = "dr_learner"

    def fit(self, x: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "DRLearner":
        raise NotImplementedError

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RLearner(BaseUpliftModel):
    """Робинсоновская остаточная схема: ортогонализация по отклику и propensity."""

    name = "r_learner"

    def fit(self, x: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "RLearner":
        raise NotImplementedError

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
