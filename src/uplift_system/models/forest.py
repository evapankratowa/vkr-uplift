"""Causal Forest: нелинейные эффекты и доверительные интервалы на клиента.

Этап 5. Обёртка над econml.grf / CausalForestDML.
"""

import numpy as np
import pandas as pd

from .base import BaseUpliftModel


class CausalForest(BaseUpliftModel):
    """Ансамбль причинных деревьев; единственная из архитектур, дающая ДИ «из коробки»."""

    name = "causal_forest"

    def fit(self, x: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "CausalForest":
        raise NotImplementedError

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_interval(self, x: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
        raise NotImplementedError
