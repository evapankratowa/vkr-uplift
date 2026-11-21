"""Общий интерфейс uplift-модели.

Единый контракт нужен, чтобы добавление архитектуры было строкой в конфиге, а не
переделкой пайплайна оценки и сборщика сегментов.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


class BaseUpliftModel(ABC):
    """Оценивает индивидуальный эффект коммуникации, в долях отклика."""

    name: str = "base"

    def __init__(
        self,
        classifier_factory: Callable[[], Any],
        regressor_factory: Callable[[], Any] | None = None,
        seed: int = 42,
        treat_prob: float | None = None,
    ) -> None:
        self.classifier_factory = classifier_factory
        self.regressor_factory = regressor_factory
        self.seed = seed
        self.treat_prob = treat_prob  # известная из дизайна вероятность назначения

    @abstractmethod
    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "BaseUpliftModel":
        """Обучение на данных эксперимента."""

    @abstractmethod
    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Оценка uplift на клиента; шкала та же, что у исхода (доли отклика)."""

    def predict_interval(self, x: pd.DataFrame, alpha: float = 0.05) -> np.ndarray | None:
        """ДИ для оценки эффекта, если архитектура их даёт; иначе None."""
        return None

    @staticmethod
    def _proba(model: Any, x: pd.DataFrame) -> np.ndarray:
        """Вероятность положительного класса; константный класс отдаётся как есть."""
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return np.full(len(x), float(model.classes_[0]))
        return proba[:, 1]
