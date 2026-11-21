"""Контракт набора данных для всех остальных модулей.

Одна строка = один клиент. Два исхода: `activation` — промежуточный шаг
(активация оффера, дешёвая и частая прокси-метрика) и `purchase` — целевое
действие (покупка с партнёрским кэшбэком). Истинные эффекты заполнены только у
синтетики; на реальных данных они None, и тогда доступны только ранжирующие
метрики.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTCOMES = ("activation", "purchase")


@dataclass(frozen=True)
class UpliftDataset:
    """Набор данных для uplift-задачи.

    features: ковариаты клиента, одна строка = один клиент (ключ в `id`).
    treatment: 1 — коммуникация отправлена, 0 — контроль. None до рандомизации.
    outcomes: наблюдаемые исходы, ключи из OUTCOMES, значения 0/1.
    ite_true: истинный индивидуальный эффект по каждому исходу, в долях отклика.
    potential: потенциальные исходы {(исход, t): Series}, только у синтетики.
    segment: persuadable / sure_thing / lost_cause / sleeping_dog, только у синтетики.
    """

    features: pd.DataFrame
    id: pd.Series
    treatment: pd.Series | None = None
    outcomes: dict[str, pd.Series] | None = None
    ite_true: dict[str, pd.Series] | None = None
    potential: dict[tuple[str, int], pd.Series] | None = None
    segment: pd.Series | None = None
    name: str = "unnamed"
    meta: dict | None = None

    def __post_init__(self) -> None:
        n = len(self.features)
        assert self.id.is_unique, "ключ клиента не уникален"
        assert len(self.id) == n, "id другой длины, чем features"
        if self.treatment is not None:
            assert len(self.treatment) == n, "treatment другой длины, чем features"
            assert set(np.unique(self.treatment)) <= {0, 1}, "treatment не бинарный"
        if self.outcomes is not None:
            for key, val in self.outcomes.items():
                assert key in OUTCOMES, f"неизвестный исход {key}"
                assert len(val) == n, f"исход {key} другой длины, чем features"
                assert set(np.unique(val)) <= {0, 1}, f"исход {key} не бинарный"

    @property
    def n(self) -> int:
        return len(self.features)

    @property
    def has_ground_truth(self) -> bool:
        return self.ite_true is not None

    def y(self, outcome: str) -> np.ndarray:
        """Наблюдаемый исход как массив."""
        assert self.outcomes is not None, "исходы не реализованы, вызови realize_outcome"
        return self.outcomes[outcome].to_numpy()

    @property
    def t(self) -> np.ndarray:
        """Назначенное воздействие как массив."""
        assert self.treatment is not None, "воздействие не назначено, вызови randomize"
        return self.treatment.to_numpy()

    def replace(self, **changes) -> "UpliftDataset":
        """Копия с заменёнными полями (dataclass frozen)."""
        fields = {
            "features": self.features,
            "id": self.id,
            "treatment": self.treatment,
            "outcomes": self.outcomes,
            "ite_true": self.ite_true,
            "potential": self.potential,
            "segment": self.segment,
            "name": self.name,
            "meta": self.meta,
        }
        fields.update(changes)
        return UpliftDataset(**fields)
