"""Мета-лернеры поверх одного базового ML: S, T, X и преобразование класса.

Обозначения как у Кюнцеля [kunzel-2019]:
    mu_0(x) = E[Y | X = x, T = 0],  mu_1(x) = E[Y | X = x, T = 1],
    tau(x)  = mu_1(x) - mu_0(x).
Все модели возвращают tau в долях отклика.
"""

import numpy as np
import pandas as pd

from .base import BaseUpliftModel


class SLearner(BaseUpliftModel):
    """Одна модель, воздействие подаётся признаком. Быстрый baseline.

    Слабость известна заранее: бустинг может просто не использовать признак
    воздействия, если эффект мал, и тогда tau вырождается в ноль.
    """

    name = "s_learner"

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "SLearner":
        data = x.copy()
        data["__treatment"] = np.asarray(treatment)
        self.model_ = self.classifier_factory()
        self.model_.fit(data, np.asarray(y))
        self.columns_ = list(data.columns)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        treated, control = x.copy(), x.copy()
        treated["__treatment"], control["__treatment"] = 1, 0
        return self._proba(self.model_, treated[self.columns_]) - self._proba(
            self.model_, control[self.columns_]
        )


class TLearner(BaseUpliftModel):
    """Две независимые модели отклика: mu_1 на тесте, mu_0 на контроле.

    При дисбалансе групп модель меньшей группы обучается на малой выборке и
    шумит — это ожидаемая слабость, которую и проверяет сетка дизайнов.
    """

    name = "t_learner"

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "TLearner":
        t, y = np.asarray(treatment), np.asarray(y)
        self.model_treat_ = self.classifier_factory()
        self.model_control_ = self.classifier_factory()
        self.model_treat_.fit(x[t == 1], y[t == 1])
        self.model_control_.fit(x[t == 0], y[t == 0])
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        return self._proba(self.model_treat_, x) - self._proba(self.model_control_, x)


class XLearner(BaseUpliftModel):
    """Двухэтапный лернер Кюнцеля: вменённые эффекты плюс взвешивание по propensity.

    Шаги:
      1. mu_0 на контроле, mu_1 на тесте;
      2. вменение: D_1 = Y - mu_0(X) на тесте, D_0 = mu_1(X) - Y на контроле;
      3. регрессии tau_1 на D_1 и tau_0 на D_0;
      4. tau(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x).
    Вес e(x) берётся из дизайна эксперимента (постоянный по Допущению 1), а не
    оценивается моделью: оценивать известную величину — лишний источник шума.
    """

    name = "x_learner"

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "XLearner":
        assert self.regressor_factory is not None, "X-learner требует фабрику регрессора"
        t, y = np.asarray(treatment), np.asarray(y)
        x_t, x_c = x[t == 1], x[t == 0]
        y_t, y_c = y[t == 1], y[t == 0]

        self.mu1_ = self.classifier_factory().fit(x_t, y_t)
        self.mu0_ = self.classifier_factory().fit(x_c, y_c)

        d_treat = y_t - self._proba(self.mu0_, x_t)
        d_control = self._proba(self.mu1_, x_c) - y_c

        self.tau1_ = self.regressor_factory().fit(x_t, d_treat)
        self.tau0_ = self.regressor_factory().fit(x_c, d_control)
        self.e_ = float(self.treat_prob if self.treat_prob is not None else t.mean())
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        tau0 = self.tau0_.predict(x)
        tau1 = self.tau1_.predict(x)
        return self.e_ * tau0 + (1.0 - self.e_) * tau1


class ClassTransformation(BaseUpliftModel):
    """Преобразование целевой переменной: Z = Y*T + (1 - Y)*(1 - T).

    При сплите 50/50 верно tau(x) = 2 * P(Z = 1 | x) - 1. При другом сплите
    оценка смещена, и требуется взвешивание по propensity — здесь оно сделано
    весами выборки. Смысл включения архитектуры в работу именно в этом
    ограничении: в банке контроль маленький, и наивный вариант ломается.
    """

    name = "class_transformation"

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "ClassTransformation":
        t, y = np.asarray(treatment), np.asarray(y)
        z = y * t + (1 - y) * (1 - t)
        e = float(self.treat_prob if self.treat_prob is not None else t.mean())
        self.e_ = e
        # вес 1/(2e) на тесте и 1/(2(1-e)) на контроле возвращает несмещённость
        weights = np.where(t == 1, 1.0 / (2 * e), 1.0 / (2 * (1 - e)))
        self.model_ = self.classifier_factory()
        self.model_.fit(x, z, sample_weight=weights)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        return 2.0 * self._proba(self.model_, x) - 1.0


class ResponseModel(BaseUpliftModel):
    """Baseline: прогнозная модель отклика, обученная только на тесте.

    Это то, что в банке делают сейчас: отбирают клиентов с высокой вероятностью
    отклика, а не с высоким приростом. Разбор, почему такой отбор проигрывает, —
    15-practice/Ошибка. Прогнозная модель для персонализации.md. Здесь она нужна
    как точка сравнения, а не как рабочая архитектура: predict_uplift возвращает
    НЕ эффект, а вероятность отклика, и это скор для ранжирования.
    """

    name = "response_model"

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "ResponseModel":
        t, y = np.asarray(treatment), np.asarray(y)
        self.model_ = self.classifier_factory()
        self.model_.fit(x[t == 1], y[t == 1])
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        return self._proba(self.model_, x)


class RandomModel(BaseUpliftModel):
    """Baseline: случайное ранжирование. Нижняя граница для любой метрики."""

    name = "random"

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "RandomModel":
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        return self.rng_.normal(0, 1, len(x))


ARCHITECTURES = {
    cls.name: cls
    for cls in (SLearner, TLearner, XLearner, ClassTransformation, ResponseModel, RandomModel)
}


def build(name: str, cfg_models: dict, seed: int = 42, treat_prob: float | None = None):
    """Создаёт архитектуру по имени из configs/models.yaml."""
    from .learners import make_factory

    assert name in ARCHITECTURES, f"неизвестная архитектура {name}, есть {sorted(ARCHITECTURES)}"
    clf, reg = make_factory(cfg_models["base_learner"])
    return ARCHITECTURES[name](
        classifier_factory=clf, regressor_factory=reg, seed=seed, treat_prob=treat_prob
    )
