"""Мета-лернеры: обучаются, ранжируют осмысленно, не оцениваются на обучении."""

import numpy as np
import pytest

from uplift_system.evaluation import ground_truth as G
from uplift_system.experiment.simulate import split_train_holdout
from uplift_system.models.meta import ARCHITECTURES, build

UPLIFT_ARCHS = ("s_learner", "t_learner", "x_learner", "class_transformation")


@pytest.fixture(scope="module")
def split(experiment):
    return split_train_holdout(experiment, holdout_share=0.3, seed=99)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_fit_predict_shape(arch, split, models_cfg):
    """Каждая архитектура обучается и возвращает вектор длины n без пропусков."""
    train, hold = split
    model = build(arch, models_cfg, seed=42, treat_prob=train.meta["treat_prob"])
    model.fit(train.features, train.t, train.y("purchase"))
    score = model.predict_uplift(hold.features)
    assert score.shape == (hold.n,)
    assert np.isfinite(score).all()


@pytest.mark.parametrize("arch", UPLIFT_ARCHS)
def test_beats_random_against_truth(arch, split, models_cfg):
    """Ранговая корреляция с истинным эффектом строго выше, чем у случайного скора."""
    train, hold = split
    ite = hold.ite_true["purchase"].to_numpy()
    model = build(arch, models_cfg, seed=42, treat_prob=train.meta["treat_prob"])
    model.fit(train.features, train.t, train.y("purchase"))
    tau_model = G.rank_correlation(model.predict_uplift(hold.features), ite)
    random_model = build("random", models_cfg, seed=42).fit(
        train.features, train.t, train.y("purchase")
    )
    tau_random = abs(G.rank_correlation(random_model.predict_uplift(hold.features), ite))
    assert tau_model > tau_random


def test_response_model_loses_to_uplift_on_policy_value(split, models_cfg):
    """Отбор по прогнозу отклика приносит меньше истинной ценности, чем uplift-модель.

    Это центральный тезис работы, и он проверяется кодом, а не только ссылкой на
    литературу.
    """
    train, hold = split
    ite = hold.ite_true["purchase"].to_numpy()
    values = {}
    for arch in ("t_learner", "response_model"):
        model = build(arch, models_cfg, seed=42, treat_prob=train.meta["treat_prob"])
        model.fit(train.features, train.t, train.y("purchase"))
        values[arch] = G.policy_value_ratio(ite, model.predict_uplift(hold.features), 0.3)
    assert values["t_learner"] > values["response_model"]


def test_class_transformation_uses_known_propensity(split, models_cfg):
    """При несбалансированном сплите преобразование класса берёт вес из дизайна."""
    train, _ = split
    model = build("class_transformation", models_cfg, seed=42, treat_prob=0.9)
    model.fit(train.features, train.t, train.y("purchase"))
    assert model.e_ == pytest.approx(0.9)


def test_holdout_split_is_disjoint_and_complete(experiment):
    """Разделение не теряет и не дублирует клиентов, содержимое сохраняется."""
    train, hold = split_train_holdout(experiment, holdout_share=0.3, seed=1)
    assert train.n + hold.n == experiment.n
    assert not set(train.id) & set(hold.id)
    assert hold.ite_true is not None and hold.segment is not None
    assert 0.25 < hold.n / experiment.n < 0.35
