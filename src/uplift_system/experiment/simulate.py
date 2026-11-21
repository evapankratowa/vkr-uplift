"""Проведение смоделированного эксперимента.

Здесь и только здесь назначается воздействие и выбирается наблюдаемый исход из
потенциальных. Генератор данных об эксперименте ничего не знает, поэтому один и
тот же DGP прогоняется под любым дизайном.

Рандомизация — независимое бернуллиевское назначение с постоянной вероятностью.
Это выполняет Допущение 1 из [kharlamov-2025]: P(T = 1 | X) = p не зависит от
ковариат, и только при нём оценка ценности политики несмещённая.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..datasets.base import OUTCOMES, UpliftDataset
from ..datasets.synthetic import generate


def randomize(dataset: UpliftDataset, control_share: float, seed: int) -> UpliftDataset:
    """Независимое назначение коммуникации с заданной долей контроля."""
    assert 0 < control_share < 1, "доля контроля вне (0, 1)"
    rng = np.random.default_rng(seed)
    treat_prob = 1.0 - control_share
    treatment = pd.Series(
        rng.binomial(1, treat_prob, dataset.n), index=dataset.features.index, name="treatment"
    )
    meta = dict(dataset.meta or {})
    meta.update({"control_share": control_share, "treat_prob": treat_prob, "split_seed": seed})
    return dataset.replace(treatment=treatment, meta=meta)


def realize_outcome(dataset: UpliftDataset) -> UpliftDataset:
    """Выбирает наблюдаемый исход из потенциальных по назначенному воздействию.

    Потенциальные исходы уже согласованы между t = 0 и t = 1 (общие ошибки),
    поэтому здесь нет никакой случайности — только выбор ветки.
    """
    assert dataset.treatment is not None, "воздействие не назначено"
    assert dataset.potential is not None, "потенциальных исходов нет"
    t = dataset.t
    outcomes = {}
    for outcome in OUTCOMES:
        y0 = dataset.potential[(outcome, 0)].to_numpy()
        y1 = dataset.potential[(outcome, 1)].to_numpy()
        observed = np.where(t == 1, y1, y0)
        outcomes[outcome] = pd.Series(observed, index=dataset.features.index, name=outcome)
    return dataset.replace(outcomes=outcomes)


def split_train_holdout(
    dataset: UpliftDataset, holdout_share: float, seed: int
) -> tuple[UpliftDataset, UpliftDataset]:
    """Делит набор на обучающую и отложенную части, сохраняя всё содержимое.

    Метрики uplift на обучающей выборке бессмысленны: модель, которая лучше
    запоминает, получает лучший Qini, и сравнение архитектур превращается в
    сравнение способности к переобучению. Все метрики считаются только здесь,
    на отложенной части.
    """
    assert 0 < holdout_share < 1, "доля отложенной выборки вне (0, 1)"
    rng = np.random.default_rng(seed)
    mask = rng.random(dataset.n) < holdout_share

    def subset(keep: np.ndarray) -> UpliftDataset:
        index = dataset.features.index[keep]
        return dataset.replace(
            features=dataset.features.loc[index],
            id=dataset.id.loc[index],
            treatment=None if dataset.treatment is None else dataset.treatment.loc[index],
            outcomes=None
            if dataset.outcomes is None
            else {key: val.loc[index] for key, val in dataset.outcomes.items()},
            ite_true=None
            if dataset.ite_true is None
            else {key: val.loc[index] for key, val in dataset.ite_true.items()},
            potential=None
            if dataset.potential is None
            else {key: val.loc[index] for key, val in dataset.potential.items()},
            segment=None if dataset.segment is None else dataset.segment.loc[index],
        )

    return subset(~mask), subset(mask)


def run_experiment(
    cfg_sim: dict[str, Any],
    control_share: float,
    seed: int,
    n_clients: int | None = None,
    features: pd.DataFrame | None = None,
    overrides: dict[str, Any] | None = None,
) -> UpliftDataset:
    """Один смоделированный эксперимент: DGP, рандомизация, реализация исходов."""
    cfg = _with_overrides(cfg_sim, overrides)
    if n_clients is not None:
        cfg = _with_overrides(cfg, {"sample": {"n_clients": n_clients}})
    data = generate(cfg, seed=seed, features=features)
    data = randomize(data, control_share=control_share, seed=seed + 10_000)
    return realize_outcome(data)


def _with_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Копия конфига с заменёнными вложенными ключами (на два уровня)."""
    if not overrides:
        return cfg
    new = {key: (dict(val) if isinstance(val, dict) else val) for key, val in cfg.items()}
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(new.get(key), dict):
            new[key].update(val)
        else:
            new[key] = val
    return new


def grid_cells(cfg_exp: dict[str, Any]) -> list[dict[str, Any]]:
    """Разворачивает сетку дизайнов из конфига в список ячеек."""
    grid = cfg_exp["grid"]
    cells = []
    for n_total in grid["n_total"]:
        for control_share in grid["control_share"]:
            for effect_scale in grid["effect_scale"]:
                for repeat in range(int(grid["repeats"])):
                    cells.append(
                        {
                            "n_total": int(n_total),
                            "control_share": float(control_share),
                            "effect_scale": float(effect_scale),
                            "repeat": repeat,
                            "seed": int(cfg_exp.get("seed", 42)) + 1000 * repeat,
                        }
                    )
    return cells


def run_grid(
    cfg_sim: dict[str, Any],
    cfg_exp: dict[str, Any],
    features: pd.DataFrame | None = None,
    cells: list[dict[str, Any]] | None = None,
) -> Iterator[tuple[dict[str, Any], UpliftDataset]]:
    """Ленивый прогон сетки: (описание ячейки, набор данных).

    Генератор, а не список: держать 81 набор по 50 тысяч строк в памяти
    незачем, каждая ячейка обрабатывается и выбрасывается.
    """
    for cell in cells if cells is not None else grid_cells(cfg_exp):
        data = run_experiment(
            cfg_sim,
            control_share=cell["control_share"],
            seed=cell["seed"],
            n_clients=cell["n_total"],
            features=features,
            overrides={"effect": {"effect_scale": cell["effect_scale"]}},
        )
        yield cell, data
