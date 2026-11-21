"""Фикстуры тестов: искусственные ковариаты, реальная выгрузка не нужна.

Тесты не читают data/raw: они должны проходить на чистой машине и в CI, где
никаких выгрузок нет. Ковариаты генерируются с теми же именами колонок и
похожей формой распределений, но никакого отношения к настоящим данным не имеют.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from uplift_system.datasets.synthetic import make_covariates_for_tests  # noqa: E402

SEED = 42


@pytest.fixture(scope="session")
def sim_cfg() -> dict:
    """Конфиг генератора из configs/, без чтения выгрузки."""
    cfg = yaml.safe_load((PROJECT_DIR / "configs" / "sim_base.yaml").read_text(encoding="utf8"))
    cfg["sample"] = {"n_clients": 4000}
    return cfg


@pytest.fixture(scope="session")
def models_cfg() -> dict:
    cfg = yaml.safe_load((PROJECT_DIR / "configs" / "models.yaml").read_text(encoding="utf8"))
    cfg["base_learner"]["params"]["iterations"] = 60  # тесты должны быть быстрыми
    return cfg


@pytest.fixture(scope="session")
def covariates():
    return make_covariates_for_tests(n=4000, seed=0)


@pytest.fixture(scope="session")
def experiment(sim_cfg, covariates):
    """Один смоделированный эксперимент: DGP, рандомизация, реализация исходов."""
    from uplift_system.datasets.synthetic import generate
    from uplift_system.experiment.simulate import randomize, realize_outcome

    data = generate(sim_cfg, seed=SEED, features=covariates)
    return realize_outcome(randomize(data, control_share=0.5, seed=SEED + 1))


@pytest.fixture
def toy_experiment() -> dict:
    """Восемь клиентов, uplift считается руками — для проверки формул метрик."""
    import numpy as np

    return {
        "y": np.array([1, 0, 1, 0, 1, 1, 0, 0]),
        "t": np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        "score": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
    }
