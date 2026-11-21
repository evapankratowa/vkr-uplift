"""Базовый ML для всех архитектур.

Один и тот же бустинг на всех мета-лернерах: иначе сравнение архитектур меряет
качество бустинга, а не архитектуру — 10-concepts/Выбор uplift-архитектуры.md, п.6.
"""

from collections.abc import Callable
from typing import Any


def make_factory(cfg: dict[str, Any]) -> tuple[Callable[[], Any], Callable[[], Any]]:
    """Возвращает пару фабрик: (классификатор, регрессор) по конфигу моделей.

    Классификатор нужен для моделей отклика, регрессор — для вменённых эффектов
    в X-learner. Параметры одни и те же, чтобы архитектуры были сопоставимы.
    """
    kind = cfg.get("kind", "catboost")
    params = dict(cfg.get("params") or {})
    if kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return (lambda: CatBoostClassifier(**params), lambda: CatBoostRegressor(**params))
    if kind == "sklearn_gbm":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        allowed = {"n_estimators", "learning_rate", "max_depth", "random_state", "subsample"}
        clean = {key: val for key, val in params.items() if key in allowed}
        return (
            lambda: GradientBoostingClassifier(**clean),
            lambda: GradientBoostingRegressor(**clean),
        )
    raise ValueError(f"неизвестный базовый ML: {kind}")
