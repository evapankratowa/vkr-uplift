"""Отбор сегмента с максимальным суммарным uplift при ограничении бюджета."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..paths import PROJECT_DIR
from . import rules

MODELS_DIR = PROJECT_DIR / "models_store"  # куда train сериализует модели (joblib)


def _deciles(uplift: pd.Series) -> pd.Series:
    """Децилы скора: 1 — лучший десяток клиентов, 10 — худший.

    Ранги с method="first", чтобы совпавшие скоры не давали дубликаты децилов:
    при прочих равных порядок берётся по исходному, а не случайный.
    """
    ranks = uplift.rank(method="first", ascending=False)
    return np.ceil(ranks * 10.0 / len(uplift)).clip(1, 10).astype(int)


def score_clients(model_name: str, features: pd.DataFrame, models_dir: Any = None) -> pd.DataFrame:
    """Применяет сохранённую модель к клиентам: client_id, uplift_score, decile.

    Модель читается из <models_dir>/<model_name>.joblib (по умолчанию models_store/
    в корне проекта). Индекс features становится client_id: это те же ковариаты,
    на которых обучалась модель, с теми же именами колонок.
    """
    directory = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = directory / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"модель {model_name!r} не найдена: {path}. "
            "Сначала выполните uplift train, файл называется <имя_архитектуры>.joblib"
        )
    model = joblib.load(path)
    scores = pd.DataFrame(
        {
            "client_id": features.index.to_numpy(),
            "uplift_score": np.asarray(model.predict_uplift(features), dtype=float),
        }
    )
    scores["decile"] = _deciles(scores["uplift_score"])
    return scores


def _extract_ids(item: Any) -> pd.Series:
    """client_id из элемента exclude_segments: DataFrame, Series или простой итерируемый."""
    if isinstance(item, pd.DataFrame):
        if "client_id" not in item.columns:
            raise KeyError(f"в сегменте-исключении нет колонки client_id: {list(item.columns)}")
        return item["client_id"]
    return pd.Series(item)


def _expected_value(scores: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series | None:
    """Ожидаемая ценность контакта, руб.: uplift * ценность отклика - стоимость канала.

    Нужны economics.value_per_response_rub и стоимость канала кампании
    (economics.channels[campaign.channel].cost_per_contact_rub); без них колонка
    не добавляется — сборщик остаётся работоспособным на неполном конфиге.
    """
    economics = cfg.get("economics") or {}
    channel = (cfg.get("campaign") or {}).get("channel")
    value = economics.get("value_per_response_rub")
    cost = (economics.get("channels") or {}).get(channel, {}).get("cost_per_contact_rub")
    if value is None or cost is None:
        return None
    return scores["uplift_score"] * float(value) - float(cost)


def select(scores: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Отбирает клиентов под бюджет контактов и отсечку min_uplift.

    Жадный отбор по убыванию uplift эквивалентен оптимуму при одинаковой стоимости
    контакта; при разной стоимости — задача о рюкзаке, см. DECISIONS.md.

    Порядок фильтров — как в жизни сборщика: сначала исключения (пересечения
    с другими кампаниями, constraints.exclude_segments), затем отсечка min_uplift,
    затем риск-ограничение (верхняя граница ДИ uplift по выплатам банка не выше
    constraints.risk_uplift_max; действует, если во входе есть колонка risk_ci_high),
    затем бюджет contacts: берутся лучшие по uplift. Возвращается новый DataFrame
    с campaign_id и expected_value_rub, если их удалось вывести из конфига.
    """
    constraints = cfg.get("constraints") or {}
    selected = scores.copy()
    if "decile" not in selected.columns:
        # децилы считаются по всей оценённой аудитории, до фильтров:
        # исключения ниже сбрасывают индекс, привязка после них дала бы ошибку
        selected["decile"] = _deciles(scores["uplift_score"])

    for item in constraints.get("exclude_segments") or []:
        selected = rules.apply_exclusions(selected, _extract_ids(item))

    min_uplift = constraints.get("min_uplift")
    if min_uplift is not None:
        selected = selected[selected["uplift_score"] >= float(min_uplift)]

    risk_max = constraints.get("risk_uplift_max")
    if risk_max is not None and "risk_ci_high" in selected.columns:
        selected = selected[selected["risk_ci_high"] <= float(risk_max)]

    selected = selected.sort_values("uplift_score", ascending=False, kind="mergesort")
    budget = constraints.get("budget_contacts")
    if budget is not None:
        selected = selected.head(int(budget))

    expected = _expected_value(selected, cfg)
    if expected is not None:
        selected = selected.assign(expected_value_rub=expected)
    campaign_id = (cfg.get("campaign") or {}).get("id")
    if campaign_id is not None:
        selected = selected.assign(campaign_id=campaign_id)
    return selected.reset_index(drop=True)


def optimal_share(
    scores: pd.DataFrame, cost_per_contact: float, value_per_response: float
) -> float:
    """Доля аудитории, на которой предельный uplift равен стоимости контакта.

    Точка отсечения для главы 3 (экономическая эффективность): порог окупаемости
    uplift* = стоимость контакта / ценность отклика, возвращается доля клиентов
    со скором не ниже порога. Доставляемость здесь не участвует: при оплате за
    состоявшийся контакт она сокращается из порога (DECISIONS.md №18), поэтому
    вызывающий передаёт уже согласованную со схемой оплаты стоимость.
    """
    assert value_per_response > 0, "ценность отклика должна быть положительной"
    uplift = np.asarray(scores["uplift_score"], dtype=float)
    if uplift.size == 0:
        return 0.0
    threshold = float(cost_per_contact) / float(value_per_response)
    return float((uplift >= threshold).mean())
