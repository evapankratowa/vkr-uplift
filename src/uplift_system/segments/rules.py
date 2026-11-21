"""Бизнес-ограничения на сегменты: пересечения, частота, исключения."""

import pandas as pd

ID_COLUMN = "client_id"
DATE_COLUMN = "sent_at"  # дата последней коммуникации клиента в истории кампаний


def check_overlap(segments: dict[str, pd.DataFrame], max_per_client: int = 1) -> pd.DataFrame:
    """Клиенты, попавшие больше чем в max_per_client кампаний. Ожидание — пустая таблица.

    segments: имя кампании -> выгрузка отобранных клиентов (колонка client_id).
    Возвращает таблицу с колонками client_id, n_campaigns, campaigns; пустая —
    значит ограничение max_campaigns_per_client соблюдено.
    """
    if not segments:
        return pd.DataFrame(columns=[ID_COLUMN, "n_campaigns", "campaigns"])
    parts = []
    for name, segment in segments.items():
        part = segment[[ID_COLUMN]].copy()
        part["campaign"] = name
        parts.append(part)
    stacked = pd.concat(parts, ignore_index=True)
    counts = stacked.groupby(ID_COLUMN).agg(
        n_campaigns=("campaign", "size"),
        campaigns=("campaign", lambda s: ",".join(sorted(s))),
    )
    too_many = counts[counts["n_campaigns"] > max_per_client].reset_index()
    return too_many.sort_values(ID_COLUMN).reset_index(drop=True)


def apply_cooldown(
    selection: pd.DataFrame, history: pd.DataFrame, cooldown_days: int
) -> pd.DataFrame:
    """Убирает клиентов, получавших коммуникацию за последние cooldown_days.

    history: client_id + sent_at (дата отправки). Опорная дата — максимум sent_at
    в истории: это момент сборки сегмента, от него и меряется «не беспокоить N дней».
    Клиенты, которых в истории нет, проходят без изменений.
    """
    if history.empty:
        return selection
    if DATE_COLUMN not in history.columns:
        raise KeyError(f"в истории нет колонки {DATE_COLUMN!r}: {list(history.columns)}")
    reference = history[DATE_COLUMN].max()
    recent = history.loc[(reference - history[DATE_COLUMN]).dt.days < cooldown_days, ID_COLUMN]
    blocked = set(recent)
    mask = selection[ID_COLUMN].isin(blocked)
    return selection[~mask].reset_index(drop=True)


def apply_exclusions(selection: pd.DataFrame, excluded_ids: pd.Series) -> pd.DataFrame:
    """Снимает клиентов из стоп-листа; печатает счётчик до и после."""
    excluded = set(pd.Series(excluded_ids).dropna())
    mask = selection[ID_COLUMN].isin(excluded)
    print(
        f"стоп-лист: было {len(selection)}, снято {int(mask.sum())}, осталось {int((~mask).sum())}"
    )
    return selection[~mask].reset_index(drop=True)
