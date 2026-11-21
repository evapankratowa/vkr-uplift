"""Чтение сырых ковариат обратно из схемы uplift — мост для флага --source db.

Собирает из нормализованных таблиц DataFrame с теми же колонками, что у сырого
CSV (источник 1 DATA_PASSPORT): client_id, категории трат в двух окнах
(`{категория}_before` и `{категория}`), кэшбэк (`cb_merch_*`, `cb_bank_*`),
счётчики (`cnt_communication`, `cnt_view`, `cnt_activation`). DataFrame
подаётся в тот же адаптер, что и файл (synthetic.load_covariates), поэтому
`train --source db` и `--source file` идут по одному коду.

Счётчики витрины относятся к окну «последние 3 месяца» (DATA_PASSPORT) —
читаются из окна last_3_month. Пропущенные комбинации (клиент, категория, окно)
возвращаются нулями: в сыром файле нули содержательны («не покупал»).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .load import CASHBACK_OUT_COLUMNS, ID_COLUMN, canonical_columns

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Engine

_SPEND_QUERY = """
    select cs.client_id, sc.name as category, ow.code as window_code, cs.amount
    from uplift.client_spend cs
    join uplift.spend_category sc on sc.category_id = cs.category_id
    join uplift.observation_window ow on ow.window_id = cs.window_id
"""

_CASHBACK_QUERY = """
    select cc.client_id, cc.source, ow.code as window_code, cc.amount
    from uplift.client_cashback cc
    join uplift.observation_window ow on ow.window_id = cc.window_id
"""

_COMMUNICATION_QUERY = """
    select cs.client_id, cs.cnt_communication, cs.cnt_view, cs.cnt_activation
    from uplift.communication_stats cs
    join uplift.observation_window ow on ow.window_id = cs.window_id
    where ow.code = 'last_3_month'
"""

_CATEGORIES_QUERY = "select category_id, name from uplift.spend_category order by category_id"


def _read(query: str, conn: Engine | Connection) -> pd.DataFrame:
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    if isinstance(conn, Engine):
        with conn.connect() as c:
            return pd.read_sql(text(query), c)
    return pd.read_sql(text(query), conn)


def read_raw(conn: Engine | Connection) -> pd.DataFrame:
    """Собирает сырой набор ковариат из схемы uplift; строки — клиенты, как в CSV."""
    categories = _read(_CATEGORIES_QUERY, conn)["name"].tolist()
    expected = canonical_columns(categories)

    spend = _read(_SPEND_QUERY, conn)
    spend["amount"] = spend["amount"].astype(float)
    spend["column"] = np.where(
        spend["window_code"] == "before",
        spend["category"] + "_before",
        spend["category"],
    )
    wide_spend = spend.pivot_table(
        index=ID_COLUMN, columns="column", values="amount", aggfunc="sum"
    )

    cashback = _read(_CASHBACK_QUERY, conn)
    cashback["amount"] = cashback["amount"].astype(float)
    cashback["column"] = cashback[["source", "window_code"]].apply(
        lambda row: CASHBACK_OUT_COLUMNS[(row["source"], row["window_code"])], axis=1
    )
    wide_cashback = cashback.pivot_table(
        index=ID_COLUMN, columns="column", values="amount", aggfunc="sum"
    )

    communication = _read(_COMMUNICATION_QUERY, conn).set_index(ID_COLUMN)

    wide = wide_spend.join(wide_cashback, how="outer").join(communication, how="outer")
    # client_id — индекс: сначала value-колонки, потом индекс вернётся reset_index
    value_columns = [c for c in expected if c != ID_COLUMN]
    wide = wide.reindex(columns=value_columns, fill_value=0.0).fillna(0.0)
    return wide.reset_index()[expected]
