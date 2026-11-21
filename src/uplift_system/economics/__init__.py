"""Финансовый эффект кампании по каналам коммуникации (F3, тикет T4)."""

from .core import (
    bootstrap_money_ci,
    build_economics_table,
    channel_scenarios,
    effect_rub,
    response_value_rub,
    select_contacts,
    uplift_threshold,
)

__all__ = [
    "bootstrap_money_ci",
    "build_economics_table",
    "channel_scenarios",
    "effect_rub",
    "response_value_rub",
    "select_contacts",
    "uplift_threshold",
]
