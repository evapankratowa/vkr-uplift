"""Приёмка модуля economics (тикет T4): ручной пересчёт, пороги, сценарии SMS."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from uplift_system.economics import core as E

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def campaign_cfg() -> dict:
    """Реальный конфиг кампании: каналы, три сценария SMS, бюджет, ценность отклика."""
    cfg = yaml.safe_load((PROJECT_DIR / "configs" / "campaign.yaml").read_text(encoding="utf8"))
    assert cfg["economics"]["response_value_rub"] == 120
    return cfg


def test_push_row_matches_manual_recalc(campaign_cfg):
    """Строка push (цена 0,125 ₽) сходится с ручным пересчётом по формуле из DATA_PASSPORT."""
    ite = np.array([0.010, -0.005, 0.002, 0.0, 0.020])
    table = E.build_economics_table(ite, campaign_cfg, n_boot=50, seed=1)
    row = table[(table["канал"] == "push") & (table["сценарий"] == "base")].iloc[0]

    threshold = 0.125 / (120 * 0.80)  # оплата за отправку, доставляемость 0.80
    expected_mask = ite > threshold  # [True, False, True, False, True]
    n_contacts = int(expected_mask.sum())
    n_incremental = float(ite[expected_mask].sum())
    assert row["порог_uplift"] == pytest.approx(threshold)
    assert row["n_контактов"] == n_contacts == 3
    assert row["n_инкр_откликов"] == pytest.approx(n_incremental)
    assert row["эффект_руб"] == pytest.approx(n_incremental * 120 - n_contacts * 0.125)
    assert row["эффект_ci_low_руб"] <= row["эффект_руб"] <= row["эффект_ci_high_руб"]


def test_threshold_formulas_match_data_passport():
    """Порог = цена/(ценность × доставляемость) за отправку, цена/ценность за контакт.

    Опорные значения в п.п. — DATA_PASSPORT, ЭКОНОМИКА.
    """
    # push: оплата за отправку
    assert E.uplift_threshold(0.125, 120, "send", 0.80) == pytest.approx(0.125 / (120 * 0.80))
    # sms: доставляемость неизвестна (None) — оптимистичная граница с 1.0
    assert E.uplift_threshold(5.75, 120, "send", None) == pytest.approx(5.75 / 120)
    # call и robot: оплата за состоявшийся контакт, доставляемость не участвует
    assert E.uplift_threshold(45.0, 120, "connected", 0.50) == pytest.approx(45.0 / 120)
    assert E.uplift_threshold(5.5, 120, "connected", 0.50) == pytest.approx(5.5 / 120)
    # контрольные числа DATA_PASSPORT в п.п.
    assert E.uplift_threshold(0.125, 120, "send", 0.80) * 100 == pytest.approx(0.13, abs=0.01)
    assert E.uplift_threshold(5.75, 120, "send", None) * 100 == pytest.approx(4.79, abs=0.01)
    assert E.uplift_threshold(45.0, 120, "connected", 0.50) * 100 == pytest.approx(37.5, abs=0.01)
    assert E.uplift_threshold(5.5, 120, "connected", 0.50) * 100 == pytest.approx(4.58, abs=0.01)
    with pytest.raises(ValueError, match="модель оплаты"):
        E.uplift_threshold(1.0, 120, "per_view", None)


def test_sms_three_price_scenarios_as_rows(campaign_cfg):
    """Все три сценария цены SMS отдельными строками: 3,2 / 5,75 / 30 ₽."""
    rng = np.random.default_rng(0)
    ite = rng.normal(0.04, 0.05, 3000)  # часть клиентов выше любого из порогов SMS
    table = E.build_economics_table(ite, campaign_cfg, n_boot=50, seed=1)

    sms = table[table["канал"] == "sms"].sort_values("цена_контакта_руб")
    assert len(sms) == 3
    assert sms["цена_контакта_руб"].tolist() == [3.2, 5.75, 30.0]
    assert sms["сценарий"].tolist() == ["sms_source_2024", "base", "sms_packet_megafon"]
    # пороги — формула «за отправку» без доставляемости: цена/120
    assert sms["порог_uplift"].tolist() == pytest.approx([3.2 / 120, 5.75 / 120, 30.0 / 120])
    # дороже канал — не больше контактов: отбор одной и той же аудитории по порогу
    contacts = sms["n_контактов"].tolist()
    assert contacts[0] >= contacts[1] >= contacts[2]


def test_bootstrap_money_ci_covers_point_and_empty():
    """Денежный бутстрап: точка внутри ДИ; пустой отбор канала — нулевой эффект."""
    rng = np.random.default_rng(7)
    money = rng.normal(100.0, 50.0, 500)
    ci = E.bootstrap_money_ci(money, n_boot=200, seed=3)
    assert ci["point"] == pytest.approx(money.sum())
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n_boot_used"] == 200

    empty = E.bootstrap_money_ci(np.array([]), n_boot=200, seed=3)
    assert empty == {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "n_boot_used": 0}


def test_select_contacts_respects_budget():
    """Бюджет контактов обрезает отбор, оставляя клиентов с наибольшим uplift."""
    ite = np.linspace(0.0, 0.05, 100)
    mask = E.select_contacts(ite, threshold=0.01, budget_contacts=10)
    assert mask.sum() == 10
    assert ite[mask].min() >= 0.01
    assert set(np.flatnonzero(mask)) == set(np.argsort(-ite)[:10])
