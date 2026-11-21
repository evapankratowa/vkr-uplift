"""Расчёт финансового эффекта кампании по каналам коммуникации (F3, тикет T4).

Формулы — docs/DATA_PASSPORT.md, раздел ЭКОНОМИКА:

    эффект канала = N_инкр_откликов × ценность отклика − N_контактов × цена канала;
    порог uplift = цена / (ценность × доставляемость) при оплате за отправку и
                   цена / ценность при оплате за состоявшийся контакт
                   (доставляемость сокращается и в пороге не участвует).

Входные числа откликов здесь считаются на синтетике с известным истинным
uplift (ITE из DGP, тот же источник истины, что использует
evaluation.policy.compare_strategies): контактируются клиенты с истинным
uplift выше порога окупаемости канала, в пределах бюджета контактов. Когда T2
даст обученные стратегии, модуль переключится на их оценки — интерфейс
сводной таблицы не изменится.
"""

from typing import Any

import numpy as np
import pandas as pd

# Модели оплаты контакта (docs/cost_inputs.yaml): send — платим за отправку
# (недоставленное тоже стоит денег), connected — только за состоявшийся контакт.
PAYMENT_MODELS = ("send", "connected")


def response_value_rub(campaign_cfg: dict[str, Any]) -> float:
    """Ценность отклика в рублях из конфига кампании.

    Читается economics.response_value_rub; поле economics.value_per_response_rub
    поддерживается как алиас (его использует сборщик сегментов).
    """
    econ = campaign_cfg["economics"]
    value = econ.get("response_value_rub", econ.get("value_per_response_rub"))
    assert value is not None, "в configs/campaign.yaml нет economics.response_value_rub"
    return float(value)


def uplift_threshold(
    cost_rub: float, value_rub: float, payment: str, deliverability: float | None
) -> float:
    """Порог uplift, начиная с которого контакт по каналу окупается, в долях отклика.

    Оплата за отправку: цена / (ценность × доставляемость); доставляемость null
    (открытых данных нет, канал sms) трактуется как 1.0 — оптимистичная граница,
    так оговорено в configs/campaign.yaml. Оплата за контакт: цена / ценность.
    Опорные значения — DATA_PASSPORT: push 0,13 п.п., sms 4,79 п.п.,
    call 37,5 п.п., robot 4,58 п.п.
    """
    if payment not in PAYMENT_MODELS:
        raise ValueError(f"неизвестная модель оплаты {payment!r}, есть {PAYMENT_MODELS}")
    if payment == "send":
        deliv = 1.0 if deliverability is None else float(deliverability)
        assert deliv > 0, "доставляемость при оплате за отправку должна быть положительной"
        return float(cost_rub) / (float(value_rub) * deliv)
    return float(cost_rub) / float(value_rub)


def effect_rub(
    n_incremental_responses: float, n_contacts: int, value_rub: float, cost_rub: float
) -> float:
    """Финансовый эффект канала в рублях: инкрементальные отклики × ценность − контакты × цена."""
    return float(n_incremental_responses) * float(value_rub) - int(n_contacts) * float(cost_rub)


def select_contacts(ite_true: np.ndarray, threshold: float, budget_contacts: int) -> np.ndarray:
    """Булева маска контактов: клиенты с истинным uplift выше порога, не более бюджета.

    При превышении бюджета оставляются клиенты с наибольшим uplift — отбор
    по экономическому смыслу канала, а не по порядку строк.
    """
    ite_true = np.asarray(ite_true, dtype=float)
    mask = ite_true > threshold
    if mask.sum() > budget_contacts:
        order = np.argsort(-ite_true, kind="stable")[: int(budget_contacts)]
        mask = np.zeros(len(ite_true), dtype=bool)
        mask[order] = True
    return mask


def bootstrap_money_ci(
    per_client_money: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> dict[str, float]:
    """Бутстрап-интервал суммарного денежного эффекта ресэмплом ПО КЛИЕНТАМ.

    Схема повторяет evaluation.metrics.bootstrap_ci (клиент — единица
    наблюдения, ресэмпл по клиентам, квантили уровня alpha): статистика здесь
    сумма по клиентам в рублях, поэтому внешняя функция не используется
    напрямую — она считает только ранжирующие метрики из METRICS. Пустой вход
    (канал никого не набрал) даёт нулевой эффект и нулевой интервал.
    """
    money = np.asarray(per_client_money, dtype=float)
    if money.size == 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "n_boot_used": 0}
    point = float(money.sum())
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = rng.integers(0, money.size, money.size)
        values[i] = money[idx].sum()
    values = values[np.isfinite(values)]
    return {
        "point": point,
        "ci_low": float(np.quantile(values, alpha / 2)),
        "ci_high": float(np.quantile(values, 1 - alpha / 2)),
        "se": float(values.std(ddof=1)),
        "n_boot_used": int(len(values)),
    }


def channel_scenarios(campaign_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Разворачивает конфиг кампании в строки расчёта: каналы + сценарии цен.

    Каждый канал из economics.channels даёт одну строку (флаг enabled про
    отбор сегмента, на экономику канала не влияет), каждый элемент
    economics.scenarios — ещё одну строку с ценой из сценария и унаследованными
    моделью оплаты и доставляемостью базового канала. Для SMS даёт обязательные
    три строки: по источнику 3,2 ₽, проиндексированная 5,75 ₽, пакетная 30 ₽.
    """
    econ = campaign_cfg["economics"]
    rows = []
    for name, spec in econ.get("channels", {}).items():
        rows.append(
            {
                "channel": name,
                "scenario": "base",
                "cost_rub": float(spec["cost_per_contact_rub"]),
                "payment": spec["payment"],
                "deliverability": spec.get("deliverability"),
            }
        )
    for scenario, spec in econ.get("scenarios", {}).items():
        base = econ["channels"][spec["channel"]]
        rows.append(
            {
                "channel": spec["channel"],
                "scenario": scenario,
                "cost_rub": float(spec["cost_per_contact_rub"]),
                "payment": base["payment"],
                "deliverability": base.get("deliverability"),
            }
        )
    return rows


def build_economics_table(
    ite_true: np.ndarray,
    campaign_cfg: dict[str, Any],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Сводная таблица экономики каналов на синтетике с известным истинным uplift.

    Для каждой строки (канал или сценарий цены): порог отбора, число контактов
    (клиенты с истинным uplift выше порога, в пределах constraints.budget_contacts),
    число инкрементальных откликов (сумма истинного uplift отобранных) и эффект
    в рублях с бутстрап-ДИ по деньгам. Колонки — русские, как в эталонных
    выгрузках output/tables/VKR_uplift_*.csv.
    """
    ite_true = np.asarray(ite_true, dtype=float)
    value = response_value_rub(campaign_cfg)
    budget = int(campaign_cfg.get("constraints", {}).get("budget_contacts", len(ite_true)))
    rows = []
    for spec in channel_scenarios(campaign_cfg):
        threshold = uplift_threshold(
            spec["cost_rub"], value, spec["payment"], spec["deliverability"]
        )
        mask = select_contacts(ite_true, threshold, budget)
        n_contacts = int(mask.sum())
        n_incremental = float(ite_true[mask].sum())
        # деньги по клиенту: ценность инкрементального отклика минус цена контакта;
        # бутстрап ресэмплит именно этот вектор
        per_client = ite_true[mask] * value - spec["cost_rub"]
        ci = bootstrap_money_ci(per_client, n_boot=n_boot, alpha=alpha, seed=seed)
        rows.append(
            {
                "канал": spec["channel"],
                "сценарий": spec["scenario"],
                "цена_контакта_руб": spec["cost_rub"],
                "модель_оплаты": spec["payment"],
                "доставляемость": spec["deliverability"],
                "порог_uplift": threshold,
                "n_контактов": n_contacts,
                "n_инкр_откликов": n_incremental,
                "эффект_руб": ci["point"],
                "эффект_ci_low_руб": ci["ci_low"],
                "эффект_ci_high_руб": ci["ci_high"],
                "эффект_se_руб": ci["se"],
                "n_boot": ci["n_boot_used"],
            }
        )
    return pd.DataFrame(rows)
