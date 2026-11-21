"""Генератор данных с известным истинным эффектом (DGP).

Ковариаты берутся из реальной выгрузки партнёрского кэшбэка, исходы и эффекты
генерируются: только так у каждого клиента есть истинный CATE, против которого
можно честно валидировать модели.

Схема — двумерный пробит. Для клиента с ковариатами x:

    U_act(t) = m_a(x) + d_a(x) * t + e_a
    U_buy(t) = m_y(x) + d_y(x) * t + e_y
    (e_a, e_y) ~ N(0, [[1, rho_err], [rho_err, 1]])

наблюдается act = 1[U_act > 0], buy = 1[U_buy > 0]. Ошибки рисуются один раз на
клиента, поэтому потенциальные исходы при t = 0 и t = 1 согласованы между собой.
Истинный CATE считается аналитически:

    ite(x) = Phi(m(x) + d(x)) - Phi(m(x))

Единицы: все ite и базовые уровни — доли отклика (0.01 = 1 процентный пункт).

Связь двух исходов задаётся ДВУМЯ параметрами, и их нельзя путать:
  rho_err   — корреляция уровней (склонные активировать склонны и покупать);
  rho_eff   — корреляция ЭФФЕКТОВ коммуникации на активацию и на покупку.
`rho_eff` — предмет исследования: при rho_eff = 1 истинные ранжирования клиентов
по двум исходам совпадают полностью, и всё расхождение оценок — это шум
обучения, а не свойство задачи.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .base import UpliftDataset

SEGMENTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")


def _standardize(values: np.ndarray) -> np.ndarray:
    """Z-преобразование; нулевая дисперсия отдаётся нулями."""
    sd = values.std()
    if sd == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / sd


def load_covariates(cfg: dict[str, Any]) -> pd.DataFrame:
    """Читает ковариаты из выгрузки: только окно «до», без утечки из окна исхода.

    Логарифмирует денежные колонки (распределения сильно скошены), выбрасывает
    мёртвые колонки из конфига. Индексом становится client_id.
    """
    cov_cfg = cfg["covariates"]
    raw = pd.read_csv(cov_cfg["source"])
    id_col = cov_cfg["id_column"]
    assert raw[id_col].is_unique, f"{id_col} не уникален в выгрузке"

    drop = set(cov_cfg.get("drop_columns") or [])
    engagement = list(cov_cfg.get("engagement_columns") or [])
    before_cols = [c for c in raw.columns if c.endswith("_before") and c not in drop]
    keep = before_cols + [c for c in engagement if c in raw.columns and c not in drop]

    features = raw[keep].copy()
    features.index = pd.Index(raw[id_col].to_numpy(), name=id_col)
    features = features.astype(float)
    if cov_cfg.get("log1p_transform", True):
        features = np.log1p(features.clip(lower=0))
    assert not features.isna().to_numpy().any(), "в ковариатах появились пропуски"
    return features


SEGMENT_KEYS_TARGET = {
    "sleeping_dog": ("cnt_communication", True),
    "sure_thing": ("cb_merch_before", True),
    "lost_cause": ("cnt_activation", False),
}
SEGMENT_KEYS_PROXY = {
    "sleeping_dog": ("cnt_communication", True),
    "sure_thing": ("cnt_activation", True),
    "lost_cause": ("cnt_view", False),
}


def assign_segments(
    features: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    keys: dict[str, tuple[str, bool]] | None = None,
) -> pd.Series:
    """Приписывает каждому клиенту сегмент отклика по ковариатам, а не броском монеты.

    Логика из предметной области:
      sleeping_dog — клиенты, которых уже больше всех заваливали коммуникациями;
      sure_thing   — те, кто и так активно покупает у партнёров;
      lost_cause   — самые невовлечённые в витрину кэшбэка;
      persuadable  — остаток: средняя вовлечённость, есть куда расти.
    Ранжирование зашумляется, иначе сегмент был бы детерминированной функцией
    одной колонки и любая модель восстанавливала бы его тривиально.
    """
    shares = cfg["effect"]["segment_shares"]
    total = sum(shares[s] for s in SEGMENTS)
    assert abs(total - 1.0) < 1e-6, f"доли сегментов дают {total}, а не 1"

    n = len(features)
    noise_sd = float(cfg["effect"].get("segment_noise_sd", 0.5))

    def key(column: str) -> np.ndarray:
        base = _standardize(features[column].to_numpy()) if column in features else np.zeros(n)
        return base + rng.normal(0, noise_sd, n)

    segment = pd.Series(index=features.index, dtype=object)
    free = np.ones(n, dtype=bool)

    def take(scores: np.ndarray, share: float, label: str, largest: bool) -> None:
        count = int(round(share * n))
        idx = np.flatnonzero(free)
        if count <= 0 or len(idx) == 0:
            return
        count = min(count, len(idx))
        order = np.argsort(-scores[idx] if largest else scores[idx], kind="stable")
        chosen = idx[order[:count]]
        segment.iloc[chosen] = label
        free[chosen] = False

    keys = keys or SEGMENT_KEYS_TARGET
    for label in ("sleeping_dog", "sure_thing", "lost_cause"):
        column, largest = keys[label]
        take(key(column), shares[label], label, largest=largest)
    segment.iloc[np.flatnonzero(free)] = "persuadable"
    assert segment.notna().all(), "часть клиентов осталась без сегмента"
    return segment


def _baseline_score(
    features: pd.DataFrame, rng: np.random.Generator, n_drivers: int = 6
) -> np.ndarray:
    """Линейный латентный скор базового уровня отклика.

    Коэффициенты случайные, но воспроизводимые: смысл DGP не в конкретных весах,
    а в том, что уровень отклика зависит от ковариат.
    """
    weights = rng.normal(0, 1, features.shape[1])
    weights[np.argsort(-np.abs(weights))[n_drivers:]] *= 0.2  # несколько признаков ведущие
    score = features.to_numpy() @ weights
    return _standardize(score)


def _fit_intercept(score: np.ndarray, target_rate: float) -> float:
    """Сдвиг c, при котором mean(Phi(score + c)) равен target_rate."""

    def gap(c: float) -> float:
        return float(stats.norm.cdf(score + c).mean() - target_rate)

    return float(optimize.brentq(gap, -12.0, 12.0, xtol=1e-10))


def _scale_effect(base_score: np.ndarray, shape: np.ndarray, target_ate: float) -> float:
    """Множитель k, при котором средний эффект Phi(m + k*shape) - Phi(m) равен target_ate."""
    m = base_score

    def gap(k: float) -> float:
        ate = float((stats.norm.cdf(m + k * shape) - stats.norm.cdf(m)).mean())
        return ate - target_ate

    lo, hi = 0.0, 1.0
    while gap(hi) < 0 and hi < 1024:
        hi *= 2
    if gap(hi) < 0:
        ceiling = gap(hi) + target_ate
        raise AssertionError(
            f"ATE {target_ate:.4f} недостижим: при данной форме эффекта и базовом уровне "
            f"потолок около {ceiling:.4f}. Снизь ATE или разброс эффекта."
        )
    return float(optimize.brentq(gap, lo, hi, xtol=1e-10))


def _effect_shape(
    segment: pd.Series,
    features: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    drivers_key: str = "drivers",
) -> np.ndarray:
    """Форма эффекта на покупку: знак задаёт сегмент, разброс внутри — драйверы."""
    eff = cfg["effect"]
    multiplier = {
        "persuadable": 1.0,
        "sure_thing": float(eff.get("sure_thing_share_of_effect", 0.05)),
        "lost_cause": float(eff.get("lost_cause_share_of_effect", 0.05)),
        "sleeping_dog": -float(eff.get("sleeping_dog_magnitude", 1.0)),
    }
    base = segment.map(multiplier).to_numpy(dtype=float)

    drivers = [c for c in eff.get(drivers_key, []) if c in features.columns]
    if drivers:
        driver_score = _standardize(features[drivers].to_numpy().sum(axis=1))
    else:
        driver_score = np.zeros(len(features))
    heterogeneity = float(eff.get("heterogeneity", 0.6))
    modulation = np.clip(1.0 + heterogeneity * driver_score, 0.2, 2.5)
    jitter = rng.normal(1.0, 0.1, len(features)).clip(0.5, 1.5)
    return base * modulation * jitter


def _proxy_shape(
    target_shape: np.ndarray,
    features: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Форма эффекта на активацию как смесь двух СТРУКТУРНЫХ форм.

        shape_act = mix * u_target + (1 - mix) * u_proxy,

    где u_target — форма эффекта на покупку, u_proxy — форма, построенная на
    своих сегментах (их задаёт SEGMENT_KEYS_PROXY: «и так активирует всё» —
    по числу активаций, «никогда не заходит в витрину» — по числу просмотров).
    Обе нормированы на единичное среднее, поэтому смесь остаётся в том же
    распределительном семействе: доля клиентов с отрицательным эффектом не
    зависит от mix и равна доле спящих собак.

    `mix` — вес смеси, а НЕ корреляция. Фактическая корреляция форм измеряется и
    кладётся в meta['corr_shapes']: в результатах приводится измеренное значение,
    а не параметр. Предыдущие две попытки (аддитивный сдвиг и гауссова вторая
    компонента) отвергнуты: первая ломала пропорциональность при mix = 1, вторая
    давала 39% отрицательных эффектов против 10% у целевого исхода, то есть
    меняла не только связь, но и само распределение эффекта.
    """
    mix = float(cfg["effect"].get("proxy_mix", cfg["effect"].get("rho_eff", 0.7)))
    assert 0.0 <= mix <= 1.0, "proxy_mix вне [0, 1]"

    mean_target = float(target_shape.mean())
    assert mean_target > 0, "средняя форма эффекта на покупку неположительна — проверь сегменты"
    u_target = target_shape / mean_target

    if mix >= 1.0:
        return u_target

    segment_proxy = assign_segments(features, cfg, rng, keys=SEGMENT_KEYS_PROXY)
    proxy_shape = _effect_shape(segment_proxy, features, cfg, rng, drivers_key="proxy_drivers")
    mean_proxy = float(proxy_shape.mean())
    assert mean_proxy > 0, "средняя форма эффекта на активацию неположительна"
    u_proxy = proxy_shape / mean_proxy

    return mix * u_target + (1.0 - mix) * u_proxy


def generate(
    cfg: dict[str, Any], seed: int | None = None, features: pd.DataFrame | None = None
) -> UpliftDataset:
    """Собирает синтетический набор: ковариаты, сегменты, истинные CATE, ошибки.

    Воздействие НЕ назначается: это забота experiment.simulate, чтобы один и тот
    же DGP можно было прогнать под разными дизайнами эксперимента.
    Возвращает набор с ite_true по обоим исходам и потенциальными исходами.
    """
    seed = int(cfg.get("seed", 42) if seed is None else seed)
    rng = np.random.default_rng(seed)

    if features is None:
        features = load_covariates(cfg)
    n_clients = int(cfg.get("sample", {}).get("n_clients", len(features)))
    if n_clients < len(features):
        keep = rng.choice(len(features), size=n_clients, replace=False)
        features = features.iloc[np.sort(keep)]

    target_cfg = cfg["target"]
    effect_scale = float(cfg["effect"].get("effect_scale", 1.0))

    segment = assign_segments(features, cfg, rng)

    # базовые латентные уровни. rho_base — корреляция УРОВНЕЙ двух исходов:
    # вовлечённые клиенты и активируют офферы, и покупают у партнёров, поэтому
    # независимыми эти уровни делать нельзя. Параметр влияет на достижимое
    # согласие ранжирований не меньше, чем rho_eff, и потому вынесен в конфиг.
    rho_base = float(cfg["effect"].get("rho_base", 0.7))
    assert -1.0 <= rho_base <= 1.0, "rho_base вне [-1, 1]"
    score_buy = _baseline_score(features, np.random.default_rng(seed + 1))
    score_other = _baseline_score(features, np.random.default_rng(seed + 2))
    score_other = score_other - score_buy * float(
        np.dot(score_other, score_buy) / np.dot(score_buy, score_buy)
    )
    score_act = _standardize(
        rho_base * score_buy + np.sqrt(max(0.0, 1.0 - rho_base**2)) * _standardize(score_other)
    )
    m_buy = score_buy + _fit_intercept(score_buy, float(target_cfg["base_rate"]))
    m_act = score_act + _fit_intercept(score_act, float(target_cfg["proxy_base_rate"]))

    # формы эффектов и калибровка на заданный ATE
    shape_buy = _effect_shape(segment, features, cfg, rng)
    shape_act = _proxy_shape(shape_buy, features, cfg, rng)
    k_buy = _scale_effect(m_buy, shape_buy, float(cfg["effect"]["ate_target"]) * effect_scale)
    k_act = _scale_effect(m_act, shape_act, float(cfg["effect"]["ate_proxy"]) * effect_scale)
    d_buy, d_act = k_buy * shape_buy, k_act * shape_act

    # истинные CATE — аналитически
    ite = {
        "purchase": stats.norm.cdf(m_buy + d_buy) - stats.norm.cdf(m_buy),
        "activation": stats.norm.cdf(m_act + d_act) - stats.norm.cdf(m_act),
    }

    # согласованные ошибки: рисуются один раз на клиента
    rho_err = float(cfg["effect"].get("rho_err", 0.5))
    cov = np.array([[1.0, rho_err], [rho_err, 1.0]])
    errors = rng.multivariate_normal([0.0, 0.0], cov, size=len(features))
    e_act, e_buy = errors[:, 0], errors[:, 1]

    potential = {}
    for t in (0, 1):
        potential[("activation", t)] = pd.Series(
            (m_act + d_act * t + e_act > 0).astype(int), index=features.index
        )
        potential[("purchase", t)] = pd.Series(
            (m_buy + d_buy * t + e_buy > 0).astype(int), index=features.index
        )

    return UpliftDataset(
        features=features,
        id=pd.Series(features.index.to_numpy(), index=features.index),
        potential=potential,
        ite_true={key: pd.Series(val, index=features.index) for key, val in ite.items()},
        segment=segment,
        name=f"synthetic_seed{seed}",
        meta={
            "seed": seed,
            "proxy_mix": float(cfg["effect"].get("proxy_mix", cfg["effect"].get("rho_eff", 0.7))),
            "rho_err": rho_err,
            "effect_scale": effect_scale,
            "ate_target_requested": float(cfg["effect"]["ate_target"]) * effect_scale,
            "ate_proxy_requested": float(cfg["effect"]["ate_proxy"]) * effect_scale,
            "ate_target_realized": float(ite["purchase"].mean()),
            "ate_proxy_realized": float(ite["activation"].mean()),
            "base_rate_purchase": float(stats.norm.cdf(m_buy).mean()),
            "base_rate_activation": float(stats.norm.cdf(m_act).mean()),
            "rho_base": rho_base,
            # разложение согласия ранжирований: корреляция ФОРМ эффекта равна
            # rho_eff по построению, а согласие наблюдаемых CATE ниже — разницу
            # даёт разное положение клиента относительно порога отклика
            "corr_shapes": float(np.corrcoef(shape_buy, shape_act)[0, 1]),
            "kendall_true_ite": float(
                stats.kendalltau(ite["activation"], ite["purchase"]).statistic
            ),
            "share_negative_purchase": float((ite["purchase"] < 0).mean()),
            "share_negative_activation": float((ite["activation"] < 0).mean()),
        },
    )


def make_covariates_for_tests(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Искусственные ковариаты с теми же именами колонок — только для тестов.

    Нужны, чтобы прогонять код без реальной выгрузки. Распределения скошенные,
    как у трат, но никакого отношения к настоящим данным не имеют.
    """
    rng = np.random.default_rng(seed)
    categories = [
        "Air",
        "Beauty shop",
        "Book store",
        "Children",
        "Cinema",
        "Clothes",
        "Electronics",
        "Fastfood",
        "Fuel",
        "House",
        "Jewelry",
        "Other",
        "Pet shop",
        "Pharmacies",
        "Supermarkets",
        "Transport",
    ]
    data = {}
    for name in categories:
        spend = rng.lognormal(3.0, 1.5, n) * rng.binomial(1, 0.5, n)
        data[f"{name}_before"] = np.log1p(spend)
    data["cb_merch_before"] = np.log1p(rng.lognormal(5.0, 1.2, n) * rng.binomial(1, 0.67, n))
    data["cb_bank_before"] = np.log1p(rng.lognormal(4.5, 1.2, n) * rng.binomial(1, 0.66, n))
    data["cnt_communication"] = np.log1p(rng.poisson(1.7, n))
    data["cnt_view"] = np.log1p(rng.lognormal(5.0, 1.0, n))
    data["cnt_activation"] = np.log1p(rng.lognormal(4.4, 0.9, n))
    index = pd.Index(np.arange(1_000_000, 1_000_000 + n), name="client_id")
    return pd.DataFrame(data, index=index)
