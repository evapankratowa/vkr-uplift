"""Карточки проверок `uplift check`: контрольные числа паспорта и целостность пайплайна.

Две группы проверок:
  - контрольные числа паспорта данных (docs/DATA_PASSPORT.md) как ассерты с
    допусками — синтетика (средний ite_true против ate_target конфига, A/A-тест),
    выгрузка кэшбэка (объём, ширина, доли, деньги, коммуникации), открытые RCT
    (X5 RetailHero, Hillstrom);
  - целостность артефактов пайплайна — манифест обучения против текущего
    configs/models.yaml, сериализованные модели, таблицы evaluate/economics,
    выгрузки сегмента.

Проверки, которым не хватает входа, не падают: выгрузки кэшбэка в публичном
репозитории нет (лицензия источника 1), открытые наборы качаются `make data`,
а артефакты появляются после train/evaluate/select-segment — такие карточки
получают жёлтый вердикт «не выполнена» с пояснением «пропущено: нет входа».

Константы-пути (DATA_RAW, OUT_TABLES, MODELS_STORE, CHECKS, CONFIGS) вынесены
на уровень модуля: тесты подменяют их на временные каталоги.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .. import pipeline
from ..config import load_config
from ..datasets.open_data import SOURCES, _read_csv_any
from ..datasets.synthetic import generate, make_covariates_for_tests
from ..experiment import analyze
from ..experiment.simulate import randomize, realize_outcome
from ..paths import CONFIGS, DATA_RAW, OUT_TABLES, PROJECT_PREFIX
from .runner import Check, Outcome

if TYPE_CHECKING:  # только для аннотаций, импорт не тянет pandas туда, где он не нужен
    from ..datasets.base import UpliftDataset

MODELS_STORE = pipeline.MODELS_STORE

# 17 категорий трат выгрузки кэшбэка (DATA_PASSPORT, источник 1)
SPEND_CATEGORIES = (
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
    "Taxi",
    "Transport",
)

# Колонки эталонной таблицы evaluate (pipeline.run_evaluation) — сверка выхода T2.
ARCHITECTURE_COLUMNS = [
    "архитектура",
    "qini обучение",
    "qini отложенная",
    "uplift@30%, п.п.",
    "PEHE",
    "tau против истины",
    "ценность@30% от оракула",
    "сек",
]

_CASHBACK_SHARE_TOL = 0.005  # допуск по долям «после»/«до» (DATA_PASSPORT №2, №3)


class _SyntheticRun:
    """Один общий ленивый прогон синтетики на все проверки генератора.

    Ковариаты — из выгрузки по ключу covariates.source, а при её отсутствии —
    искусственные с теми же именами колонок (тот же fallback, что в командах
    simulate/train/economics). DGP прогоняется один раз и кешируется: карточек
    несколько, генератор — один. Реализованный исход (randomize + realize)
    строится только для проверки A/A.
    """

    def __init__(self, cfg_sim: dict[str, Any]):
        self.cfg = cfg_sim
        self._generated: UpliftDataset | None = None
        self._realized: UpliftDataset | None = None

    def generated(self) -> UpliftDataset:
        """Набор с ite_true без назначения воздействия (контрольный ATE)."""
        if self._generated is None:
            self._generated = generate(self.cfg, seed=self._seed(), features=self._features())
        return self._generated

    def realized(self) -> UpliftDataset:
        """Тот же DGP с назначенным воздействием и наблюдаемыми исходами (A/A)."""
        if self._realized is None:
            # доля теста 0.5 и сдвиг сида — конвенция experiment.run_experiment
            self._realized = realize_outcome(
                randomize(self.generated(), 0.5, self._seed() + 10_000)
            )
        return self._realized

    def _seed(self) -> int:
        return int(self.cfg.get("seed", 42))

    def _features(self) -> pd.DataFrame | None:
        source = Path(self.cfg["covariates"]["source"])
        if source.exists():
            return None  # generate прочитает выгрузку сам по ключу covariates.source
        n_clients = int(self.cfg.get("sample", {}).get("n_clients", 50_000))
        return make_covariates_for_tests(n=n_clients, seed=self._seed())


def _synthetic_cards(
    run: _SyntheticRun, cfg_sim: dict[str, Any], cfg_exp: dict[str, Any]
) -> list[Check]:
    """Контрольные числа синтетики: средний ite_true против ate_target и A/A-тест."""
    ate_target = float(cfg_sim["effect"]["ate_target"])

    def check_ate() -> Outcome:
        data = run.generated()
        got = float(data.ite_true["purchase"].mean())
        rel = abs(got - ate_target) / abs(ate_target)
        note = f"средний ite_true {got:.4f} против ate_target {ate_target:.4f} ({rel:.2%})"
        return Outcome.passed(got, note) if rel < 0.01 else Outcome.failed(got, note)

    design = cfg_exp.get("design") or {}
    validation = cfg_exp.get("validation") or {}
    alpha = float(design.get("alpha", 0.05))
    n_runs = int(validation.get("aa_runs", 1000))
    aa_seed = int(cfg_exp.get("seed", 42))

    def check_aa() -> Outcome:
        data = run.realized()
        result = analyze.aa_test(data, "purchase", n_runs=n_runs, alpha=alpha, seed=aa_seed)
        fpr = float(result["false_positive_rate"])
        tol = 3.0 * (alpha * (1.0 - alpha) / n_runs) ** 0.5
        note = f"доля ложных отклонений {fpr:.3f} против alpha {alpha:.2f} (допуск ±{tol:.3f})"
        return (
            Outcome.passed(result, note)
            if abs(fpr - alpha) <= tol
            else Outcome.failed(result, note)
        )

    return [
        Check(
            id="synthetic.ate",
            what="истинный ATE генератора",
            compare="средний ite_true покупки на свежем прогоне DGP против ate_target конфига",
            expect=f"средний ite_true = {ate_target:.2f}, расхождение < 1%",
            on_fail="СТОП",
            fn=check_ate,
        ),
        Check(
            id="synthetic.aa",
            what="валидность сплита и генератора (A/A)",
            compare="доля ложных отклонений A/A против уровня значимости дизайна",
            expect=f"FPR A/A около {alpha:.2f} (±3σ биномиального шума)",
            on_fail="СТОП",
            fn=check_aa,
        ),
    ]


def _cashback_cards(cfg_sim: dict[str, Any]) -> list[Check]:
    """Контрольные числа выгрузки кэшбэка (DATA_PASSPORT №1–7) как ассерты.

    Выгрузки в публичном репозитории нет (лицензия источника 1): все шесть
    карточек тогда пропускаются жёлтым «пропущено: нет входа». Файл читается
    лениво и один раз; сырые строки в отчёт не попадают — только агрегаты.
    """
    path = Path(cfg_sim["covariates"]["source"])
    cache: dict[str, pd.DataFrame] = {}

    def frame() -> pd.DataFrame:
        if "frame" not in cache:
            cache["frame"] = _read_csv_any(path)
        return cache["frame"]

    def guard(fn: Callable[[pd.DataFrame], Outcome]) -> Callable[[], Outcome]:
        """Обёртка: нет файла — жёлтый пропуск; файл битый — красный вердикт карточки."""

        def inner() -> Outcome:
            if not path.exists():
                return Outcome.skipped(
                    f"пропущено: нет входа ({path.name} кладут в data/raw руками, см. README)"
                )
            try:
                data = frame()
            except Exception as exc:
                return Outcome.failed(
                    path.name, f"выгрузку прочитать не удалось: {type(exc).__name__}: {exc}"
                )
            return fn(data)

        return inner

    def rows_outcome(data: pd.DataFrame) -> Outcome:
        n_rows = len(data)
        unique = bool(data["client_id"].is_unique) if "client_id" in data.columns else False
        value = {"строк": n_rows, "client_id уникален": unique}
        note = f"строк {n_rows} против 50 000; ключ client_id уникален: {'да' if unique else 'нет'}"
        ok = n_rows == 50_000 and unique
        return Outcome.passed(value, note) if ok else Outcome.failed(value, note)

    def shape_outcome(data: pd.DataFrame) -> Outcome:
        n_cols = data.shape[1]
        n_cat = sum(1 for name in SPEND_CATEGORIES if name in data.columns)
        value = {"колонок": n_cols, "категорий трат": n_cat}
        note = f"колонок {n_cols} против 42; категорий трат {n_cat} против 17"
        ok = n_cols == 42 and n_cat == len(SPEND_CATEGORIES)
        return Outcome.passed(value, note) if ok else Outcome.failed(value, note)

    def taxi_outcome(data: pd.DataFrame) -> Outcome:
        present = all(name in data.columns for name in ("Taxi", "Taxi_before"))
        zero = bool(present and (data["Taxi"] == 0).all() and (data["Taxi_before"] == 0).all())
        value = {"Taxi и Taxi_before нули": zero}
        note = (
            "Taxi и Taxi_before — нули по всей выборке"
            if zero
            else "колонки Taxi/Taxi_before отсутствуют или не нулевые"
        )
        return Outcome.passed(value, note) if zero else Outcome.failed(value, note)

    def rates_outcome(data: pd.DataFrame) -> Outcome:
        expected = {
            "cb_merch_last_3_month": 0.4795,
            "cb_bank_last_3_month": 0.3838,
            "cb_merch_before": 0.6737,
            "cb_bank_before": 0.6569,
        }
        absent = [name for name in expected if name not in data.columns]
        if absent:
            return Outcome.failed({"нет колонок": absent}, f"нет колонок: {', '.join(absent)}")
        got = {name: float((data[name] > 0).mean()) for name in expected}
        ok = all(abs(got[name] - want) <= _CASHBACK_SHARE_TOL for name, want in expected.items())
        parts = [f"{name}: {got[name]:.4f} против {want:.4f}" for name, want in expected.items()]
        note = "; ".join(parts)
        return Outcome.passed(got, note) if ok else Outcome.failed(got, note)

    def money_outcome(data: pd.DataFrame) -> Outcome:
        # медианы и средние среди ненулевых: нули и сторно отбрасываются фильтром > 0
        expected = {
            "cb_merch_last_3_month": (206.0, 632.05),
            "cb_bank_last_3_month": (120.0, 360.97),
        }
        absent = [name for name in expected if name not in data.columns]
        if absent:
            return Outcome.failed({"нет колонок": absent}, f"нет колонок: {', '.join(absent)}")
        value: dict[str, float] = {}
        ok = True
        parts = []
        for name, (median_want, mean_want) in expected.items():
            positive = data.loc[data[name] > 0, name]
            median_got = float(positive.median())
            mean_got = float(positive.mean())
            value[f"{name} медиана"] = median_got
            value[f"{name} среднее"] = mean_got
            median_ok = abs(median_got - median_want) <= 1.0
            mean_ok = abs(mean_got - mean_want) <= 0.01 * mean_want
            ok = ok and median_ok and mean_ok
            parts.append(
                f"{name}: медиана {median_got:.0f} против {median_want:.0f}, "
                f"среднее {mean_got:.2f} против {mean_want:.2f}"
            )
        return (
            Outcome.passed(value, "; ".join(parts))
            if ok
            else Outcome.failed(value, "; ".join(parts))
        )

    def communication_outcome(data: pd.DataFrame) -> Outcome:
        if "cnt_communication" not in data.columns:
            return Outcome.failed(
                {"нет колонки": "cnt_communication"}, "нет колонки cnt_communication"
            )
        column = data["cnt_communication"]
        share = float((column > 0).mean())
        median = float(column.median())
        maximum = int(column.max())
        value = {"доля получавших": share, "медиана": median, "максимум": maximum}
        ok = abs(share - 0.6572) <= _CASHBACK_SHARE_TOL and median == 1 and maximum == 18
        note = (
            f"получали коммуникации {share:.4f} против 0.6572; медиана {median:.0f} против 1; "
            f"максимум {maximum} против 18"
        )
        return Outcome.passed(value, note) if ok else Outcome.failed(value, note)

    return [
        Check(
            id="cashback.rows",
            what="объём выгрузки кэшбэка",
            compare="число строк и уникальность client_id против DATA_PASSPORT №1",
            expect="50 000 строк, ключ client_id уникален",
            on_fail="СТОП",
            fn=guard(rows_outcome),
        ),
        Check(
            id="cashback.shape",
            what="ширина выгрузки кэшбэка",
            compare="число колонок и категорий трат против DATA_PASSPORT №1",
            expect="42 колонки, 17 категорий трат",
            on_fail="СТОП",
            fn=guard(shape_outcome),
        ),
        Check(
            id="cashback.taxi",
            what="мёртвые колонки выгрузки",
            compare="Taxi и Taxi_before против DATA_PASSPORT №1a",
            expect="Taxi и Taxi_before — нули по всей выборке",
            on_fail="СТОП",
            fn=guard(taxi_outcome),
        ),
        Check(
            id="cashback.rates",
            what="доли целевого действия и риска",
            compare="доли с кэшбэком > 0 в окнах «после» и «до» против DATA_PASSPORT №2, №3",
            expect="партнёрский: 0.4795 после, 0.6737 до; банковский: 0.3838 после, 0.6569 до",
            on_fail="СТОП",
            fn=guard(rates_outcome),
        ),
        Check(
            id="cashback.money",
            what="суммы кэшбэка среди ненулевых",
            compare="медианы и средние по колонкам кэшбэка против DATA_PASSPORT №4, №5",
            expect=(
                "партнёрский: медиана 206 ₽, среднее 632.05 ₽; "
                "банковский: медиана 120 ₽, среднее 360.97 ₽"
            ),
            on_fail="СТОП",
            fn=guard(money_outcome),
        ),
        Check(
            id="cashback.communication",
            what="коммуникации клиентам",
            compare="доля получавших, медиана и максимум cnt_communication против DATA_PASSPORT №6",
            expect="получали 0.6572; медиана 1; максимум 18",
            on_fail="СТОП",
            fn=guard(communication_outcome),
        ),
    ]


def _x5_card() -> Check:
    """Контрольные числа X5 RetailHero (DATA_PASSPORT №9)."""
    path = DATA_RAW / SOURCES["x5_train"]["filename"]

    def fn() -> Outcome:
        if not path.exists():
            return Outcome.skipped(f"пропущено: нет входа ({path.name} — скачать: make data)")
        raw = _read_csv_any(path)
        raw.columns = [str(column).strip().lower() for column in raw.columns]
        absent = [name for name in ("treatment_flg", "target") if name not in raw.columns]
        if absent:
            return Outcome.failed({"нет колонок": absent}, f"нет колонок: {', '.join(absent)}")
        rows = len(raw)
        share_treat = float(raw["treatment_flg"].mean())
        share_response = float(raw["target"].mean())
        value = {"строк": rows, "доля теста": share_treat, "доля отклика": share_response}
        ok = (
            rows == 200_039
            and abs(share_treat - 0.50) <= 0.01
            and abs(share_response - 0.62) <= 0.01
        )
        note = (
            f"строк {rows} против 200 039; доля теста {share_treat:.2f} против 0.50; "
            f"доля отклика {share_response:.2f} против 0.62"
        )
        return Outcome.passed(value, note) if ok else Outcome.failed(value, note)

    return Check(
        id="x5.control",
        what="контрольные числа X5 RetailHero",
        compare="uplift_train.csv против DATA_PASSPORT №9",
        expect="200 039 строк; доля теста 0.50; доля отклика 0.62",
        on_fail="СТОП",
        fn=fn,
    )


def _hillstrom_card() -> Check:
    """Контрольные числа Hillstrom (DATA_PASSPORT №10)."""
    path = DATA_RAW / SOURCES["hillstrom"]["filename"]

    def fn() -> Outcome:
        if not path.exists():
            return Outcome.skipped(f"пропущено: нет входа ({path.name} — скачать: make data)")
        raw = _read_csv_any(path)
        raw.columns = [str(column).strip().lower() for column in raw.columns]
        absent = [name for name in ("segment", "visit") if name not in raw.columns]
        if absent:
            return Outcome.failed({"нет колонок": absent}, f"нет колонок: {', '.join(absent)}")
        rows = len(raw)
        arms = raw["segment"].astype(str).str.strip().str.lower().value_counts(normalize=True)
        arm_shares = {
            arm: float(arms.get(arm, 0.0)) for arm in ("mens e-mail", "womens e-mail", "no e-mail")
        }
        visit = float(raw["visit"].mean())
        value = {"строк": rows, "доли веток": arm_shares, "visit": visit}
        arms_ok = all(abs(share - 1.0 / 3.0) <= 0.02 for share in arm_shares.values())
        ok = rows == 64_000 and arms_ok and abs(visit - 0.15) <= 0.01
        note = (
            f"строк {rows} против 64 000; доли веток "
            + ", ".join(f"{arm}: {share:.3f}" for arm, share in arm_shares.items())
            + f"; visit {visit:.4f} против 0.15"
        )
        return Outcome.passed(value, note) if ok else Outcome.failed(value, note)

    return Check(
        id="hillstrom.control",
        what="контрольные числа Hillstrom",
        compare="hillstrom_raw.csv против DATA_PASSPORT №10",
        expect="64 000 строк; три ветки примерно по трети; visit около 0.15",
        on_fail="СТОП",
        fn=fn,
    )


def _models_cards() -> list[Check]:
    """Согласованность models_store с текущим configs/models.yaml."""
    models_yaml = CONFIGS / "models.yaml"
    cfg_models = load_config(models_yaml, required=("base_learner", "architectures"))
    names = pipeline.enabled_models(cfg_models)

    def manifest_fn() -> Outcome:
        manifest_path = MODELS_STORE / pipeline.MANIFEST_NAME
        if not manifest_path.exists():
            return Outcome.skipped(
                f"пропущено: нет входа ({MODELS_STORE.name}/manifest.json — сначала uplift train)"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        got = str(manifest.get("config_hash", ""))
        expected = hashlib.sha256(models_yaml.read_bytes()).hexdigest()
        value = {"хеш манифеста": got[:12], "хеш конфига": expected[:12]}
        note = (
            f"хеш конфига в манифесте {got[:12]}… совпадает "
            f"с {expected[:12]}… у configs/models.yaml"
            if got == expected
            else f"хеш конфига в манифесте {got[:12]}… разошёлся с "
            f"{expected[:12]}… у configs/models.yaml: модели обучены под другой конфиг"
        )
        return Outcome.passed(value, note) if got == expected else Outcome.failed(value, note)

    def files_fn() -> Outcome:
        if not (MODELS_STORE / pipeline.MANIFEST_NAME).exists():
            return Outcome.skipped(
                f"пропущено: нет входа ({MODELS_STORE.name}/manifest.json — сначала uplift train)"
            )
        missing = [name for name in names if not (MODELS_STORE / f"{name}.joblib").exists()]
        value = {"архитектур в конфиге": len(names), "нет файлов": missing}
        note = (
            f"все {len(names)} включённых архитектур сериализованы"
            if not missing
            else f"нет файлов моделей: {', '.join(missing)}"
        )
        return Outcome.passed(value, note) if not missing else Outcome.failed(value, note)

    return [
        Check(
            id="models.manifest",
            what="соответствие моделей конфигу",
            compare="config_hash манифеста обучения против sha256 configs/models.yaml",
            expect="хеши совпадают: models_store обучен под текущий конфиг",
            on_fail="СТОП",
            fn=manifest_fn,
        ),
        Check(
            id="models.files",
            what="целостность models_store",
            compare="joblib-файлы против списка включённых архитектур configs/models.yaml",
            expect=f"файл <имя>.joblib есть у каждой из {len(names)} включённых архитектур",
            on_fail="СТОП",
            fn=files_fn,
        ),
    ]


def _artifacts_cards() -> list[Check]:
    """Артефакты evaluate/economics/select-segment: наличие и контрольное содержимое."""

    def econ_fn() -> Outcome:
        candidates = sorted(OUT_TABLES.glob(f"{PROJECT_PREFIX}_economics_*.csv"))
        if not candidates:
            return Outcome.skipped(
                "пропущено: нет входа (таблица economics в output/tables — "
                "сначала uplift economics)"
            )
        table = pd.read_csv(candidates[-1])
        expected = [3.2, 5.75, 30.0]
        if {"канал", "цена_контакта_руб"} <= set(table.columns):
            prices = sorted(
                float(value) for value in table.loc[table["канал"] == "sms", "цена_контакта_руб"]
            )
        else:
            prices = []
        value = {"цены SMS": prices}
        note = f"цены SMS {prices} против трёх сценариев {expected} (файл {candidates[-1].name})"
        return Outcome.passed(value, note) if prices == expected else Outcome.failed(value, note)

    def arch_fn() -> Outcome:
        candidates = sorted(OUT_TABLES.glob(f"{PROJECT_PREFIX}_architectures_*.csv"))
        if not candidates:
            return Outcome.skipped(
                "пропущено: нет входа (таблица архитектур в output/tables — "
                "сначала uplift evaluate)"
            )
        header = pd.read_csv(candidates[-1], nrows=0)  # только заголовок: содержимое не нужно
        got = list(header.columns)
        missing = [column for column in ARCHITECTURE_COLUMNS if column not in got]
        value = {"колонок": len(got), "нет колонок": missing}
        note = (
            f"колонки совпадают с эталоном ({len(got)} шт., файл {candidates[-1].name})"
            if not missing
            else f"нет колонок: {', '.join(missing)}"
        )
        return Outcome.passed(value, note) if not missing else Outcome.failed(value, note)

    def segments_fn() -> Outcome:
        standard_csv = [
            path
            for path in OUT_TABLES.glob(f"{PROJECT_PREFIX}_segment_*.csv")
            if "_segment_ab_" not in path.name
        ]
        ab_csv = list(OUT_TABLES.glob(f"{PROJECT_PREFIX}_segment_ab_*.csv"))
        standard_pq = [
            path
            for path in OUT_TABLES.glob(f"{PROJECT_PREFIX}_segment_*.parquet")
            if "_segment_ab_" not in path.name
        ]
        if not (standard_csv or ab_csv or standard_pq):
            return Outcome.skipped(
                "пропущено: нет входа (выгрузки сегмента в output/tables — "
                "сначала uplift select-segment)"
            )
        ok = bool(standard_csv and ab_csv and standard_pq)
        value = {
            "standard csv": len(standard_csv),
            "ab csv": len(ab_csv),
            "standard parquet": len(standard_pq),
        }
        note = (
            "standard и ab выгрузки на месте (csv/parquet)"
            if ok
            else "неполный набор выгрузок сегмента"
        )
        return Outcome.passed(value, note) if ok else Outcome.failed(value, note)

    return [
        Check(
            id="economics.sms",
            what="сводная таблица экономики",
            compare="цены SMS в таблице economics против трёх сценариев ТЗ (T4)",
            expect="SMS строками по ценам 3.2, 5.75 и 30 ₽",
            on_fail="СТОП",
            fn=econ_fn,
        ),
        Check(
            id="evaluate.columns",
            what="таблица сравнения архитектур",
            compare="заголовок таблицы architectures против эталонных колонок evaluate",
            expect="колонки как у VKR_uplift_architectures_<дата>.csv",
            on_fail="СТОП",
            fn=arch_fn,
        ),
        Check(
            id="segments.exports",
            what="выгрузки сегмента рассылки",
            compare="файлы standard/ab (csv, parquet) в output/tables против контракта выгрузки",
            expect="есть standard и ab выгрузки в csv и parquet",
            on_fail="ИДУ ДАЛЬШЕ С ПОМЕТКОЙ",
            fn=segments_fn,
        ),
    ]


def build_checks(cfg_sim: dict[str, Any], cfg_exp: dict[str, Any]) -> list[Check]:
    """Собирает все карточки проверок: контрольные числа паспорта + целостность пайплайна."""
    run = _SyntheticRun(cfg_sim)
    checks = [*_synthetic_cards(run, cfg_sim, cfg_exp), *_cashback_cards(cfg_sim)]
    checks.append(_x5_card())
    checks.append(_hillstrom_card())
    checks.extend(_models_cards())
    checks.extend(_artifacts_cards())
    return checks
