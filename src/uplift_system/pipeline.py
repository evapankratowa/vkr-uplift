"""Пайплайн обучения и оценки: общий код команд `train` и `evaluate`.

Две команды работают над одной и той же выборкой: train сериализует модели
в models_store/<имя>.joblib и манифест, evaluate читает их обратно, считает
метрики (evaluation/metrics.py, ground_truth.py) и пишет таблицу сравнения
архитектур и кривые Qini. Нарезка на обучение/отложенную часть общая и
детерминированная: иначе evaluate мерил бы не те модели.
"""

import hashlib
import json
import logging
import time
from copy import copy
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .datasets.base import OUTCOMES, UpliftDataset
from .evaluation import ground_truth as G
from .evaluation import metrics as M
from .experiment.simulate import run_experiment, split_train_holdout
from .models.meta import build
from .paths import DATA_INTERIM, out_path

logger = logging.getLogger("uplift")

MODELS_STORE = Path(__file__).resolve().parents[2] / "models_store"
MANIFEST_NAME = "manifest.json"

OUTCOME = "purchase"  # целевой исход: покупка с партнёрским кэшбэком
TOP_K = 0.3  # доля аудитории для uplift@k и ценности политики
HOLDOUT_SEED_OFFSET = 777  # нарезка как в ноутбуке 03: seed моделей + 777
# Скор этих baseline не есть оценка CATE: PEHE против истины для них не считается
NO_CATE_ARCHS = ("response_model", "random")


def enabled_models(cfg_models: dict[str, Any]) -> list[str]:
    """Имена включённых в конфиге архитектур и baseline, в порядке конфига."""
    names = [
        item["name"]
        for item in cfg_models.get("architectures", []) + cfg_models.get("baselines", [])
        if item.get("enabled", True)
    ]
    assert names, "в конфиге нет включённых архитектур и baseline"
    return names


def split_for_evaluation(
    dataset: UpliftDataset, cfg_models: dict[str, Any], cfg_exp: dict[str, Any] | None = None
) -> tuple[UpliftDataset, UpliftDataset]:
    """Детерминированная нарезка обучение/отложенная для train и evaluate.

    Доля отложенной части — из конфига валидации (дефолт 0.3), сид — seed
    моделей + 777: та же нарезка, что в ноутбуке 03. train и evaluate обязаны
    получать одинаковые части, иначе метрики считаются по чужим клиентам.
    """
    validation = (cfg_exp or {}).get("validation") or {}
    holdout_share = float(validation.get("holdout_share", 0.3))
    seed = int(cfg_models.get("seed", 42)) + HOLDOUT_SEED_OFFSET
    return split_train_holdout(dataset, holdout_share, seed=seed)


def load_interim_dataset(path: Path) -> UpliftDataset:
    """Восстанавливает набор из parquet ячейки simulate (формат _write_cell в cli).

    Колонки ковариат — всё, что не служебное: treatment, segment, y_<исход>,
    ite_true_<исход>. Истина и сегменты есть только у синтетики: если колонок
    нет, соответствующие поля остаются None и доступны только ранжирующие метрики.
    """
    frame = pd.read_parquet(path)
    if "client_id" in frame.columns:
        frame = frame.set_index("client_id")
    reserved = (
        {"treatment", "segment"}
        | {f"y_{o}" for o in OUTCOMES}
        | {f"ite_true_{o}" for o in OUTCOMES}
    )
    assert f"y_{OUTCOME}" in frame.columns, f"в {path.name} нет колонки y_{OUTCOME}"

    treatment = frame["treatment"].astype(int)
    outcomes = {o: frame[f"y_{o}"].astype(int) for o in OUTCOMES if f"y_{o}" in frame.columns}
    ite_columns = [o for o in OUTCOMES if f"ite_true_{o}" in frame.columns]
    ite_true = (
        {o: frame[f"ite_true_{o}"].astype(float) for o in ite_columns} if ite_columns else None
    )
    segment = frame["segment"] if "segment" in frame.columns else None
    features = frame.drop(columns=[c for c in frame.columns if c in reserved]).astype(float)

    dataset = UpliftDataset(
        features=features,
        id=pd.Series(features.index.to_numpy(), index=features.index),
        treatment=treatment,
        outcomes=outcomes,
        ite_true=ite_true,
        segment=segment,
        name=path.stem,
    )
    logger.info(
        "parquet %s: клиентов %d, признаков %d, истина %s",
        path.name,
        dataset.n,
        features.shape[1],
        "есть" if dataset.has_ground_truth else "нет",
    )
    return dataset


def latest_interim_path(data_dir: Path | None = None) -> Path | None:
    """Самый свежий parquet обучающей выборки из data/interim; None, если пусто."""
    directory = Path(data_dir) if data_dir is not None else DATA_INTERIM
    candidates = sorted(
        directory.glob("experiment_*.parquet"), key=lambda p: (p.stat().st_mtime, p.name)
    )
    return candidates[-1] if candidates else None


def covariates_from_frame(raw: pd.DataFrame, cov_cfg: dict[str, Any]) -> pd.DataFrame:
    """Адаптер сырых ковариат к виду признаков: то же, что synthetic.load_covariates.

    Дублирует трансформацию load_covariates (окно «до» + счётчики вовлечённости,
    log1p денег), но принимает уже загруженный DataFrame — этим пользуется мост
    --source db: db.read_raw собирает те же колонки, что у сырого CSV, и дальше
    данные идут по одному коду с файлом. См. docstring uplift_system.db.read.
    """
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


def _synthetic_covariates(source: str, cfg_sim: dict[str, Any]) -> pd.DataFrame | None:
    """Ковариаты для synthetic-прогона: файл из конфига, БД по --source db, иначе fallback.

    Для файла возвращает None, если выгрузка есть на диске: generate прочитает её
    сам через load_covariates. Выгрузки в публичном репозитории нет (лицензия
    источника 1), поэтому без БД работает fallback на искусственные ковариаты
    с теми же именами колонок — как в команде simulate.
    """
    if source == "db":
        from .db.connection import get_connection
        from .db.read import read_raw

        engine = get_connection()  # строка из DATABASE_URL, дефолт — учебный стенд
        with engine.connect() as conn:
            raw = read_raw(conn)
        logger.info("ковариаты прочитаны из БД: клиентов %d", len(raw))
        return covariates_from_frame(raw, cfg_sim["covariates"])

    source_path = Path(cfg_sim["covariates"]["source"])
    if source_path.exists():
        return None
    n_clients = int(cfg_sim.get("sample", {}).get("n_clients", 50_000))
    seed = int(cfg_sim.get("seed", 42))
    logger.warning(
        "выгрузка %s не найдена: ковариаты заменены синтетическими (n=%d, seed=%d)",
        source_path,
        n_clients,
        seed,
    )
    from .datasets.synthetic import make_covariates_for_tests

    return make_covariates_for_tests(n=n_clients, seed=seed)


def load_training_dataset(
    dataset_name: str | None,
    source: str,
    cfg_sim: dict[str, Any],
    cfg_exp: dict[str, Any],
) -> UpliftDataset:
    """Разрешает обучающую выборку для train/evaluate.

    Без --dataset берётся самый свежий parquet из data/interim (результат
    simulate); если его нет — честный fallback на синтетику с предупреждением.
    --dataset synthetic прогоняет DGP на ковариатах из источника source
    (file — CSV из конфига, db — read_raw из PostgreSQL), x5/hillstrom —
    открытые RCT из datasets.open_data. Одинаковые аргументы у train и
    evaluate обязаны давать один и тот же набор.
    """
    if dataset_name is None:
        path = latest_interim_path()
        if path is not None:
            return load_interim_dataset(path)
        logger.warning("в %s нет parquet: выборка генерируется синтетически", DATA_INTERIM)
        dataset_name = "synthetic"
    if dataset_name == "synthetic":
        features = _synthetic_covariates(source, cfg_sim)
        proxy_grid = cfg_exp.get("grid_proxy_mix") or {}
        control_share = float(
            proxy_grid.get("control_share") or cfg_exp["grid"]["control_share"][0]
        )
        seed = int(cfg_exp.get("seed", cfg_sim.get("seed", 42)))
        logger.info("генерация synthetic: seed %d, доля контроля %.2f", seed, control_share)
        return run_experiment(cfg_sim, control_share=control_share, seed=seed, features=features)
    if dataset_name in ("x5", "hillstrom"):
        from .datasets.open_data import LOADERS

        logger.info("загрузка открытого набора %s", dataset_name)
        return LOADERS[dataset_name]()
    raise ValueError(f"неизвестный датасет {dataset_name}, есть synthetic | x5 | hillstrom")


def _dump_model(model: Any, path: Path) -> None:
    """Сериализует модель без фабрик базового ML.

    Фабрики из models.learners.make_factory — лямбды, стандартный pickle их
    не берёт. Для predict_uplift они не нужны (после fit работают обученные
    подмодели), поэтому сохраняется поверхностная копия без этих атрибутов.
    """
    to_save = copy(model)
    for attr in ("classifier_factory", "regressor_factory"):
        to_save.__dict__.pop(attr, None)
    joblib.dump(to_save, path)


def train_models(
    cfg_models: dict[str, Any],
    train: UpliftDataset,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Обучает включённые архитектуры на обучающей части и сериализует joblib.

    Каждая модель — в <models_dir>/<имя>.joblib (конвенция segments.builder:
    score_clients читает именно такие файлы). Возвращает таблицу с колонками
    «архитектура», «сек» — время обучения, оно же идёт в манифест и итоговую
    таблицу evaluate. Доля теста для архитектур, которым она нужна (X-learner,
    преобразование класса), берётся из метаданных дизайна, при их отсутствии —
    оценивается по обучающей части.
    """
    directory = Path(models_dir) if models_dir is not None else MODELS_STORE
    directory.mkdir(parents=True, exist_ok=True)
    seed = int(cfg_models.get("seed", 42))
    treat_prob = None if train.meta is None else train.meta.get("treat_prob")

    rows = []
    for name in enabled_models(cfg_models):
        model = build(name, cfg_models, seed=seed, treat_prob=treat_prob)
        started = time.perf_counter()
        model.fit(train.features, train.t, train.y(OUTCOME))
        seconds = round(time.perf_counter() - started, 1)
        _dump_model(model, directory / f"{name}.joblib")
        logger.info("обучена %s: %.1f с", name, seconds)
        rows.append({"архитектура": name, "сек": seconds})
    return pd.DataFrame(rows)


def write_manifest(
    cfg_models: dict[str, Any],
    train: UpliftDataset,
    config_path: Path,
    stats: pd.DataFrame,
    models_dir: Path | None = None,
    run_date: str | None = None,
) -> Path:
    """Пишет манифест обучения: модель, параметры, хеш конфига, дата.

    Хеш — sha256 файла конфигурации моделей: по нему check сверяет, что
    сериализованные модели соответствуют текущему configs/models.yaml.
    """
    directory = Path(models_dir) if models_dir is not None else MODELS_STORE
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    params = dict(cfg_models["base_learner"].get("params") or {})
    manifest = {
        "date": run_date or date.today().isoformat(),
        "config": str(config_path),
        "config_hash": config_hash,
        "seed": int(cfg_models.get("seed", 42)),
        "dataset": {"name": train.name, "n_train": int(train.n)},
        "base_learner": cfg_models["base_learner"],
        "models": [
            {
                "name": row["архитектура"],
                "file": f"{row['архитектура']}.joblib",
                "params": params,
                "seconds": float(row["сек"]),
            }
            for row in stats.to_dict("records")
        ],
    }
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("манифест: %s (хеш конфига %s…)", path.name, config_hash[:12])
    return path


def load_trained_models(
    cfg_models: dict[str, Any], models_dir: Path | None = None
) -> dict[str, Any]:
    """Читает сериализованные модели для всех включённых в конфиге имён."""
    directory = Path(models_dir) if models_dir is not None else MODELS_STORE
    models = {}
    for name in enabled_models(cfg_models):
        path = directory / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"модель {name!r} не найдена: {path}. "
                "Сначала выполните uplift train — файл называется <имя_архитектуры>.joblib"
            )
        models[name] = joblib.load(path)
    return models


def _seconds_from_manifest(directory: Path) -> dict[str, float]:
    """Время обучения из манифеста; без манифеста — пусто (колонка «сек» будет NaN)."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: float(item["seconds"]) for item in payload.get("models", [])}


def run_evaluation(
    cfg_models: dict[str, Any],
    dataset: UpliftDataset,
    models_dir: Path | None = None,
    cfg_exp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Считает метрики обученных моделей: таблица по эталонным колонкам + скоры.

    Колонки повторяют VKR_uplift_architectures_<дата>.csv из эталонного прогона:
    Qini на обучении и отложенной, uplift@30% в п.п., PEHE, tau против истины,
    ценность@30% от оракула, секунды обучения из манифеста. Метрики против
    истины — только где она есть (синтетика): на открытых RCT колонки NaN.
    Сортировка — по ценности от оракула, как в сравнении архитектур статьи;
    без истины — по Qini отложенной.
    """
    directory = Path(models_dir) if models_dir is not None else MODELS_STORE
    train, holdout = split_for_evaluation(dataset, cfg_models, cfg_exp)
    models = load_trained_models(cfg_models, directory)

    y_train, t_train = train.y(OUTCOME), train.t
    y_hold, t_hold = holdout.y(OUTCOME), holdout.t
    ite_true = holdout.ite_true[OUTCOME].to_numpy() if holdout.has_ground_truth else None

    rows = []
    scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        score_train = np.asarray(model.predict_uplift(train.features), dtype=float)
        score_hold = np.asarray(model.predict_uplift(holdout.features), dtype=float)
        scores[name] = score_hold
        row = {
            "архитектура": name,
            "qini обучение": M.qini_auc(y_train, t_train, score_train),
            "qini отложенная": M.qini_auc(y_hold, t_hold, score_hold),
            "uplift@30%, п.п.": M.uplift_at_k(y_hold, t_hold, score_hold, TOP_K) * 100,
            "PEHE": np.nan,
            "tau против истины": np.nan,
            "ценность@30% от оракула": np.nan,
        }
        if ite_true is not None:
            row["tau против истины"] = G.rank_correlation(score_hold, ite_true)
            row["ценность@30% от оракула"] = G.policy_value_ratio(ite_true, score_hold, TOP_K)
            if name not in NO_CATE_ARCHS:
                row["PEHE"] = G.pehe(ite_true, score_hold)
        rows.append(row)

    table = pd.DataFrame(rows)
    table["сек"] = table["архитектура"].map(_seconds_from_manifest(directory))
    sort_key = "ценность@30% от оракула" if ite_true is not None else "qini отложенная"
    table = table.sort_values(sort_key, ascending=False, kind="mergesort").reset_index(drop=True)
    return {"table": table, "train": train, "holdout": holdout, "scores": scores}


def plot_qini_curves(holdout: UpliftDataset, scores: dict[str, np.ndarray], path: Path) -> Path:
    """Кривые Qini обученных моделей на отложенной выборке.

    Ось Y — накопленный прирост покупок, штуки; пунктир — случайный отбор.
    Идеальная кривая не рисуется: она выше рабочих в разы и сжимает их в полосу
    у оси (тот же приём, что в ноутбуке 03). Фон белый, PNG в режиме RGB —
    требование пайплайна сборки документа.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y, t = holdout.y(OUTCOME), holdout.t
    for name, score in scores.items():
        x_curve, y_curve = M.qini_curve(y, t, score)
        qini = M.qini_auc(y, t, score)
        ax.plot(x_curve, y_curve, label=f"{name} (Qini={qini:.3f})")
    x_perfect, y_perfect = M.perfect_qini_curve(y, t)
    if holdout.has_ground_truth:
        ite = holdout.ite_true[OUTCOME].to_numpy()
        x_oracle, y_oracle = M.qini_curve(y, t, ite)
        ax.plot(
            x_oracle,
            y_oracle,
            lw=2,
            color="purple",
            label=f"истинный CATE (оракул) (Qini={M.qini_auc(y, t, ite):.3f})",
        )
    ax.plot([0, x_perfect[-1]], [0, y_perfect[-1]], ls="--", color="gray", label="случайный отбор")
    ax.annotate(
        f"идеальная кривая (Qini=1, максимум {int(y_perfect.max())} покупок)\n"
        "не показана: она сжимает остальные",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=7,
        color="dimgray",
    )
    ax.set_xlabel("число клиентов в отборе")
    ax.set_ylabel("накопленный прирост покупок, штуки")
    ax.set_title(f"Кривые Qini на отложенной выборке, n={holdout.n}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    _as_rgb(path)
    return path


def _as_rgb(path: Path) -> None:
    """Пересохраняет PNG в режиме RGB: matplotlib пишет RGBA, документу нужен RGB."""
    from PIL import Image

    with Image.open(path) as img:
        if img.mode != "RGB":
            img.convert("RGB").save(path)


def write_outputs(result: dict[str, Any], run_date: str) -> tuple[Path, Path]:
    """Пишет результаты evaluate: таблица архитектур и график Qini в output/."""
    csv_path = out_path("tables", f"architectures_{run_date}", "csv")
    png_path = out_path("figures", f"qini_curves_{run_date}", "png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(csv_path, index=False)
    plot_qini_curves(result["holdout"], result["scores"], png_path)
    return csv_path, png_path
