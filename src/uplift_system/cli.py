"""Командный интерфейс системы.

Пять команд соответствуют функциональным требованиям главы 1 ВКР:
  simulate        смоделировать эксперимент и собрать обучающую выборку
  train           обучить uplift-модели по конфигу
  evaluate        посчитать метрики и сравнить стратегии
  select-segment  собрать сегмент для рассылки и выгрузить его
  check           прогнать проверки и собрать отчёт
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from . import pipeline
from .config import load_config
from .datasets.base import UpliftDataset
from .datasets.synthetic import make_covariates_for_tests
from .experiment import analyze
from .experiment.simulate import grid_cells, run_grid
from .paths import CONFIGS, DATA_INTERIM

logger = logging.getLogger("uplift")


def main(argv: list[str] | None = None) -> int:
    """Точка входа. Возвращает код возврата процесса."""
    parser = argparse.ArgumentParser(prog="uplift", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="смоделировать эксперимент")
    p_sim.add_argument(
        "--config",
        default="configs/sim_base.yaml",
        help="конфиг генератора DGP: ковариаты, исходы, эффект (YAML)",
    )
    p_sim.add_argument(
        "--experiment",
        default="configs/experiment.yaml",
        help="конфиг дизайна: сетка прогонов, сплит, валидация (YAML)",
    )

    p_train = sub.add_parser("train", help="обучить модели")
    p_train.add_argument("--config", default="configs/models.yaml")
    p_train.add_argument("--dataset", default=None, help="synthetic | x5 | hillstrom")
    p_train.add_argument(
        "--source",
        choices=["file", "db"],
        default="file",
        help="источник ковариат для synthetic: файл из конфига или PostgreSQL (DATABASE_URL)",
    )

    p_eval = sub.add_parser("evaluate", help="метрики и сравнение стратегий")
    p_eval.add_argument("--config", default="configs/models.yaml")
    p_eval.add_argument("--dataset", default=None, help="synthetic | x5 | hillstrom")
    p_eval.add_argument(
        "--source",
        choices=["file", "db"],
        default="file",
        help="тот же источник данных, что был у train: evaluate восстанавливает выборку заново",
    )

    p_seg = sub.add_parser("select-segment", help="собрать сегмент для рассылки")
    p_seg.add_argument("--config", default="configs/campaign.yaml")

    p_check = sub.add_parser("check", help="прогнать проверки")
    p_check.add_argument("--all", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    raise NotImplementedError(f"команда {args.command} ещё не реализована")


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Прогон сетки дизайнов: обучающие выборки в data/interim, сводка в stdout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg_sim = load_config(args.config, required=("covariates", "target", "effect", "sample"))
    cfg_exp = load_config(args.experiment, required=("grid",))
    logger.info("конфиги загружены: %s, %s", args.config, args.experiment)

    features = _covariates_for_simulation(cfg_sim)
    cells = grid_cells(cfg_exp)
    logger.info("старт сетки: %d ячеек", len(cells))

    run_date = date.today().isoformat()
    rows = []
    for cell, data in run_grid(cfg_sim, cfg_exp, features=features, cells=cells):
        path = _write_cell(data, cell, run_date)
        stats = analyze.ate(data, "purchase")
        aa = _aa_result(data, cfg_exp)
        logger.info(
            "ячейка %s -> %s: строк %d, ATE %.4f, A/A FPR %.3f",
            _cell_stem(cell, run_date),
            path.name,
            data.n,
            stats["ate"],
            aa["false_positive_rate"] if aa else float("nan"),
        )
        rows.append(
            {
                "n_total": cell["n_total"],
                "control_share": cell["control_share"],
                "effect_scale": cell["effect_scale"],
                "repeat": cell["repeat"],
                "rows": data.n,
                "ate": stats["ate"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "aa_fpr": aa["false_positive_rate"] if aa else float("nan"),
            }
        )

    summary = pd.DataFrame(rows)
    print("\nсводка по ячейкам сетки:")
    print(summary.to_string(index=False))
    logger.info("готово: %d файлов в %s", len(rows), DATA_INTERIM)
    return 0


def _load_configs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Конфиги команды: модели (из --config), генератора и дизайна (из configs/)."""
    cfg_models = load_config(args.config, required=("base_learner", "architectures"))
    cfg_sim = load_config(CONFIGS / "sim_base.yaml", required=("covariates", "effect", "sample"))
    cfg_exp = load_config(CONFIGS / "experiment.yaml", required=("grid",))
    logger.info("конфиги загружены: %s", args.config)
    return cfg_models, cfg_sim, cfg_exp


def _cmd_train(args: argparse.Namespace) -> int:
    """Обучает включённые архитектуры: joblib в models_store/ плюс манифест.

    Выборка — последний parquet из data/interim (или --dataset: synthetic
    прогоняет DGP на ковариатах из --source, x5/hillstrom — открытые RCT).
    Модели сериализуются по конвенции models_store/<имя>.joblib, которую читает
    segments.builder.score_clients; манифест несёт параметры, хеш конфига и дату.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg_models, cfg_sim, cfg_exp = _load_configs(args)

    dataset = pipeline.load_training_dataset(args.dataset, args.source, cfg_sim, cfg_exp)
    train, _ = pipeline.split_for_evaluation(dataset, cfg_models, cfg_exp)
    logger.info("выборка %s: клиентов %d, обучение на %d", dataset.name, dataset.n, train.n)

    stats = pipeline.train_models(cfg_models, train)
    manifest_path = pipeline.write_manifest(cfg_models, train, Path(args.config), stats)
    print("\nобученные модели:")
    print(stats.to_string(index=False))
    logger.info(
        "готово: %d моделей в %s, манифест %s", len(stats), pipeline.MODELS_STORE, manifest_path
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Считает метрики обученных моделей: таблица в output/tables, Qini в output/figures.

    Выборка восстанавливается тем же резолвером, что у train (дефолт — последний
    parquet из data/interim), нарезка на части общая и детерминированная. Таблица
    повторяет колонки эталона VKR_uplift_architectures_<дата>.csv; график —
    кривые Qini всех включённых моделей на отложенной выборке.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg_models, cfg_sim, cfg_exp = _load_configs(args)

    dataset = pipeline.load_training_dataset(args.dataset, args.source, cfg_sim, cfg_exp)
    result = pipeline.run_evaluation(cfg_models, dataset, cfg_exp=cfg_exp)
    run_date = date.today().isoformat()
    csv_path, png_path = pipeline.write_outputs(result, run_date)

    print("\nметрики архитектур (отложенная выборка):")
    print(result["table"].to_string(index=False))
    logger.info("готово: таблица %s, график %s", csv_path, png_path)
    return 0


def _covariates_for_simulation(cfg_sim: dict) -> pd.DataFrame | None:
    """Ковариаты для прогона: из выгрузки, а при её отсутствии — синтетические.

    В публичном репозитории выгрузки кэшбэка нет (лицензионные причины), поэтому
    команда не падает, а честно предупреждает и подставляет искусственные
    ковариаты с теми же именами колонок — воспроизводимость по seed сохраняется.
    """
    source = Path(cfg_sim["covariates"]["source"])
    if source.exists():
        return None  # generate прочитает файл сам по ключу covariates.source
    n_clients = int(cfg_sim.get("sample", {}).get("n_clients", 50_000))
    seed = int(cfg_sim.get("seed", 42))
    logger.warning(
        "выгрузка %s не найдена: ковариаты заменены синтетическими (n=%d, seed=%d)",
        source,
        n_clients,
        seed,
    )
    return make_covariates_for_tests(n=n_clients, seed=seed)


def _write_cell(data: UpliftDataset, cell: dict, run_date: str) -> Path:
    """Собирает DataFrame ячейки (ковариаты, воздействие, исходы, истина) и пишет parquet."""
    frame = data.features.copy()
    frame["treatment"] = data.treatment
    for outcome in ("purchase", "activation"):
        frame[f"y_{outcome}"] = data.outcomes[outcome]
        frame[f"ite_true_{outcome}"] = data.ite_true[outcome]
    frame["segment"] = data.segment
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    path = DATA_INTERIM / f"{_cell_stem(cell, run_date)}.parquet"
    frame.to_parquet(path)
    return path


def _cell_stem(cell: dict, run_date: str) -> str:
    """Имя файла ячейки: параметры сетки + дата прогона, как в data/interim/*.parquet."""
    share = int(round(cell["control_share"] * 100))
    scale = int(round(cell["effect_scale"] * 100))
    n_total = cell["n_total"]
    repeat = cell["repeat"]
    return f"experiment_n{n_total}_control{share}_scale{scale}_rep{repeat}_{run_date}"


def _aa_result(data: UpliftDataset, cfg_exp: dict) -> dict[str, float] | None:
    """Итог A/A-теста по ячейке; None, если A/A выключен в конфиге валидации."""
    validation = cfg_exp.get("validation") or {}
    if not validation.get("aa_test", True):
        return None
    design = cfg_exp.get("design") or {}
    return analyze.aa_test(
        data,
        "purchase",
        n_runs=int(validation.get("aa_runs", 1000)),
        alpha=float(design.get("alpha", 0.05)),
        seed=int(cfg_exp.get("seed", 42)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
