"""Командный интерфейс системы.

Шесть команд соответствуют функциональным требованиям главы 1 ВКР:
  simulate        смоделировать эксперимент и собрать обучающую выборку
  train           обучить uplift-модели по конфигу
  evaluate        посчитать метрики и сравнить стратегии
  economics       оценить финансовый эффект кампании по каналам
  select-segment  собрать сегмент для рассылки и выгрузить его
  check           прогнать проверки и собрать отчёт
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from . import pipeline
from .checks import cards
from .checks.runner import CheckRunner
from .config import load_config
from .datasets.base import UpliftDataset
from .datasets.synthetic import load_covariates, make_covariates_for_tests
from .economics.run import main as economics_main
from .experiment import analyze
from .experiment.simulate import grid_cells, run_grid
from .paths import CHECKS, CONFIGS, DATA_INTERIM
from .segments import builder, export

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
    p_seg.add_argument(
        "--sim-config",
        default="configs/sim_base.yaml",
        help="конфиг ковариат аудитории: выгрузка + синтетический fallback (YAML)",
    )

    p_econ = sub.add_parser("economics", help="финансовый эффект кампании по каналам")
    p_econ.add_argument(
        "--config",
        default="configs/campaign.yaml",
        help="конфиг кампании: ценность отклика, каналы, сценарии SMS, бюджет (YAML)",
    )
    p_econ.add_argument(
        "--sim-config",
        default="configs/sim_base.yaml",
        help="конфиг генератора DGP для автономного синтетического прогона (YAML)",
    )
    p_econ.add_argument("--n-boot", type=int, default=1000, help="число бутстрап-итераций")
    p_econ.add_argument(
        "--out",
        default=None,
        help="куда писать CSV (по умолчанию output/tables/<префикс>_economics_<дата>.csv)",
    )

    p_check = sub.add_parser("check", help="прогнать проверки и собрать отчёт")
    p_check.add_argument(
        "--all",
        action="store_true",
        help="прогнать все карточки проверок (сейчас единственный режим)",
    )
    p_check.add_argument(
        "--config",
        default="configs/sim_base.yaml",
        help="конфиг генератора DGP: контрольные числа синтетики (YAML)",
    )
    p_check.add_argument(
        "--experiment",
        default="configs/experiment.yaml",
        help="конфиг дизайна: уровень значимости и прогоны A/A (YAML)",
    )
    p_check.add_argument(
        "--out-dir",
        default=None,
        help="куда писать отчёт (по умолчанию checks/)",
    )

    args = parser.parse_args(argv)
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "economics":
        return _cmd_economics(args)
    if args.command == "select-segment":
        return _cmd_select_segment(args)
    if args.command == "check":
        return _cmd_check(args)
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


def _cmd_economics(args: argparse.Namespace) -> int:
    """Финансовый эффект кампании по каналам: тонкая обёртка над economics.run.

    Вся логика — в economics.run.main: автономный синтетический прогон с seed
    из конфигов, сводная таблица по каналам (три сценария цены SMS строками)
    в output/tables и stdout, бутстрап-ДИ по деньгам. Обёртка только пробрасывает
    аргументы командной строки, чтобы команда жила в общем интерфейсе `uplift`.
    """
    forwarded = [
        "--config",
        args.config,
        "--sim-config",
        args.sim_config,
        "--n-boot",
        str(args.n_boot),
    ]
    if args.out is not None:
        forwarded += ["--out", args.out]
    return economics_main(forwarded)


def _cmd_check(args: argparse.Namespace) -> int:
    """Прогоняет все карточки проверок и собирает отчёт в checks/.

    Контрольные числа — DATA_PASSPORT.md как ассерты с допусками (синтетика,
    выгрузка кэшбэка, X5, Hillstrom), плюс целостность артефактов пайплайна
    (манифест моделей против configs/models.yaml, таблицы evaluate/economics,
    выгрузки сегмента). Отчёт пишется дважды: checks/checks.json
    (machine-readable, фиксированное имя) и checks/checks_<дата>.md (для
    человека). Код возврата 1, если есть хоть одна красная проверка; проверки
    без входа (сырые данные, сеть, непрогнанные команды) жёлтые и не роняют
    команду.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg_sim = load_config(args.config, required=("covariates", "effect", "sample"))
    cfg_exp = load_config(args.experiment, required=("grid", "design", "validation"))
    logger.info("конфиги загружены: %s, %s", args.config, args.experiment)

    runner = CheckRunner(cards.build_checks(cfg_sim, cfg_exp))
    runner.run()

    out_dir = Path(args.out_dir) if args.out_dir else CHECKS
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = runner.to_json(out_dir / "checks.json")
    markdown_path = out_dir / f"checks_{date.today().isoformat()}.md"
    markdown_path.write_text(runner.to_markdown(), encoding="utf-8")

    print(runner.to_markdown())
    counts = runner.counts()
    if runner.exit_code:
        logger.error(
            "красных проверок: %d — отчёт: %s, %s",
            counts["red"],
            json_path,
            markdown_path,
        )
    else:
        logger.info(
            "готово: зелёных %d, жёлтых %d — отчёт: %s, %s",
            counts["green"],
            counts["yellow"],
            json_path,
            markdown_path,
        )
    return runner.exit_code


def _cmd_select_segment(args: argparse.Namespace) -> int:
    """Собирает сегмент рассылки: скоринг, отбор под бюджет и правила, выгрузка.

    Аудитория — ковариаты из --sim-config: выгрузка по ключу covariates.source,
    а при её отсутствии (в публичном репозитории кэшбэк-файла нет) —
    синтетический fallback с теми же колонками, воспроизводимый по seed.
    Скоринг — segments.builder.score_clients моделью campaign.model из
    models_store/ (конвенция имён train), отбор — segments.builder.select:
    исключения, отсечка min_uplift, риск-граница и бюджет contacts — размер
    сегмента бюджет не превышает. Выгрузка — segments.export в форматах standard
    и ab, каждый в csv и parquet (контракт — docs/EXTERNAL_INTEGRATIONS.md,
    раздел 3); контрольная группа формата ab не пересекается с сегментом.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, required=("campaign", "constraints", "export"))
    cfg_sim = load_config(args.sim_config, required=("covariates",))
    logger.info("конфиги загружены: %s, %s", args.config, args.sim_config)

    campaign = cfg["campaign"]
    features = _audience_features(cfg_sim)
    scores = builder.score_clients(campaign["model"], features)
    logger.info("скоринг %s: оценено клиентов %d", campaign["model"], len(scores))

    selected = builder.select(scores, cfg)
    budget = (cfg.get("constraints") or {}).get("budget_contacts")
    logger.info("отбор: в сегменте %d из %d (бюджет %s)", len(selected), len(scores), budget)

    paths = _export_segment_files(
        selected, scores, campaign, cfg.get("export") or {}, int(cfg.get("seed", 42))
    )

    print(f"\nсегмент кампании {campaign.get('id')}: клиентов {len(selected)}, бюджет {budget}")
    print(f"  средний uplift сегмента: {selected['uplift_score'].mean():.5f}")
    if "expected_value_rub" in selected.columns:
        total_value = selected["expected_value_rub"].sum()
        print(f"  суммарная ожидаемая ценность контактов: {total_value:.0f} ₽")
    print("файлы выгрузки:")
    for path in paths:
        print(f"  {path}")
    logger.info("готово: %d файлов выгрузки", len(paths))
    return 0


def _audience_features(cfg_sim: dict) -> pd.DataFrame:
    """Ковариаты аудитории для скоринга: выгрузка, а при её отсутствии — синтетика.

    Та же логика fallback, что в economics.run: публичный репозиторий не содержит
    кэшбэк-выгрузки (лицензионные причины), поэтому подставляются искусственные
    ковариаты с теми же именами колонок — скоринг остаётся воспроизводимым по seed.
    """
    source = Path(cfg_sim["covariates"]["source"])
    if source.exists():
        return load_covariates(cfg_sim)
    n_clients = int(cfg_sim.get("sample", {}).get("n_clients", 50_000))
    seed = int(cfg_sim.get("seed", 42))
    logger.warning(
        "выгрузка %s не найдена: аудитория заменена синтетической (n=%d, seed=%d)",
        source,
        n_clients,
        seed,
    )
    return make_covariates_for_tests(n=n_clients, seed=seed)


def _export_segment_files(
    selected: pd.DataFrame,
    scores: pd.DataFrame,
    campaign: dict,
    export_cfg: dict,
    seed: int,
) -> list[Path]:
    """Пишет выгрузки сегмента: форматы standard и ab, каждый в csv и parquet.

    Имена и состав колонок — конвенция segments.export: <префикс>_segment_<кампания>
    и <префикс>_segment_ab_<кампания>; колонки — export.columns конфига, у формата
    ab в конец добавляется group (target | control). Контрольная группа доли
    export.control_share выбирается из оценённой аудитории вне сегмента, поэтому
    пересечения с целевой выгрузкой нет по построению.
    """
    campaign_id = campaign.get("id", "campaign")
    columns = export_cfg.get("columns")
    control_share = float(export_cfg.get("control_share", 0.1))
    paths = []
    for fmt in ("csv", "parquet"):
        paths.append(export.export_segment(selected, campaign_id, fmt=fmt, columns=columns))
        paths.append(
            export.export_segment_ab(
                selected,
                scores,
                campaign_id,
                control_share=control_share,
                seed=int(seed),
                fmt=fmt,
                columns=columns,
            )
        )
    return paths


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
