"""Точка входа модуля экономики: python -m uplift_system.economics.run.

Автономный прогон до появления T2: синтетический набор генерируется внутри
команды по configs/ с зафиксированным seed, числа инкрементальных откликов
берутся из известного истинного uplift (ITE из DGP). Сводная таблица пишется
в output/tables/<префикс>_economics_<дата>.csv и дублируется в stdout.
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..datasets.synthetic import generate, make_covariates_for_tests
from ..paths import out_path
from .core import build_economics_table, response_value_rub

logger = logging.getLogger("uplift.economics")


def main(argv: list[str] | None = None) -> int:
    """Точка входа. Возвращает код возврата процесса."""
    parser = argparse.ArgumentParser(
        prog="uplift economics",
        description="сводная таблица финансового эффекта по каналам (синтетика, seed из конфигов)",
    )
    parser.add_argument(
        "--config",
        default="configs/campaign.yaml",
        help="конфиг кампании: ценность отклика, каналы, сценарии SMS, бюджет (YAML)",
    )
    parser.add_argument(
        "--sim-config",
        default="configs/sim_base.yaml",
        help="конфиг генератора DGP для автономного синтетического прогона (YAML)",
    )
    parser.add_argument(
        "--n-boot", type=int, default=1000, help="число бутстрап-итераций по деньгам"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="куда писать CSV (по умолчанию output/tables/<префикс>_economics_<дата>.csv)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    campaign_cfg = load_config(args.config, required=("economics",))
    sim_cfg = load_config(args.sim_config, required=("covariates", "effect"))
    seed = int(campaign_cfg.get("seed", sim_cfg.get("seed", 42)))
    logger.info("конфиги загружены: %s, %s; seed %d", args.config, args.sim_config, seed)

    features = _covariates_for_economics(sim_cfg)
    dataset = generate(sim_cfg, seed=seed, features=features)
    ite_true = dataset.ite_true["purchase"].to_numpy()
    logger.info(
        "синтетика: n=%d, realized ATE %.4f; ценность отклика %.2f ₽",
        dataset.n,
        dataset.meta["ate_target_realized"],
        response_value_rub(campaign_cfg),
    )

    table = build_economics_table(ite_true, campaign_cfg, n_boot=args.n_boot, seed=seed)

    if args.out:
        path = Path(args.out)
    else:
        path = out_path("tables", f"economics_{date.today().isoformat()}", "csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    printable = table.copy()
    for col in printable.columns:
        if printable[col].dtype == float:
            printable[col] = printable[col].map(lambda x: round(x, 4))
    print("\nэкономика каналов (синтетика, seed зафиксирован):")
    print(printable.to_string(index=False))
    logger.info("готово: %s", path)
    return 0


def _covariates_for_economics(sim_cfg: dict) -> pd.DataFrame | None:
    """Ковариаты для автономного прогона: из выгрузки, а при её отсутствии — синтетические.

    В публичном репозитории выгрузки кэшбэка нет (лицензионные причины), поэтому
    команда не падает, а честно предупреждает и подставляет искусственные
    ковариаты с теми же именами колонок — воспроизводимость по seed сохраняется
    (та же логика, что в cli._covariates_for_simulation).
    """
    source = Path(sim_cfg["covariates"]["source"])
    if source.exists():
        return None  # generate прочитает файл сам по ключу covariates.source
    n_clients = int(sim_cfg.get("sample", {}).get("n_clients", 50_000))
    seed = int(sim_cfg.get("seed", 42))
    logger.warning(
        "выгрузка %s не найдена: ковариаты заменены синтетическими (n=%d, seed=%d)",
        source,
        n_clients,
        seed,
    )
    return make_covariates_for_tests(n=n_clients, seed=seed)


if __name__ == "__main__":
    raise SystemExit(main())
