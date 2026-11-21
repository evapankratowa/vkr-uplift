"""Командный интерфейс системы.

Пять команд соответствуют функциональным требованиям главы 1 ВКР:
  simulate        смоделировать эксперимент и собрать обучающую выборку
  train           обучить uplift-модели по конфигу
  evaluate        посчитать метрики и сравнить стратегии
  select-segment  собрать сегмент для рассылки и выгрузить его
  check           прогнать проверки и собрать отчёт
"""

import argparse


def main(argv: list[str] | None = None) -> int:
    """Точка входа. Возвращает код возврата процесса."""
    parser = argparse.ArgumentParser(prog="uplift", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="смоделировать эксперимент")
    p_sim.add_argument("--config", default="configs/sim_base.yaml")
    p_sim.add_argument("--experiment", default="configs/experiment.yaml")

    p_train = sub.add_parser("train", help="обучить модели")
    p_train.add_argument("--config", default="configs/models.yaml")
    p_train.add_argument("--dataset", default=None, help="synthetic | x5 | hillstrom")

    p_eval = sub.add_parser("evaluate", help="метрики и сравнение стратегий")
    p_eval.add_argument("--config", default="configs/models.yaml")

    p_seg = sub.add_parser("select-segment", help="собрать сегмент для рассылки")
    p_seg.add_argument("--config", default="configs/campaign.yaml")

    p_check = sub.add_parser("check", help="прогнать проверки")
    p_check.add_argument("--all", action="store_true")

    args = parser.parse_args(argv)
    raise NotImplementedError(f"команда {args.command} ещё не реализована")


if __name__ == "__main__":
    raise SystemExit(main())
