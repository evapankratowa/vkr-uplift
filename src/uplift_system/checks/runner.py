"""Проверки как код: карточка объявляется до прогона, вердикт пишется после.

Вердикт — одно из пяти: сошлось | разошлось, объяснимо | стоп | не выполнена |
невыполнима. Вердикту сопоставлен цвет: «сошлось» — зелёный, «стоп» — красный,
остальные три — жёлтый (проверка пропущена из-за отсутствия входа, расхождение
объяснимо или проверка невыполнима по внешним причинам).

Прогон не падает на исключениях внутри проверки: ошибка превращается в вердикт
(«стоп» для карточек с on_fail == «СТОП», иначе «разошлось, объяснимо»), чтобы
отчёт собирался всегда, а код возврата отражал все проблемы разом. Машиночитаемый
отчёт по умолчанию кладётся в checks/checks.json.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..paths import CHECKS

VERDICT_PASS = "сошлось"
VERDICT_FAIL_EXPLAINED = "разошлось, объяснимо"
VERDICT_STOP = "стоп"
VERDICT_NOT_RUN = "не выполнена"
VERDICT_INFEASIBLE = "невыполнима"

_VERDICT_COLORS = {
    VERDICT_PASS: "green",
    VERDICT_FAIL_EXPLAINED: "yellow",
    VERDICT_STOP: "red",
    VERDICT_NOT_RUN: "yellow",
    VERDICT_INFEASIBLE: "yellow",
}

_COLOR_RU = {"green": "зелёная", "yellow": "жёлтая", "red": "красная"}


def verdict_color(verdict: str) -> str:
    """Цвет вердикта для отчёта: green | yellow | red."""
    return _VERDICT_COLORS.get(verdict, "yellow")


@dataclass
class Outcome:
    """Результат одной проверки: статус, найденное значение и пояснение.

    status — "pass" | "fail" | "skip". При "fail" вердикт карточки решает её
    on_fail: «СТОП» → «стоп» (красный), иначе → «разошлось, объяснимо» (жёлтый).
    При "skip" вердикт — «не выполнена» (жёлтый): проверке не хватает входа
    (файла данных или прогона команды), это не дефект данных.
    """

    status: str
    value: Any = None
    note: str = ""

    @classmethod
    def passed(cls, value: Any = None, note: str = "") -> "Outcome":
        """Проверка сошлась: value — найденное значение, note — расшифровка."""
        return cls("pass", value, note)

    @classmethod
    def failed(cls, value: Any = None, note: str = "") -> "Outcome":
        """Проверка разошлась: вердикт решает on_fail карточки."""
        return cls("fail", value, note)

    @classmethod
    def skipped(cls, note: str) -> "Outcome":
        """Проверка пропущена: нет входа (note обязателен, поясняет какого)."""
        return cls("skip", None, note)


@dataclass
class Check:
    """Карточка проверки."""

    id: str
    what: str
    compare: str
    expect: str
    on_fail: str  # "СТОП" | "ИДУ ДАЛЬШЕ С ПОМЕТКОЙ"
    fn: Callable[[], Outcome] | None = None
    result: Any = None
    verdict: str = "не выполнена"
    note: str = ""


@dataclass
class CheckRunner:
    """Прогоняет карточки и складывает результат в checks/checks.json."""

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        """Добавляет карточку в прогон (в порядке объявления)."""
        self.checks.append(check)

    def run(self) -> None:
        """Выполняет все карточки, заполняя result, verdict и note каждой."""
        for check in self.checks:
            self._run_one(check)

    @staticmethod
    def _run_one(check: Check) -> None:
        """Одна карточка: исключения становятся вердиктом, а не обрывом прогона."""
        if check.fn is None:
            check.verdict = VERDICT_NOT_RUN
            check.note = check.note or "функция проверки не задана"
            return
        try:
            outcome = check.fn()
        except Exception as exc:
            check.result = None
            check.verdict = VERDICT_STOP if check.on_fail == "СТОП" else VERDICT_FAIL_EXPLAINED
            check.note = f"исключение в проверке: {type(exc).__name__}: {exc}"
            return
        check.result = outcome.value
        if outcome.status == "skip":
            check.verdict = VERDICT_NOT_RUN
            check.note = outcome.note or "пропущено: нет входа"
        elif outcome.status == "pass":
            check.verdict = VERDICT_PASS
            check.note = outcome.note
        elif check.on_fail == "СТОП":
            check.verdict = VERDICT_STOP
            check.note = outcome.note
        else:
            check.verdict = VERDICT_FAIL_EXPLAINED
            check.note = outcome.note

    @property
    def exit_code(self) -> int:
        """0, если красных вердиктов нет; 1, если есть хотя бы один «стоп»."""
        return 1 if any(check.verdict == VERDICT_STOP for check in self.checks) else 0

    def counts(self) -> dict[str, int]:
        """Число карточек по цветам: green | yellow | red."""
        counts = {"green": 0, "yellow": 0, "red": 0}
        for check in self.checks:
            counts[verdict_color(check.verdict)] += 1
        return counts

    def _by_verdict(self) -> dict[str, int]:
        """Число карточек по вердиктам (русские названия, как в отчёте)."""
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.verdict] = counts.get(check.verdict, 0) + 1
        return counts

    def to_json(self, path: Path | None = None) -> Path:
        """Пишет machine-readable отчёт (по умолчанию checks/checks.json)."""
        target = Path(path) if path is not None else CHECKS / "checks.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated": date.today().isoformat(),
            "summary": {
                "total": len(self.checks),
                **self.counts(),
                "by_verdict": self._by_verdict(),
                "exit_code": self.exit_code,
            },
            "checks": [
                {
                    "id": check.id,
                    "what": check.what,
                    "compare": check.compare,
                    "expect": check.expect,
                    "on_fail": check.on_fail,
                    "result": _jsonable(check.result),
                    "verdict": check.verdict,
                    "color": verdict_color(check.verdict),
                    "note": check.note,
                }
                for check in self.checks
            ],
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target

    def to_markdown(self) -> str:
        """Таблица отчёта проверок: проверка | что с чем | ожидали | получили | вердикт."""
        counts = self.counts()
        lines = [
            f"# Отчёт проверок — {date.today().isoformat()}",
            "",
            f"**Итог:** {len(self.checks)} проверок — зелёных {counts['green']}, "
            f"жёлтых {counts['yellow']}, красных {counts['red']}. "
            f"Код возврата: {self.exit_code}.",
            "",
            "Цвета: зелёная — сошлось; жёлтая — пропущена (нет входа) или объяснимое "
            "расхождение; красная — стоп.",
            "",
            "| # | Проверка | Что с чем | Ожидали | Получили | Вердикт | Пояснение |",
            "|---:|---|---|---|---|---|---|",
        ]
        for number, check in enumerate(self.checks, start=1):
            verdict = f"{check.verdict} ({_COLOR_RU[verdict_color(check.verdict)]})"
            row = (
                f"| {number} | {_cell(check.id)} | {_cell(check.compare)} | {_cell(check.expect)} "
                f"| {_cell(_fmt(check.result))} | {_cell(verdict)} | {_cell(check.note)} |"
            )
            lines.append(row)
        lines.append("")
        return "\n".join(lines)


def _cell(text: Any) -> str:
    """Текст ячейки таблицы: без символов, ломающих разметку."""
    return str(text).replace("|", "∕").replace("\n", " ")


def _fmt(value: Any) -> str:
    """Найденное значение проверки в читаемый вид для колонки «получили»."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "; ".join(f"{key} = {_fmt(val)}" for key, val in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item) for item in value)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Значение результата — в типы стандартного json (numpy/pandas — через item())."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            return str(value)
    return str(value)
