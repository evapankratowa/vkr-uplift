"""Проверки как код: карточка объявляется до прогона, вердикт пишется после.

Вердикт — одно из пяти: сошлось | разошлось, объяснимо | стоп | не выполнена |
невыполнима.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Check:
    """Карточка проверки."""

    id: str
    what: str
    compare: str
    expect: str
    on_fail: str  # "СТОП" | "ИДУ ДАЛЬШЕ С ПОМЕТКОЙ"
    fn: Callable[[], Any] | None = None
    result: Any = None
    verdict: str = "не выполнена"
    note: str = ""


@dataclass
class CheckRunner:
    """Прогоняет карточки и складывает результат в checks/checks.json."""

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def to_json(self, path: Path | None = None) -> Path:
        raise NotImplementedError

    def to_markdown(self) -> str:
        """Таблица отчёта проверок: проверка | что с чем | ожидали | получили | вердикт."""
        raise NotImplementedError
