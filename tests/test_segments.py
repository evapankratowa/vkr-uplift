"""Сборщик сегментов — этап 7, код ещё не написан.

Тесты объявлены заранее, чтобы список ограничений был виден до реализации,
и помечены xfail: они обязаны покраснеть, когда сборщик появится.
"""

import pytest

pytestmark = pytest.mark.xfail(reason="segments/* реализуется на этапе 7", strict=False)


def test_budget_respected() -> None:
    """Размер отобранного сегмента не превышает budget_contacts."""
    raise NotImplementedError


def test_no_client_in_two_campaigns() -> None:
    """При max_campaigns_per_client=1 пересечение сегментов пустое."""
    raise NotImplementedError


def test_negative_uplift_excluded() -> None:
    """Клиенты с uplift ниже min_uplift не попадают в выгрузку."""
    raise NotImplementedError


def test_risk_threshold_respected() -> None:
    """Верхняя граница ДИ uplift по выплатам банка не превышает risk_uplift_max."""
    raise NotImplementedError


def test_export_filename_has_project_prefix() -> None:
    """Имя файла выгрузки начинается с имени папки проекта."""
    raise NotImplementedError
