"""Подключение к PostgreSQL: строка из DATABASE_URL, дефолт — учебный стенд.

Дефолт совпадает с docker-compose.yml (postgres:16, БД и роль uplift).
SQLAlchemy импортируется лениво: в окружении без драйвера пакет поднимается,
а обращение к БД падает только при реальном вызове.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL = "postgresql://uplift:uplift@localhost:5432/uplift"


def database_url(explicit: str | None = None) -> str:
    """Строка подключения: явный аргумент, затем DATABASE_URL, затем дефолт."""
    return explicit or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_connection(url: str | None = None, connect_timeout: int | None = None) -> Engine:
    """Создаёт SQLAlchemy Engine (драйвер psycopg 3).

    `postgresql://` без указания драйвера нормализуется в `postgresql+psycopg://`:
    дефолтный драйвер SQLAlchemy — psycopg2, а в проекте ставится psycopg 3.
    `connect_timeout` — таймаут подключения в секундах (для проверок доступности).
    """
    from sqlalchemy import create_engine

    resolved = database_url(url)
    if resolved.startswith("postgresql://"):
        resolved = "postgresql+psycopg://" + resolved[len("postgresql://") :]
    connect_args = {"connect_timeout": connect_timeout} if connect_timeout else {}
    return create_engine(resolved, connect_args=connect_args)
