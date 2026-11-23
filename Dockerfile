# Образ приложения: Python 3.11, пакет uplift-system, полный пайплайн.
# Сборка:  docker build -t vkr-uplift .
# Запуск:  docker compose up app   (см. docker-compose.yml — БД поднимается отдельным сервисом)
FROM python:3.11-slim

WORKDIR /app

# make нужен точке входа (CMD make all) — в slim-образе его нет по умолчанию.
# Компилятор не требуется: все зависимости requirements.txt имеют готовые
# wheels под linux/aarch64 и linux/amd64 (econml вынесен в requirements-stage5.txt).
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# Сначала зависимости — слой кешируется отдельно от кода.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY Makefile ./
RUN pip install --no-cache-dir -e .

# Данные не вшиваются в образ: data/ монтируется volume-ом (см. docker-compose.yml),
# либо скачивается внутри контейнера: python -m uplift_system.datasets.open_data --dataset all
# По умолчанию прогоняется весь пайплайн: simulate -> train -> evaluate -> economics -> segments -> checks.
# Отдельный этап: docker compose run --rm app make simulate
# Сверка с золотыми эталонами tests/golden/ (допуск 1e-6):
#   docker compose run --rm app pytest tests/test_golden.py
CMD ["make", "all"]
