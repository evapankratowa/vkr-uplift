# Образ приложения: Python 3.11, пакет uplift-system, полный пайплайн.
# Сборка:  docker build -t vkr-uplift .
# Запуск:  docker compose up app   (см. docker-compose.yml — БД поднимается отдельным сервисом)
FROM python:3.11-slim

WORKDIR /app

# Сначала зависимости — слой кешируется отдельно от кода.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY Makefile ./
RUN pip install --no-cache-dir -e .

# Данные не вшиваются в образ: data/ монтируется volume-ом (см. docker-compose.yml),
# либо скачивается внутри контейнера: python -m uplift_system.datasets.open_data --dataset all
CMD ["make", "all"]
