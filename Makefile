# Каждая команда воспроизводима с чистого окружения. SEED зафиксирован в configs/.
.PHONY: install lint test data simulate train evaluate segments checks all

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .

lint:
	ruff check src tests && ruff format --check src tests

# Скачивает открытые наборы в data/raw. Выгрузку кэшбэка кладут руками
# (источник — учебный набор DANO/НИУ ВШЭ, ссылка в README, раздел «Входные и выходные данные»):
# положить файл как data/raw/cashback_raw_2022-11-15.csv
data:
	python -m uplift_system.datasets.open_data --dataset all

test:
	pytest

simulate:
	uplift simulate --config configs/sim_base.yaml --experiment configs/experiment.yaml

train:
	uplift train --config configs/models.yaml

evaluate:
	uplift evaluate --config configs/models.yaml

segments:
	uplift select-segment --config configs/campaign.yaml

checks:
	uplift check --all

all: simulate train evaluate segments checks
