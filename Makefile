# Каждая команда воспроизводима с чистого окружения. SEED зафиксирован в configs/.
# UPLIFT — консольная команда пакета (появляется после make install); можно
# переопределить, например: make all UPLIFT="python -m uplift_system.cli".
UPLIFT ?= uplift
.PHONY: install lint test data simulate train evaluate economics segments checks all

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .

lint:
	ruff check src tests && ruff format --check src tests

# Скачивает открытые наборы в data/raw. Шаг ОПЦИОНАЛЕН: требует прямую сеть
# (хосты открытых датасетов) и не входит в make all — без него пайплайн идёт
# на синтетических ковариатах (fallback с предупреждением, seed в configs/).
# Выгрузку кэшбэка кладут руками (источник — учебный набор DANO/НИУ ВШЭ,
# ссылка в README, раздел «Входные и выходные данные»): положить файл как
# data/raw/cashback_raw_2022-11-15.csv
data:
	python -m uplift_system.datasets.open_data --dataset all

test:
	pytest

simulate:
	$(UPLIFT) simulate --config configs/sim_base.yaml --experiment configs/experiment.yaml

train:
	$(UPLIFT) train --config configs/models.yaml

evaluate:
	$(UPLIFT) evaluate --config configs/models.yaml

economics:
	$(UPLIFT) economics --config configs/campaign.yaml

segments:
	$(UPLIFT) select-segment --config configs/campaign.yaml

checks:
	$(UPLIFT) check --all

all: simulate train evaluate economics segments checks
