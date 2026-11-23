# Каждая команда воспроизводима с чистого окружения. SEED зафиксирован в configs/.
# Локально цели используют .venv, созданный make install; в Docker, где venv нет,
# используются команды системного Python. PYTHONPATH очищается от чужих пакетов.
# UPLIFT можно переопределить вручную.
VENV_PYTHON := .venv/bin/python
PYTHON := env -u PYTHONPATH $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python)
UPLIFT ?= env -u PYTHONPATH $(if $(wildcard .venv/bin/uplift),.venv/bin/uplift,uplift)
PYTEST := env -u PYTHONPATH $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)
RUFF := env -u PYTHONPATH $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
.PHONY: install lint test data simulate train evaluate economics segments checks all

install:
	python -m venv .venv
	env -u PYTHONPATH $(VENV_PYTHON) -m pip install -r requirements.txt
	env -u PYTHONPATH $(VENV_PYTHON) -m pip install -e .

lint:
	$(RUFF) check src tests && $(RUFF) format --check src tests

# Скачивает открытые наборы в data/raw. Шаг ОПЦИОНАЛЕН: требует прямую сеть
# (хосты открытых датасетов) и не входит в make all — без него пайплайн идёт
# на синтетических ковариатах (fallback с предупреждением, seed в configs/).
# Выгрузку кэшбэка кладут руками (источник — учебный набор DANO/НИУ ВШЭ,
# ссылка в README, раздел «Входные и выходные данные»): положить файл как
# data/raw/cashback_raw_2022-11-15.csv
data:
	$(PYTHON) -m uplift_system.datasets.open_data --dataset all

test:
	$(PYTEST)

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
