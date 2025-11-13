DC := docker compose  # Pour Docker Compose V2 (officiel)
# DC := docker-compose  # Pour compatibilité Docker Compose V1

VENV=.venv
PYTHON=$(VENV)/bin/python

install:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

activate:
	@echo "Run: source $(VENV)/bin/activate"

feast-apply:
	$(PYTHON) -m src.feast.pipeline --skip-feast-apply && feast -c src/feast apply

feast-refresh:
	$(PYTHON) -m src.feast.pipeline --seasons 2025-26 --targets POINT_TOTAL IS_WIN --materialize

model-point-total:
	$(PYTHON) -m src.modeling.run_point_total --tune --trials 20

model-is-win:
	$(PYTHON) -m src.modeling.run_is_win --tune --trials 2

up:
	$(DC) up -d

down:
	$(DC) down

logs:
	$(DC) logs -f

restart:
	$(DC) down && $(DC) up -d

clean:
	$(DC) down -v

ps:
	$(DC) ps

exec:
	$(DC) exec jupyter-gpu bash
