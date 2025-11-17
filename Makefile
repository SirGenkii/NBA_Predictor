DC := docker compose  # Pour Docker Compose V2 (officiel)
# DC := docker-compose  # Pour compatibilité Docker Compose V1

VENV=.venv
PYTHON=$(VENV)/bin/python

NBA_MASS_PID_FILE ?= data/mass_prediction_nba/nba_mass_prediction.pid
NBA_MASS_LOG_FILE ?= data/mass_prediction_nba/logs/nba_mass_prediction_watcher.log

install:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

activate:
	@echo "Run: source $(VENV)/bin/activate"

feast-apply:
	$(PYTHON) -m src.feast.pipeline --skip-feast-apply && feast -c src/feast apply

#TO RUN MANUALY : python -m src.feast.pipeline --seasons 2025-26 --targets POINT_TOTAL IS_WIN --materialize
feast-refresh: 
	$(PYTHON) -m src.feast.pipeline --seasons 2025-26 --targets POINT_TOTAL IS_WIN --materialize


MODEL_PT_FLAGS ?=
model-point-total:
	$(PYTHON) -m src.modeling.run_point_total $(MODEL_PT_FLAGS)


model-point-total-tuned:
	$(PYTHON) -m src.modeling.run_point_total --tune --trials 1

model-point-total-prod:
	$(PYTHON) -m src.modeling.run_point_total --register-prod

model-point-total-tuned-prod:
	$(PYTHON) -m src.modeling.run_point_total --tune --trials 1 --register-prod

# Limitation mémoire via cgroup (systemd-run) pour éviter l'OOM tout en permettant le swap.
# Exemple: MEMORY_HIGH=75% make model-point-total-tuned-prod-capped
MEMORY_HIGH ?= 75%
MEMORY_MAX ?= infinity  # laisser infinité pour autoriser le dépassement via swap
model-point-total-tuned-prod-capped:
	@if command -v systemd-run >/dev/null 2>&1; then \
		systemd-run --user --scope -p MemoryHigh=$(MEMORY_HIGH) -p MemoryMax=$(MEMORY_MAX) \
			$(PYTHON) -m src.modeling.run_point_total --tune --trials 1 --register-prod; \
	else \
		echo "systemd-run introuvable: impossible d'appliquer le plafond mémoire cgroup."; \
		exit 1; \
	fi


model-is-win:
	$(PYTHON) -m src.modeling.run_is_win

model-is-win-tuned:
	$(PYTHON) -m src.modeling.run_is_win --tune --trials 20

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

nba-mass-start:
	@mkdir -p $(dir $(NBA_MASS_LOG_FILE)) $(dir $(NBA_MASS_PID_FILE))
	@if [ -f $(NBA_MASS_PID_FILE) ] && kill -0 $$(cat $(NBA_MASS_PID_FILE)) 2>/dev/null; then \
		echo "NBA mass prediction watcher already running (PID $$(cat $(NBA_MASS_PID_FILE)))."; \
	else \
		echo "Starting NBA mass prediction watcher..."; \
		nohup $(PYTHON) -m src.scripts.mass_prediction_watcher > $(NBA_MASS_LOG_FILE) 2>&1 & \
		echo $$! > $(NBA_MASS_PID_FILE); \
		echo "Watcher started with PID $$(cat $(NBA_MASS_PID_FILE)). Logs -> $(NBA_MASS_LOG_FILE)"; \
	fi

nba-mass-stop:
	@if [ -f $(NBA_MASS_PID_FILE) ]; then \
		PID=$$(cat $(NBA_MASS_PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping NBA mass prediction watcher (PID $$PID)..."; \
			kill $$PID && echo "Watcher stopped."; \
		else \
			echo "PID $$PID not running."; \
		fi; \
		rm -f $(NBA_MASS_PID_FILE); \
	else \
		echo "No NBA watcher PID file found."; \
	fi

nba-mass-restart: nba-mass-stop
	@sleep 1
	@$(MAKE) nba-mass-start
