PYTHON ?= python3
VENV_DIR ?= .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON_BIN := $(VENV_BIN)/python
PIP_BIN := $(VENV_BIN)/pip
export PYTHONPATH := $(CURDIR)/src

INGEST_TS ?= $(shell date -u +%Y%m%dT%H%M%SZ)
MATCH_DATE ?= $(shell date -u +%Y-%m-%d)
MODEL_URI ?=

DC := docker compose  # Pour Docker Compose V2 (officiel)
# DC := docker-compose  # Pour compatibilité Docker Compose V1

.PHONY: venv install activate freeze rebuild-bronze update-data train-model predict \
	up down logs restart clean ps exec build-containers rebuild-containers \
	run-team-game-facts run-team-boxscores-agg run-player-availability \
	run-team-form-windowed run-matchups-h2h-base run-matchups-h2h-features

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP_BIN) install --upgrade pip

venv: $(PYTHON_BIN)

install: venv requirements.txt
	$(PIP_BIN) install -r requirements.txt

activate: venv
	@echo "Pour activer l'environnement : source $(VENV_BIN)/activate"

freeze: install
	$(PIP_BIN) freeze > requirements.lock

rebuild-bronze: install
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -m src.scripts.rebuild_bronze \
		--ingest-ts "$(INGEST_TS)" \
		--include-raw \
		--include-raw-last \
		--normalise-columns

update-data: install
	@mkdir -p logs
	@echo "Running update_data_flow (logs/update_data_flow.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from flows import update_data_flow; update_data_flow(ingest_ts=os.environ['INGEST_TS'], rebuild_bronze_data=False)" >> logs/update_data_flow.log 2>&1
	@echo "update_data_flow completed. See logs/update_data_flow.log"

run-team-game-facts: install
	@mkdir -p logs
	@echo "Running build_team_game_facts (logs/team_game_facts.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from nba_predictor.transformations.team_game_facts import build_team_game_facts; build_team_game_facts(ingest_ts=os.environ['INGEST_TS'])" >> logs/team_game_facts.log 2>&1
	@echo "build_team_game_facts completed. See logs/team_game_facts.log"

run-team-boxscores-agg: install
	@mkdir -p logs
	@echo "Running build_team_boxscores_agg (logs/team_boxscores_agg.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from nba_predictor.transformations.team_boxscores_agg import build_team_boxscores_agg; build_team_boxscores_agg(ingest_ts=os.environ['INGEST_TS'])" >> logs/team_boxscores_agg.log 2>&1
	@echo "build_team_boxscores_agg completed. See logs/team_boxscores_agg.log"

run-player-availability: install
	@mkdir -p logs
	@echo "Running build_player_availability (logs/player_availability.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from nba_predictor.transformations.player_availability import build_player_availability; build_player_availability(ingest_ts=os.environ['INGEST_TS'])" >> logs/player_availability.log 2>&1
	@echo "build_player_availability completed. See logs/player_availability.log"

run-team-form-windowed: install
	@mkdir -p logs
	@echo "Running build_team_form_windowed (logs/team_form_windowed.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from nba_predictor.transformations.team_form_windowed import build_team_form_windowed; build_team_form_windowed(ingest_ts=os.environ['INGEST_TS'])" >> logs/team_form_windowed.log 2>&1
	@echo "build_team_form_windowed completed. See logs/team_form_windowed.log"

run-matchups-h2h-base: install
	@mkdir -p logs
	@echo "Running build_matchups_h2h_base (logs/matchups_h2h_base.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from nba_predictor.transformations.matchups_h2h_base import build_matchups_h2h_base; build_matchups_h2h_base(ingest_ts=os.environ['INGEST_TS'])" >> logs/matchups_h2h_base.log 2>&1
	@echo "build_matchups_h2h_base completed. See logs/matchups_h2h_base.log"

run-matchups-h2h-features: install
	@mkdir -p logs
	@echo "Running build_matchups_h2h_features (logs/matchups_h2h_features.log)"
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from nba_predictor.transformations.matchups_h2h_features import build_matchups_h2h_features; build_matchups_h2h_features(ingest_ts=os.environ['INGEST_TS'])" >> logs/matchups_h2h_features.log 2>&1
	@echo "build_matchups_h2h_features completed. See logs/matchups_h2h_features.log"

run-silver-pipeline: run-team-game-facts run-team-boxscores-agg run-player-availability run-team-form-windowed run-matchups-h2h-base run-matchups-h2h-features
	@echo "Sequential silver pipeline completed. See individual logs in logs/."

train-model: install
	INGEST_TS=$(INGEST_TS) $(PYTHON_BIN) -c "import os; from flows import train_model_flow; train_model_flow(ingest_ts=os.environ.get('INGEST_TS'))"

predict: install
	MATCH_DATE=$(MATCH_DATE) MODEL_URI=$(MODEL_URI) $(PYTHON_BIN) -c "import os; from flows import predict_flow; predict_flow(match_date=os.environ['MATCH_DATE'], model_uri=os.environ.get('MODEL_URI') or None)"

up:
	$(DC) up -d

build-containers:
	$(DC) build

rebuild-containers:
	$(DC) build --no-cache

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
	$(DC) exec feast bash
