# Feast Integration Roadmap

## Objectives
- Serve the same feature definitions to both offline training (current silver/gold datasets) and online inference (prediction notebooks/API) via Feast.
- Automate recent-game refresh → feature materialization → model scoring so predictions always use the latest prematch information.
- Provide a path to production (CI-friendly repo, reproducible feature registry, scheduled materializations).

## Phase 0 – Foundations
1. **Inventory existing features**
   - Map current feature modules (`src/features/*.py`, `feature_builder.py`, recipe steps) to future Feast Feature Views.
   - Document entity keys and data sources (match-level vs team-level) plus required columns (`GAME_ID`, `TEAM_ID`, etc.).
2. **Decide storage backends**
   - Offline store: Parquet/Delta on local filesystem (Feast file store) for simplicity.
   - Online store: configure SQLite immediately (upgrade to Redis later if low-latency serving is required).
3. **Create `feast/` repo**
   - Standard structure: `feature_store.yaml`, `data_sources.py`, `entities.py`, `feature_views.py`, `services.py`.
   - Add Make targets or Poetry scripts to run `feast apply`, `materialize`, `materialize-incremental`.

## Phase 1 – Data Contracts
1. **Define entities**
   - `match_id` (alias for `GAME_ID`).
   - `team_id` (two records per match for team-centric features).
   - Potential derived entities: `matchup_id` (home/away pair + date) for features that combine both teams.
2. **Data sources**
   - Bronze tables (latest merged boxscores) as raw sources.
   - Intermediate silver tables as derived sources (if easier) but prefer using bronze + transformations to avoid double-maintenance.
3. **Ingestion jobs**
   - Convert notebook logic (`03/04`) into Python functions returning clean DataFrames saved under `data/feast_sources/...`.
   - Guarantee schema stability (column names, dtypes, unique keys) before pointing Feast at those files.

## Phase 2 – Feature Definitions
1. **Modularize feature logic**
   - Refactor `src/features/*` into reusable helpers that can be called both by Feast `@feature_view` transformations and by the legacy silver pipeline during transition.
2. **Create Feature Views**
   - `TeamRollingStats`: rolling averages/variances keyed by `(team_id, event_timestamp)`.
   - `AvailabilityFeatures`: injury/suspension aggregates.
   - `OddsFeatures`: latest available bookmaker lines keyed by match.
   - `MatchupFeatures`: join two team Feature Views + contextual flags (rest, travel, back-to-back).
   - `TargetFeatures`: derived labels stored separately (for training only).
3. **Feature Services**
   - At least one service for `POINT_TOTAL` modeling that bundles the necessary feature views.

## Phase 3 – Materialization Pipeline
1. **CLI / scripts**
   - Use `python -m src.feast.pipeline` (added) to refresh recent games, rebuild bronze/silver/gold snapshots, and optionally trigger Feast commands.
   - `feast materialize <start> <end>` or `feast materialize-incremental <ts>` after every refresh; the CLI flag `--materialize` chains this automatically.
2. **Scheduling**
   - Add a `Makefile` target or Prefect/airflow flow to run nightly: scrape → rebuild bronze/silver snapshots → `feast materialize incremental`.
3. **Validation**
   - Compare Feast offline feature retrieval vs current silver dataset (row count, sample columns) to ensure parity before switching modeling notebooks to Feast.

## Phase 4 – Model Training & Prediction Integration
1. **Training**
   - Update `ModelTrainer` so that `DatasetConfig` can either point to parquet snapshots (legacy) or call `feast.get_historical_features`.
   - Store training datasets generated via Feast for reproducibility (MLflow artifact).
2. **Prediction**
   - Modify `run_prediction_pipeline` to:
     - Build `PredictionRequest` DataFrames (home/away IDs, timestamps).
     - Call `feast.get_online_features` (or `get_historical_features` with latest timestamp) to retrieve feature vectors.
     - Remove manual `run_pipeline` silver/gold steps once parity is confirmed.
3. **Caching**
   - Add lightweight cache (joblib/Parquet) of the materialized features for scenarios where Feast is unavailable in notebooks.

## Multi-target Feature Services
1. **Shared feature views**
   - Keep core feature views (team rolling stats, matchup context, availability, odds) target-agnostic. They are materialized once and reused everywhere.
2. **Target-specific projections**
   - Build thin Feature Views for label engineering or target-specific heuristics (e.g., point-total calibrated residuals, win-probability odds deltas). These views depend on the same sources but only contain the columns required by that modeling task.
3. **Feature Services**
   - Compose services per target: `point_total_service` might bundle `matchup_features_view` + `home/away` team projections, while `is_win_service` adds classification-centric features. Services reference the same base Feature Views, so materialization cost stays shared.
4. **Model configs**
   - Modeling notebooks/CLI pick the right `FeatureService` based on the desired target, ensuring that additional targets can be onboarded without duplicating upstream feature definitions.

## Phase 5 – Ops & Monitoring
1. **CI Checks**
   - Lint Feast repo, run `feast plan` to ensure definitions are valid.
   - Optional: unit tests that call feature transformations on sample data.
2. **Data Quality**
   - Implement expectation suites (Great Expectations/whylogs) on the Feast offline store before materialization.
3. **Observability**
   - Log materialization metrics (rows processed, latency) and attach to MLflow runs or monitoring dashboards.

## Open Questions / TBD
- Do we need an online store immediately? If not, plan for easy plug-in later.
- Permissions & environment when running scraping + Feast materialization in CI/CD.
- Best way to share the same feature definitions with other modeling tasks that live outside Feast (e.g., legacy notebooks).
- **Current workflow (WIP)**
  1. `python -m src.feast.pipeline --seasons 2024-25 --targets POINT_TOTAL IS_WIN --materialize`  
     ↳ refreshes raw data, rebuilds bronze/silver/gold artifacts, runs `feast apply`, and materializes recent features.
  2. `python -m src.modeling.run_point_total --tune --trials 20` (or `make model-point-total`) trains via Feast features and logs to MLflow. Prediction pipeline already consumes the same FeatureService.
