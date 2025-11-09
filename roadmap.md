# NBA Predictor – Roadmap

## Current Snapshot
- **Dataset**: Historical matches live under `data/01_bronze` (merged raw boxscores & games for all seasons plus “last” updates). Notebook `05_boxscores_feature_enginering.ipynb` currently jumps straight from bronze inputs to fully engineered tables: it saves a `final` CSV (features + targets) and a “final cleaned” version where post-match columns are manually dropped. These two CSVs implicitly behave like our `02_silver` and `03_gold` layers, but the transformation is notebook-driven, not scripted. Each matchup is still duplicated into two rows (`TEAM_ID`, `OPP_TEAM_ID`, `IS_HOME`, `IS_WIN`).
- **Feature engineering**: `src/feature_builder.py` and `src/config.py` define rollups, win streaks, Elo, rest advantage, etc. Odds columns are present but mostly dropped before modeling.
- **Modeling code**: `src/modeling.py` only supports binary classification (`IS_WIN`). `POINT_DIFF` is computed upstream but is not used anywhere else in the repo besides `src/modeling.py:32-47` and the drop lists in `src/config.py:15-16`.
- **Pipelines**: Logs reference `src/nba_predictor/datasets/training.py` and Prefect-like flows (`flows/train_model_flow.py`), but the corresponding source files are absent in the current working tree, so dataset building needs to be restored or rewritten.
- **Tracking & Feature store**: No MLflow or Feast configuration exists yet. `data/v2/gold/monitoring_snapshots` hints at manual monitoring, but nothing is wired into code.

---

## Milestone 0 – Restore data builders & audit the gold set
1. **Stabilize notebook-driven builders**  
   - Keep `05_boxscores_feature_enginering.ipynb` as the orchestrator, but extract its heavy lifting into versioned Python modules (`src/datasets/silver.py`, `src/datasets/gold.py`, etc.).  
   - Ensure the notebook simply calls these helpers to populate `data/02_silver` (match-level aggregates + raw targets) and `data/03_gold` (clean, leak-safe tables).  
   - Preserve the “two rows per match” contract while documenting every transformation step so future edits happen in code, not scattered notebook cells.
2. **Dataset manifest & schema registry**  
   - Emit a data dictionary (column name, source, transformation, dtype, nullability, season coverage).  
   - Version training sets under `data/03_gold/training_sets/<ts>.parquet` with metadata (ingest timestamp, feature hashes) to guarantee reproducibility for MLflow.
3. **Quality gates**  
   - Build validation scripts that assert row counts per season, duplicates per (`GAME_ID`, `TEAM_ID`), and sanity ranges for scoring fields (`PTS`, `OPP_PTS`, odds).  
   - Snapshot aggregate stats (mean totals, pace, possessions) for monitoring regressions when new raw data is appended.
4. **Readable feature/target recipes**  
   - Reorganize the feature engineering code into small, well-documented functions (e.g., `compute_team_rollups`, `build_point_targets`) so the notebook and future scripts can import them.  
   - Clearly tag each function with the stages/targets it feeds to keep the “final” vs “final cleaned” distinction transparent.
5. **Dataset entrypoints**  
   - Provide both a CLI (`python -m src.datasets.build --build-from-bronze ...`) and a companion notebook (`06_build_datasets.ipynb`) so data builds can run outside of legacy notebooks while staying user-friendly.  
   - The CLI/notebook should rely on a shared `src/datasets/bronze.py` assembler that converts the latest `games`/`boxscores` CSVs into match-level rows (two per game) with odds + top-player features before handing off to the silver/gold builders.

---

## Milestone 1 – Feature layer upgrades (points, pivots & leakage control)
1. **Explicit scoring targets**  
   - Ensure the gold dataset exposes `POINTS_FOR`, `POINTS_AGAINST`, `POINT_DIFF = POINTS_FOR - POINTS_AGAINST`, and `POINT_TOTAL = POINTS_FOR + POINTS_AGAINST` for every row.  
   - Extend `COLS_TO_DROP_TARGET_*` to reflect the additional targets and keep symmetry between home/away rows.
2. **Feature-plan system**  
   - Introduce a declarative `FeaturePlan` (YAML/JSON or Python dataclass) describing how each feature is computed, which stage it belongs to, and which targets it’s valid for.  
   - Encode column provenance (e.g., `source: boxscores`, `window: 10`, `requires_future_info: false`) so we can automatically filter out leakage-prone columns when exporting gold sets or training MLflow runs.  
   - Provide CLI entry points such as `python -m src.datasets.build --stage silver` and `--stage gold --target point_total` so we never rely on manual notebook drops again.
3. **Pivot & Gaussian context features**  
   - Derive bookmaker pivots when odds/lines are available (convert moneyline + spread to implied totals; otherwise use rolling/team means).  
   - Create rolling aggregates for `POINTS_FOR`, `POINTS_AGAINST`, totals, and differentials over N ∈ {3,5,10,25,50,100,200}.  
   - Compute opponent-adjusted z-scores (team rolling mean minus opponent rolling mean) to feed the Gaussian heads.
4. **Match-level totals**  
   - De-duplicate totals per game (one canonical total per `GAME_ID`) to support supervision for models predicting the combined score directly.  
   - Store helper tables (`game_totals`, `game_spreads`) that will later become Feast feature views.
5. **Metadata**  
   - Track season, playoff flag, rest days, travel distance, injury context to capture variance drivers for totals.

---

## Milestone 2 – Modeling system rewrite
1. **Modular package**  
   - Replace `src/modeling.py` with a package (e.g., `src/modeling/__init__.py`, `datasets.py`, `features.py`, `pipelines.py`, `models.py`).  
   - Create `DatasetConfig` objects describing target, features to drop, label transformations, and evaluation slices.
2. **Targets & heads**  
   - `WinClassifier`: existing classification baseline (probability of `IS_WIN`).  
   - `PointsForRegressor`: predicts team points (per row).  
   - `PointDiffGaussian`: predicts mean & variance of `POINT_DIFF`.  
   - `PointTotalGaussian`: predicts mean & variance of `POINT_TOTAL` (game-level or derived from the two-row representation).  
   - Optionally `SpreadRegressor` to align with betting lines.
3. **Gaussian distribution system**  
   - Implement a two-output head (μ, log σ) on top of gradient boosting or neural net regressors. Optimize negative log-likelihood of observed diff/total under `N(μ, σ²)`.  
   - Allow alternative parametrizations (predict μ via LightGBM and σ via a secondary model on residuals).  
   - Expose utilities to sample from the predicted distribution and compute probabilities of beating custom pivots.
4. **Training orchestration**  
   - CLI / script (e.g., `python -m src.modeling.train --target point_total`) that loads the gold dataset, applies preprocessing (scaling, feature selection, class balancing), splits by season, trains, evaluates, and logs artifacts.  
   - Cross-validation strategies mindful of temporal ordering (walk-forward, season-based splits).  
   - Automatic feature importance, SHAP summary, calibration diagnostics for both classification and regression heads.
5. **Evaluation metrics**  
   - Classification: ROC-AUC, log-loss, Brier score, precision@k, moneyline ROI simulation.  
   - Regression: MAE/RMSE for points, log-likelihood for Gaussian heads, coverage of prediction intervals, correlation with bookmaker totals/spreads.  
   - Store evaluation reports per season and for specific contexts (home/away, rest advantage buckets).

---

## Milestone 3 – MLflow instrumentation
1. **MLflow project scaffolding**  
   - Add `mlflow` to `requirements.txt`, create `mlflow.cfg` (tracking URI, artifact location under `data/mlruns`).  
   - Provide Makefile helpers: `make mlflow-ui`, `make train_win`, `make train_totals`.
2. **Logging hooks**  
   - Wrap training scripts so every run logs: dataset version hash, feature list, target definition, hyperparameters, metrics, plots (confusion matrix, calibration curves, residual histograms), and serialized models.  
   - Register best models per target in the MLflow Model Registry (`nba-predictor/win`, `nba-predictor/points`, `nba-predictor/point-diff`, `nba-predictor/point-total`).
3. **Autologging for notebooks**  
   - Provide helper context managers so exploratory notebooks (e.g., `08_modeling_and_save_full_dataset.ipynb`) push experiments to MLflow with minimal code changes.
4. **Deployment hooks**  
   - Export MLflow model flavors (sklearn, pyfunc) that production scripts can load, ensuring feature order consistency via the saved feature manifest.

---

## Milestone 4 – Feast feature store integration
1. **Feast repo setup**  
   - Create `feature_repo/feature_store.yaml`, configure offline store (DuckDB/Parquet over `data/v2`) and online store (SQLite for local, Redis/BigTable for prod later).  
   - Define entities: `team_id`, `matchup_id`/`game_id`, `season`.  
   - Register batch data sources for `team_game_facts`, `team_form_windowed`, `matchups_h2h_features`, `player_availability`, odds history, totals.
2. **Feature views**  
   - `TeamStatsView`: rolling stats, rest metrics, Elo diff.  
   - `MatchupH2HView`: head-to-head aggregates.  
   - `PlayerAvailabilityView`: counts and rates from `feature_aggregation`.  
   - `OddsPivotView`: bookmaker lines / implied pivots for Gaussian heads.  
   - `GameTotalsView`: canonical total/point diff labels for Feast-on-demand transforms.
3. **Materialization jobs**  
   - CLI to backfill Feast offline store for historical seasons and to materialize the latest N days into the online store for inference.  
   - Integrate with the rebuilt training script so datasets are fetched via `feast.get_historical_features(...)`, ensuring consistency between training and serving.
4. **Serving integration**  
   - Update prediction scripts (e.g., `src/predictions.py`) to request the latest Feast feature vectors given `team_id`, `opp_team_id`, `game_date`, ensuring both rows of a matchup are pulled atomically.  
   - Provide fallbacks if Feast misses (e.g., compute from cached parquet).

---

## Milestone 5 – Orchestration, monitoring, and delivery
1. **Prefect / orchestration flows**  
   - Reintroduce `flows/train_model_flow.py` (or Prefect 2 flows) to chain: raw data ingestion → feature store backfill → dataset build → model training → evaluation → MLflow logging.  
   - Parameterize flows by season ranges and targets (win, points, totals).
2. **Inference & simulation pipeline**  
   - Refresh `src/predictions.py` to fetch Feast features, load MLflow models, output win probabilities plus Gaussian distributions for diff/total, and compute implied probabilities of beating betting lines.  
   - Extend simulators (`src/simulator_betting.py`, `src/simulator_grid.py`) to consume the new predictive distributions.
3. **Monitoring**  
   - Automate population of `data/v2/gold/monitoring_snapshots` with: dataset drift stats, calibration plots, realized vs predicted totals/diffs, player availability impacts.  
   - Schedule alerts for data freshness (no new raw games in X hours) and model performance regressions (e.g., log-loss 5% worse on last week’s games).
4. **Documentation & onboarding**  
   - Update `README.md` / `README_NBA_Predictor.md` with instructions for running Feast + MLflow + training scripts, environment variables, and hardware expectations.  
   - Provide a “playbook” notebook showing how to load the gold dataset, train a Gaussian total model, and log results to MLflow.

---

## Open Questions & Dependencies
1. **Source restoration**: confirm where the missing `src/nba_predictor/datasets/*.py`, `transformations/*.py`, and `flows/*.py` files live (previous branch? remote?). The roadmap assumes they’ll be restored or rewritten.
2. **Bookmaker pivots**: specify how totals/spreads are sourced (historical odds data vs manually set pivot). Needed to align the Gaussian heads with betting markets.
3. **Compute budget**: determine whether GPU training (CatBoost/XGBoost) is required for the Gaussian models or if CPU-based LightGBM/NGBoost is sufficient.
4. **Serving target**: clarify if predictions must be real-time (requiring Feast online store) or batch (offline store only).  
5. **Evaluation ground truth**: decide whether overtime points should be capped/normalized when training totals/diffs.

This roadmap should guide the rebuild: finish Milestone 0 to regain deterministic datasets, then iterate through feature upgrades, modeling, MLflow instrumentation, Feast integration, and finally orchestration + monitoring.
