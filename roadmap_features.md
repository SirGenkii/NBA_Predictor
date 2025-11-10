# Feature Roadmap (Prematch-only / Home-Away representation)

## Goals

- **Single-line per match** dataset oriented around `HOME_TEAM_ID`, `AWAY_TEAM_ID`, `GAME_DATE`. All features must be prematch-only so the same code can serve both offline training and online predictions (Feast-ready).
- **Silver** contains the full feature set + every target (`IS_WIN`, `POINT_DIFF`, `POINT_TOTAL`, spread-related targets later).  
  **Gold** is a filtered view per target (e.g., `POINT_TOTAL`) with the other targets dropped and any target-specific filters applied.
- **No post-match leakage**: columns like `fieldGoalsMade_traditional`, `TEAM_NAME`, `IS_WIN_SHIFTED`, etc., are only intermediate helpers and must be dropped before saving silver/gold.
- **Modular feature library** that we can later map to Feast FeatureViews: `TeamFeatures`, `MatchupFeatures`, `AvailabilityFeatures`, `OddsFeatures`, `TargetDerivations`.

## Refactor Plan

### 1. Bronze → Silver Rework
1. **Aggregate match-level rows**
   - Continue building team-level stats, but pivot to a single match row (`HOME_`, `AWAY_`, `DIFF_` columns) before we start generating derived features.
   - Keep only IDs and columns needed to compute rollings (e.g., `points_traditional`). Once a rolling is computed, drop the raw column immediately.
2. **Shift logic sanity**
   - For each feature requiring history (`WIN_STREAK`, `ROLL_PTS_*`), confirm the data is sorted by `TEAM_ID/OPP_TEAM_ID` + `GAME_DATE` before applying `shift(1)`.  
   - Document how `IS_WIN_SHIFTED` (and similar helper columns) are computed, used, and dropped—so there’s no doubt the features represent “values prior to the match”.
3. **Availability / injuries**
   - Replace the current per-game columns with rolling aggregates only: `% of last N games with a top player absent`, `avg top_player_absent_rate` etc.  
   - Drop `has_absent`, `top_player_absent`, etc., from the final dataset.
4. **Matchup features**
   - Generate explicit matchup columns: `MATCH_PACE_n = (HOME pace + AWAY pace)/2`, `PACE_DIFF_n = HOME pace - AWAY pace`, `TOTAL_POINTS_EXPECTED_n`, `OFF_DEF_GAP_n`, etc.
   - Compute home/away specific metrics (home offense vs away defense) so we can run a single row per match.
5. **Dropping raw columns**
   - Maintain a whitelist / pattern match (e.g., keep columns starting with `HOME_`, `AWAY_`, `DIFF_`, `MATCH_`, `TOTAL_`, `ODDS`). All other columns (especially raw boxscore stats) should be dropped before saving silver.

### 2. Gold per Target
1. Define `gold_steps_for_target` to drop the other targets (`IS_WIN`, etc.) and apply any quality checks.
2. Ensure gold contains only the features + target relevant for the modeling stage (no identifiers except `GAME_ID`, `GAME_DATE`, `HOME_TEAM_ID`, `AWAY_TEAM_ID` if required by the training code).

### 3. Feature Library Structure
1. Adopt a modular layout for feature generation:  
   - `team_features.py`, `matchup_features.py`, `availability_features.py`, `odds_features.py`.  
   - Each file exports functions returning DataFrames keyed by (`GAME_ID`, `TEAM_ID`) or (`GAME_ID`) so they can be composed.
2. Introduce helper decorators or metadata (e.g., `@prematch_feature`) to indicate what stage a function belongs to and what dependencies it has.
3. Document each feature category (benefits, formula, expected columns) in this roadmap and eventually in `/docs/features`.

### 4. Online-ready pipeline (future Feast integration)
1. The new feature code should emulate the Feast structure:  
   - Entities: `team_id`, `game_id`, `match_id`.  
   - Feature views per category (team rolling stats, matchup stats, availability, odds).  
2. Once silver is purely prematch, we can create Feast feature repos pointing to the same code, ensuring consistent offline/online features.

### 5. Additional Feature Ideas
1. **Expected possessions / implied spread** from odds (using closing moneyline only).  
2. **Variance features**: `ROLL_POINTS_STD_n`, `ROLL_TOTAL_STD_n`, `ROLL_REST_STD_n`.  
3. **Home/away splits**: separate rollings for home-only and away-only stats before computing diffs.  
4. **Season context**: flags for playoff, back-to-back, travel distance.

### 6. Modeling-Friendly Output
1. Provide config files listing the final column whitelist so `DatasetConfig` doesn’t rely on dropping columns manually.  
2. The modeling notebooks should consume silver/gold without needing additional drop logic.

## Notes

- The existing feature generation code (`recipes.py`, `feature_builder.py`) is due for a full rewrite following this structure. The goal is an optimized, coherent, prematch-only feature set that is easy to extend and aligns with the future Feast deployment.
- No dataset rebuild yet—this roadmap describes the work needed before the next rebuild so that the generated silver/gold parquets are already clean.
