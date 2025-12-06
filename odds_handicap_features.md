# Odds Handicap Features

Brain dump on how to turn the new `handicap` JSON column (over/under pivots with odds) into stable model features, covering old sparse data and recent rich snapshots.

## Data shapes to support
- Sparse without odds: few pivots with default odds `1.73 / 1.73` (historical backfill).
- Sparse with near-fair odds: 1–3 pivots around the bookmaker center (≈1.9 / 1.9) with mild asymmetry possible.
- Dense with full curve: long list of pivots with asymmetric odds (recent bet365 scrape; matches screenshot use-case).

Each entry: `{label: float, odds_over: float, odds_under: float}` inside a JSON map keyed by the label as string.

## Conversions and normalization
- Convert decimal odds to implied probabilities: `p_raw = 1 / odds`.
- Remove overround per pivot: `p_over = p_raw_over / (p_raw_over + p_raw_under)`; `p_under = 1 - p_over`. Keep the overround size as a feature.
- Sort by label; treat labels as total points pivots.

## Canonical curve representation
Goal: approximate the market CDF of game total points (probability of Over as function of pivot) even when only a few pivots exist.

Recommended approach (train + inference):
- **Interpolation/Fit**: Fit a monotone logistic (or probit) curve to `(label, p_over)` after overround removal. With 1 pivot, fix slope to a prior (see fallback); with 2–3 pivots, estimate slope; with many pivots, fit normally and smooth.
- **Vig features**: average overround, max overround, std of overround across pivots.
- **Central line**: pivot where `p_over = 0.5` on the fitted curve (market total). If only one pivot exists, use that label as central.
- **Slope near line**: derivative of `p_over` at the central line (proxy for market uncertainty/variance).
- **Width & coverage**: min/max label, range, pivot count, average spacing.
- **Asymmetry**: difference in `p_over` between neighboring pivots around the line; skew of odds differences.
- **Edge features**: `p_over` and `p_under` at the provided pivots (top-N ordered by proximity to central), capped to handle sparse cases.
- **Quality flags**: booleans for `is_default_odds` (only 1.73/1.73), `is_sparse` (<=2 pivots), `is_dense` (>=5 pivots), `has_asymmetry` (|p_over-0.5| > eps near central).

Optional richer descriptors:
- Area under fitted curve between min/max pivots (confidence summary).
- Entropy of implied distribution at pivot grid.
- Interquartile width from fitted curve (Q75-Q25 of total points).

## Representation options (pros / cons)
- **Logistic/probit fit (curve descriptor)**: smooth, low-dim, robust to noise; needs slope prior when sparse; easy scalar features for trees; slight bias if market shape deviates from sigmoid.
- **Parametric totals dist (normal / skew-normal / t)**: yields mean/variance/skew directly; compact; can over-assume symmetry; fitting skew can be unstable on sparse pivots.
- **Monotone spline / isotonic interpolation**: non-parametric, respects monotonicity; handles weird shapes; may overfit tiny k; need smoothing for dense noisy curves.
- **Fixed pivot grid + kernel/NN smoothing**: produces fixed-length vector even when sparse; great for tree/GBM/CatBoost; requires choosing grid + bandwidth; slight blur of sharp edges.
- **Raw pivot stats only (count/range/center)**: trivial to compute; loses shape info; weak for dense modern data.

## Recommendation for current stack (xgb, lgbm, catboost)
- Use a **hybrid**: (a) scalar curve descriptors from the logistic/probit fit (central label, slope, vig stats, range, asymmetry, entropy/IQR) and (b) a **fixed-grid embedding** of `p_over` on a small pivot grid (e.g., central prior ±25 in 1–2 pt steps) filled via kernel smoothing or nearest neighbor with a missing flag per cell.
- Tree/GBM models digest both sparse scalars and medium-length vectors well; CatBoost handles categorical flags cleanly.
- Keep **flags**: `handicap_missing` (no JSON), `handicap_is_default_odds`, `handicap_is_sparse`, `handicap_is_dense`. Trees leverage these to route missing odds instead of forcing defaults.
- Leave numeric features as NaN when truly unavailable and pair with flags; for the grid, either NaN + flag per cell or backfill with prior probability 0.5 plus a cell-level missing mask.

## Fallbacks by data case
- **Default odds (case 1)**: set `p_over = 0.5` for all pivots, inject prior slope (e.g., logistic scale s ≈ 8–10 points -> slope ~0.125 at mid). Central line = mean of available labels. Mark `is_default_odds`.
- **Few pivots with near-fair odds (case 2 / 2 bis)**: fit logistic with weak prior on slope; use observed asymmetry if present. Central line ~ pivot where odds closest to fair.
- **Full curve (case 3)**: fit normally; rely on slope/asymmetry/overround spread; compute coverage metrics.

## Feature bundle to emit per match
- Scalars: `handicap_central_label`, `handicap_slope_at_central`, `handicap_range`, `handicap_pivot_count`, `handicap_avg_spacing`, `handicap_avg_vig`, `handicap_max_vig`, `handicap_entropy`, `handicap_iqr_width`.
- Local odds snippets: `handicap_over_p0`, `handicap_over_p1`, ... for first K pivots nearest central (store corresponding labels too, e.g., `handicap_label_p0`).
- Flags: `handicap_is_default_odds`, `handicap_is_sparse`, `handicap_is_dense`, `handicap_has_asymmetry`.
- Provenance: `handicap_source` (oddsportal scrape vs screenshot OCR), `handicap_pivots_min`, `handicap_pivots_max`.

## Handling missing odds entirely
- If `handicap` is null/empty: set `handicap_missing = 1`, emit NaN for numeric features (or prior defaults in the grid), keep other match features intact. Do not invent a central pivot.
- If JSON is present but unparsable: same handling plus a `handicap_parse_error` flag to debug data quality.

## Notes on slope prior
- Sparse/default cases need a slope prior so the fitted curve is not flat. Use a logistic scale `s ≈ 9` points (gives slope ~0.14 at p=0.5). For 1–2 pivots, fix slope to this prior; for >=3 pivots, fit freely with mild regularization.

## Model-facing feature suggestions
- Scalars (from curve fit): `handicap_central_label`, `handicap_slope_at_central`, `handicap_avg_vig`, `handicap_max_vig`, `handicap_entropy`, `handicap_iqr_width`, `handicap_range`, `handicap_pivot_count`, `handicap_avg_spacing`, `handicap_asymmetry_near_central`.
- Grid embedding (absolute): labels 170–260 inclusive with step 1, named `handicap_over_170_0`, `handicap_over_171_0`, …, `handicap_over_260_0` (underscore for decimal). Missing flag per cell: `handicap_over_<label>_missing` (1 = missing). NaN for missing values, no default 0.5.
- Grid embedding (relative): same step (1) on a prior-centered grid (e.g., prior total ±25). Names `handicap_over_rel_<label>` with matching `_missing` flags. Prior can be season/global expected total.
- Flags: `handicap_missing`, `handicap_parse_error`, `handicap_is_default_odds`, `handicap_is_sparse`, `handicap_is_dense`, `handicap_has_asymmetry`.

## Suggested pipeline hook
- Add a transformer in `src/datasets/bronze.py` (or a new helper) that:
  - Parses the JSON string to ordered pivots.
  - Computes probabilities, overround, fit, and the feature bundle above.
  - Gracefully handles null/empty by emitting NaNs + flags.
- Keep the raw `handicap` JSON in bronze/silver; store derived features in model-ready datasets via `assemble_match_dataset`.
- Unit-test with the three provided cases plus: empty JSON, single pivot asymmetric, dense noisy curve.

## Inference via screenshots
- OCR the pivot/odds grid, run the same transformer to get curve features.
- Models can then consume the same feature schema regardless of source (historical scrape vs live screenshot).

## MLflow visualization
- New plot (logged per training): sample 10 random test games and overlay (a) the market/handicap over-prob curve from grid features, (b) the model-implied over-prob curve from the predicted mean/sigma, and (c) a vertical line for the actual total (plus the predicted mean marker). Logged under `plots/{model}_handicap_curve_<idx>.png`.
- Expected inputs: fixed-grid features named `handicap_over_<label>` (labels sanitized with `_` for decimals, e.g., `handicap_over_195_5`) with optional missing flags `handicap_over_<label>_missing`. When absent or all NaN the plot is skipped gracefully.
- Curve reconstruction: use grid labels/probs as-is (clipped 0–1), sorted by label; model curve uses `1 - norm.cdf(pivot)` with `sigma` from the sigma model (floored at `min_sigma`).
- Benefits: quick qualitative check that the learned distribution aligns with bookmaker shape and actual outcome, across sparse vs dense handicap info.

## Interpolation strategy for grid fill
- Default: Gaussian smoothing of known pivots onto the grid, with bandwidth `sigma_grid=1.5` points (configurable). Weight for a grid label `g` from pivot `p`: `w = exp(-(g-p)^2 / (2*sigma_grid^2))`, truncated when `|g-p| > 4`. Normalize weights per `g`, take weighted mean of available `p_over`. If no pivot contributes, leave NaN and `_missing=1`.
- Optional monotonicity: after smoothing, apply isotonic regression decreasing in the pivot to enforce P(Over) monotone. Can be toggled if needed.
- Default: leave isotone **off** to keep raw smoothed shape; enable via flag if you want monotonic enforcement.
- Single-pivot edge: same smoothing works; will create a bump around that pivot; flags still mark missing cells.
- Config: expose `sigma_grid` and a flag to disable smoothing (keep NaNs) via CLI/config; plan to allow an “apply optimized sigma” flag for reruns.
- Future optimization: after a first run, export dense curves, hold out some pivots, and grid-search `sigma_grid` to minimize reconstruction error; reuse the best sigma in subsequent feature-generation runs without recalculating each time.
- Storage: defaults live in `src/config.py` (env-overridable) so reruns can flip `HANDICAP_APPLY_OPTIMIZED_SIGMA` and the chosen `HANDICAP_SIGMA_GRID_OPTIMIZED`.
- Feature pruning: a generic pruning step drops columns with extreme missingness or zero variance (configurable in `src/config.py`). This will trim unused grid columns (e.g., high pivots rarely present) before modeling.

## Relative grid prior
- Use the current-season mean of `POINT_TOTAL` (computed from the dataset slice being processed) as the prior center. If missing, fall back to the global mean.

## Top-K local odds
- K = 3. Sort pivots by distance to `handicap_central_label`. Emit `handicap_over_p0/p1/p2`, `handicap_label_p0/p1/p2`, with `_missing` flags if not enough pivots; leave NaN when absent.

## Config / CLI knobs
- Expose: `sigma_grid` (default 1.5), `handicap_enable_smoothing` (default True), `handicap_apply_isotone` (default False), `handicap_apply_optimized_sigma` (default False; when True, load a stored sigma from a config/MLflow param rather than re-optimizing).

## Next steps
- Implement the transformer + tests.
- Backfill features on historical datasets; inspect distributions (central label, slope) to set reasonable priors.
- Add a data-quality report to spot default-odds rows and sparse coverage before training.
