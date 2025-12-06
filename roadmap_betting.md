# Betting Simulation Roadmap

## Objectives
- Evaluate trained models on truly out-of-sample (last walk-forward fold) with bookmaker totals/odds to assess value.
- Simulate betting strategies day-by-day using predicted probabilities vs market implied probabilities.
- Compare staking schemes: flat, Kelly (full), fractional Kelly (configurable, e.g., 2/3).
- Handle multi-bet days: place bets for all games of the same day; if total Kelly stake > 100%, normalize stakes proportionally.
- Persist results (metrics + per-bet logs) locally and log summaries/plots to MLflow if run in training context.

## Data & inputs
- Use the last walk-forward fold’s test split as the holdout set (matching the training config).
- For each test row, require handicap grid/odds: pick a target pivot (e.g., closest to market central or a chosen grid label) and extract `odds_over/odds_under` (vig-removed implied probs).
- Model prob: `P_over` from model distribution (1 - CDF at pivot) or direct prediction if available.
- Edge: `edge = P_over * odds_over - (1 - P_over)` (EV per 1u stake), similarly for under.

## Simulation strategies
- **Flat stake**: stake = flat_fraction of bankroll (default 1%) on selections where edge > threshold (configurable).
- **Kelly full**: stake = kelly_fraction * ( (P* (odds-1) - (1-P)) / (odds-1) ), clipped to [0, kelly_cap], with `kelly_fraction`=1.0.
- **Fractional Kelly**: same, but `kelly_fraction` configurable (e.g., 0.66).
- **Daily normalization**: group by GAME_DATE; if sum(stake) > daily_stake_cap (default off, otherwise % bankroll), scale stakes proportionally so total stake <= cap.
- Thresholds: configurable edge threshold (e.g., >0.01) to trigger a bet.

## Outputs/metrics
- Per-bet log: game_id, date, pivot used, odds_over/under, P_model_over/under, stake, win/loss, PnL.
- Aggregate metrics: total bets, hit rate, ROI, avg stake, max drawdown, per-season ROI.
- Plots: cumulative PnL over time (by date), ROI by edge bucket, stake distribution.
- MLflow logging (optional): metrics + plots + CSV; log params (kelly_fraction, edge_threshold, normalization rule).

## Reusable code components
- `load_holdout_split(training_cfg, dataset_cfg)` -> X_test, y_test, df_meta (to get GAME_DATE/SEASON).
- `extract_market_line(row)` -> pivot, odds_over, odds_under, implied probs (vig-removed).
- `model_prob_over(pred_mean, pred_sigma, pivot)` -> P_over.
- `compute_edge(P_over, odds_over)` -> EV metrics.
- `simulate_day(stakes_df, strategy_cfg)` -> apply Kelly/flat, normalize per day.
- `run_simulation(test_df, model_preds, odds_fields, strategy_cfg)` -> per-bet log + aggregates.

## Open questions to finalize
- Pivot choice: use market central line (nearest to p=0.5) or a fixed grid label? (Default: market central from handicap features.)
- Edge threshold defaults? (Start with 0.01/0.02.)
- Bankroll cap per day (1u or configurable percent)?
- Handling sparse odds: skip days without odds or use default fair odds?

## Updated decisions
- Pivot: market central line (vig-removed over/under closest to 0.5) as primary; allow 2 bets (over et under) si chacun dépasse le seuil d’edge.
- Edge threshold: default 0.01, configurable.
- Staking: flat as % bankroll (default 1%); Kelly fractionnel/capé; daily stake cap param (default off) avec normalisation par jour si dépassé.
- Skip matches sans odds. Utiliser probas calibrées au pivot si dispo; fallback CDF avec sigma (calibrée si présente, sinon brute).
- Simulation dans un notebook (pas dans la pipeline MLflow pour l’instant); logging local (CSV/plots), MLflow optionnel plus tard.
