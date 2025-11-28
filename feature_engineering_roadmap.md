# Feature Engineering Roadmap (Point Total)

## Objectifs
- Mieux factoriser « Points = Possessions × Points par possession » via des features explicites.
- Réduire le bruit (4.7k features, seulement 14% d’importance cumulée sur le top 100).
- Mettre sous contrôle les signaux absences/rotations. (Pas d’odds/market data : ignoré totalement.)

## W1: Possessions & Meta-modèle de total attendu — _implémenté (code), script d'entraînement ajouté, lancer le fit_
- `src/features/matchup.py`: ajouter `MATCH_POSSESSIONS_EXPECTED_{n}` = f(MATCH_PACE_{n}, MATCH_EXTRA_POSSESSIONS_{n}, REST_ADVANTAGE, IS_BACK_TO_BACK/3IN4/5IN7).
- `src/features/matchup.py`: `EXPECTED_TOTAL_FROM_RATINGS_{n}` = (PPP_home + PPP_away) × MATCH_POSSESSIONS_EXPECTED, où PPP_home = HOME_ROLL_offensiveRating_advanced_{n} * (1 / AWAY_ROLL_defensiveRating_advanced_{n}) * 100 (ou ratio off/def).
- Calibration: petit modèle linéaire/ElasticNet entraîné offline sur (pace, extra possessions, rest flags) → `META_POSSESSIONS_PRED` et `META_TOTAL_PRED`; stocker coefficients pour inference (JSON `artifacts/meta_models/meta_possessions_v1.json` + tag mlflow). **Script ajouté**: `python -m src.scripts.train_meta_possessions` (dépose le JSON). Relancer la pipeline après fit.
- Intégration pipeline: step `meta_features` ajouté dans `FeaturePlan` juste après `matchup_scoring` (train/predict). Fallback statique si le JSON manque.
- Eval: RMSE possessions et point_total (train/val), importance attendue dans le top 20; journaliser la version du meta-modèle.

## W2: Profil de tir & PPP décomposé — _implémenté (EFG/FT edges), z-score à étendre si besoin_
- `matchup.py` via `meta_features`: `EXPECTED_EFG_EDGE_{n}` et `EXPECTED_FT_POINTS_EDGE_{n}` ajoutés.
- `normalization.py`: z-score saisonnier à ajouter si besoin (non modifié pour l’instant).
- Consignes techniques: `MATCHUP_WINDOWS` seulement; prefixes `EXPECTED_` whitelisted.

## W3: Variance & game state — _implémenté (variance score/flag + blowout)_
- `meta_features`: `MATCH_VARIANCE_SCORE_{n}` + `MATCH_HIGH_VARIANCE_FLAG_{n}` (p75 interne ou thresholds JSON si fourni) + `MATCH_BLOWOUT_RISK_{n}` (ELO gap vs variance, cap [0,1]).
- À faire: whitelisting/suppression des anciens flags clutch via env/whitelist si bruit.

## W4: Disponibilité/rotation → impact en points — _implémenté (gap + ratio), affiner sources absents si besoin_
- `meta_features`: `MATCH_MISSING_PTS_GAP_{n}` + ratio si `TOTAL_POINTS_EXPECTED_{n}` dispo (basé sur `player_perf_score_sum` × `top_player_absent_rate`).
- Bench: inchangé (garder `MATCH_BENCH_PACE_ADJ_{n}`), `BENCH_NET_RATING_GAP` reste à faire si données bench.

## W5: H2H allégé — _implémenté (H2H_REDUCED=1 par défaut)_
- `config.py`: `H2H_REDUCED` (default 1) → fenêtres [5,10,25] utilisées dans `team.py`.
- Option poids/filtrage reste à faire si besoin.

## (Retiré) Marché
- Pas de données odds → ignorer toute feature bookmaker (`IMPLIED_*`) et ne pas toucher `bronze.py` pour les spreads/totals.

## W7: Normalisation/feature selection — _whitelist optionnelle ajoutée, shortlist auto dispo_ 
- `cleanup.py`: support `FEATURE_WHITELIST=<name>` lisant `config/feature_whitelists/<name>.txt` (ex: `point_total_v1.txt`).
- Script: `python -m src.scripts.build_whitelist_from_importance --importance-csv <csv>` génère `config/feature_whitelists/point_total_short.txt` (top-N + META/EXPECTED forcés). Shortlist générée depuis l’importance XGB récente.
- Impl: si `FEATURE_WHITELIST` est défini, seuls les IDs + entrées/prefixes de la whitelist sont conservés (plus d’effet no-op).
- À faire: comparer runs avec `FEATURE_WHITELIST=point_total_short` (ablation), logguer la version dans mlflow.

## Validation & suivi
- Metrics: RMSE/MAE point_total; R²; calibration (Brier si classification over/under).
- Importance check: viser >30% d’importance cumulée sur top 100 et présence des nouvelles features (meta possession/total, EFG/FT edges, variance_score) dans top 30.
- Logging: consigner versions dans mlflow (params: feature_whitelist_version, meta_model_version).

## Pipeline & modèles — _étendu (step meta_features + ElasticNet)_
- Commande de génération : `python -m src.feast.pipeline --seasons 2025-26 --targets POINT_TOTAL --materialize` (doit fonctionner même sans JSON du méta-modèle grâce au fallback statique).
- `FeaturePlan`: step `meta_features` ajouté (pipeline train/predict). `META_POSSESSIONS_PRED`, `META_TOTAL_PRED`, `EXPECTED_TOTAL_FROM_RATINGS_*`, `EXPECTED_EFG_EDGE_*`, `EXPECTED_FT_POINTS_EDGE_*`, variance, blowout et missing points générés avec fallback si JSON absent.
- Modèles: LGBM/XGB/xgb_calibrated conservés, ElasticNet ajouté aux modèles dispo par défaut; stacking inchangé.
