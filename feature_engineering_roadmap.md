# Feature Engineering Roadmap (Point Total)

## Objectifs
- Mieux factoriser « Points = Possessions × Points par possession » via des features explicites.
- Réduire le bruit (4.7k features, seulement 14% d’importance cumulée sur le top 100).
- Mettre sous contrôle les signaux absences/rotations et le contexte marché (si données odds réactivées).

## W1: Possessions & Meta-modèle de total attendu
- `src/features/matchup.py`: ajouter `MATCH_POSSESSIONS_EXPECTED_{n}` = f(MATCH_PACE_{n}, MATCH_EXTRA_POSSESSIONS_{n}, REST_ADVANTAGE, IS_BACK_TO_BACK/3IN4/5IN7).
- `src/features/matchup.py`: `EXPECTED_TOTAL_FROM_RATINGS_{n}` = (PPP_home + PPP_away) × MATCH_POSSESSIONS_EXPECTED, où PPP_home = HOME_ROLL_offensiveRating_advanced_{n} * (1 / AWAY_ROLL_defensiveRating_advanced_{n}) * 100 (ou ratio off/def).
- Calibration: petit modèle linéaire/ElasticNet entraîné offline sur (pace, extra possessions, rest flags) → `META_POSSESSIONS_PRED` et `META_TOTAL_PRED`; stocker coefficients pour inference (fichier JSON/YAML + tag mlflow). Protéger les colonnes absentes avec des guards (`if col in df`), aligner sur `MATCHUP_WINDOWS`.
- Intégration pipeline: pas de nouveau step dans `FeaturePlan`, ajouter dans `add_matchup_scoring_features` pour que training & predict partagent la même logique; conserver les noms en `MATCH_` pour que `cleanup.enforce_feature_whitelist` les garde.
- Eval: RMSE possessions et point_total (train/val), importance attendue dans le top 20; journaliser la version du meta-modèle.

## W2: Profil de tir & PPP décomposé
- `matchup.py`: `EXPECTED_EFG_EDGE_{n}` = EFG_off − EFG_concede (utiliser percentages 3PA, 2PA, 3P%, FT% + OPP_*).
- `matchup.py`: `EXPECTED_FT_POINTS_{n}` = FTr * FT% (off) vs FTr concédé; `FT_EDGE_{n}`.
- `normalization.py`: z-score saisonnier pour `MATCH_PACE_{n}`, `TOTAL_POINTS_EXPECTED_{n}`, `MATCH_EXTRA_POSSESSIONS_{n}`, `EXPECTED_EFG_EDGE_{n}`, `FT_EDGE_{n}`. Ajouter une garde sur `SEASON` (présent dans les jeux train/predict).
- Consignes techniques: rester dans `MATCHUP_WINDOWS`; réutiliser le pattern existant (`if all(col in df.columns for col in ...)`), pas d’override des colonnes existantes; penser à compléter `MATCH_ALLOWED_PREFIXES` si de nouveaux préfixes apparaissent.

## W3: Variance & game state
- `team.py`: déjà `*_STD`; condenser en `MATCH_VARIANCE_SCORE_{n} = MATCH_POINTS_VARIANCE_{n} + MATCH_3PT_VARIANCE_{n}` et `HIGH_VARIANCE_FLAG_{n}` (seuil percentile).
- `matchup.py`: `MATCH_BLOWOUT_RISK_{n}` = g(ELO gap, variance) mais supprimer/abaisser les flags CLUTCH inutiles; reposer sur seuils continus.
- Eval: vérifier que variance_score entre dans le top 50 importance; sinon simplifier ou dropper. Nettoyer les anciens flags CLUTCH/IMPORTANCE si 0 importance (via whitelist dynamique).
- Implémentation: réutiliser `MATCH_ELO_GAP`/`MATCH_ELO_GAP_SEASON` si présents; fallback à 0 pour éviter NaN en prod.

## W4: Disponibilité/rotation → impact en points
- `rotation.py` ou module dédié: calculer `MISSING_PTS_HOME/ AWAY` = somme des `player_perf_score` des absents (par catégorie) ramenée aux possessions attendues → `MISSING_PPP_*`.
- `matchup.py`: `MATCH_MISSING_PTS_GAP_{n}` et ratio vs `TOTAL_POINTS_EXPECTED_{n}`.
- Bench: `BENCH_NET_RATING_GAP_{n}` si net rating bench dispo; sinon garder `MATCH_BENCH_PACE_ADJ_{n}`.
- Consignes techniques: calculer dans la vue équipe (`match_rows_to_team_rows`), puis projeter via `attach_team_features` pour rester dans le flux `FeaturePlan`; protéger les absents en assignant 0 si colonnes manquantes; ne pas dropper par `drop_raw_team_columns`.

## W5: H2H allégé
- Garder seulement `H2H_LAST_{10,25}_PTS_AGAINST` et `H2H_LAST_{5}_AVG_TOTAL_POINTS`; dropper le reste (counts/winrates peu informatifs).
- Option: poids décroissant par nb d’échantillons pour limiter bruit sur petits n.
- Consignes techniques: soit limiter la génération dans `compute_h2h` (ajout d’un param), soit filtrer en sortie via un whitelist dédiée avant `cleanup`; vérifier que `FeaturePlan` reste inchangé pour rétro-compat des notebooks d’entraînement/pred.

## W6: Marché (si odds réactivées)
- `src/datasets/bronze.py`: réintégrer spreads/totals (`HOME_SPREAD`, `AWAY_SPREAD`, `TOTAL_POINTS`).
- `matchup.py`: `IMPLIED_TOTAL`, `IMPLIED_SPREAD`, `MARKET_RESIDUAL_TOTAL = IMPLIED_TOTAL − META_TOTAL_PRED`.
- Consignes techniques: sécuriser la fusion pour éviter les fuites (uniquement infos pré-match); ajouter les préfixes `IMPLIED_` à `MATCH_ALLOWED_PREFIXES`; tester la présence des colonnes avant calcul; stocker la version de la source odds (tag mlflow).

## W7: Normalisation/feature selection
- `cleanup.py`: whitelist dynamique/paramétrable; au minimum dropper les 235 features à 0 importance + familles CLUTCH/MATCH_IS_*.
- Entraînement: ablations par bloc (H2H, context, ELO, availability) pour mesurer ΔRMSE/MAE; retenir top-k par mutual information/gain avant fit.
- Consignes techniques: exposer un paramètre (env/CLI) pour activer la whitelist courte; logguer la liste des features retenues dans mlflow; garder la compatibilité avec les notebooks de prédiction (`10_predict_point_total.ipynb`), donc versionner la whitelist.

## Validation & suivi
- Metrics: RMSE/MAE point_total; R²; calibration (Brier si classification over/under).
- Importance check: viser >30% d’importance cumulée sur top 100 et présence des nouvelles features (meta possession/total, EFG/FT edges, variance_score) dans top 30.
- Logging: consigner versions dans mlflow (params: feature_whitelist_version, meta_model_version).
