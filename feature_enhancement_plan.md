# Feature Enhancement Plan (Tempo & Context)

## Contexte
- Les features dominantes dans `feature_importance/tmpae1ffqv6.csv` proviennent quasi exclusivement de `MATCH_PACE_*`, `TOTAL_POINTS_EXPECTED_*` et des rollups de pace/points (OPP pace 25/50/100/200, etc.).
- Les colonnes d’absences (`top_player_absent_*`), `DIFF_*` ou même `PACE_DIFF_*` n’apportent quasiment rien (importance = 0).
- Les stratégies safe reposent sur ces probas → même petites erreurs de tempo biaisent les picks (trop de votes pour les cotes 1.15–1.25).

## Objectifs
1. Raffiner les signaux de tempo/scoring pour qu’ils reflètent mieux le contexte du match.
2. Réactiver l’information d’absences / disponibilité via un score pondéré.
3. Ajouter des interactions ciblées (delta home vs away, pivot bookmaker) pour réduire la variance résiduelle du modèle.

## Idées de Features Prioritaires

### 1. Tempo & Total Points
- **Seasonal Pace Index**: rapport entre `MATCH_PACE_x` et la moyenne pace NBA de la saison (pour normaliser les fenêtres longues).
- **Pace Trend**: différence `MATCH_PACE_10 - MATCH_PACE_50` pour détecter les équipes qui accélèrent récemment.
- **Expected vs Line Gap**: `TOTAL_POINTS_EXPECTED_x - bookmaker_pivot` (à extraire du payload OCR / JSON) pour intégrer l’information live.
- **Weighted combo**: `0.7 * MATCH_PACE_10 + 0.3 * MATCH_PACE_50` ou un simple ratio court/long (feature `PACE_RATIO_10_50`).

### 2. Interactions Home/Away
- **Off vs Def Ratings**: `HOME_ROLL_offensiveRating_10 - AWAY_ROLL_defensiveRating_10` et réciproque.
- **Pace diff redéfini**: `HOME pace trend - AWAY pace trend` normalisé par la moyenne NBA.
- **H2H delta**: plutôt que deux colonnes `HOME_H2H_LAST_5` et `AWAY...`, créer `H2H_TOTAL_POINTS_DELTA` (home - away) + `H2H_AVG_TOTAL_POINTS`.

### 3. Disponibilité / Absent Impact
- Remplacer complètement la logique existante (actuelle = compteurs `top_player_absent_*` peu fiables). Nouveau plan :
  1. Repartir des données bronze (boxscores + `build_player_status_features`) pour identifier les absents et leur poids (minutes moyennes ou score EPM si dispo).
  2. Calculer un **Player Impact Score** par match = somme pondérée des joueurs absents côté HOME/AWAY.
  3. Créer les features `HOME_AVAILABILITY_IMPACT`, `AWAY_AVAILABILITY_IMPACT`, puis `AVAILABILITY_IMPACT_DIFF`.
  4. Supprimer / déprécier les colonnes `top_player_*` inutilisées.
- Cela implique de réécrire les steps availability dans `src/features/availability.py` pour garantir un pipeline clean et leak-free.

### 4. Contexte bookmaker
- **Pivot Alignment**: `MODEL_PRED_TOTAL - bookmaker_pivot` (calculé en prediction mais stockable en feature pour calibrations futures).
- **Odds Gap**: si l’API OCR fournit les cotes, les transformer en implied probability et les comparer à nos propres probas (feature à stocker côté strat si besoin).

## Implémentation (proposée)
1. Ajouter un module `src/features/tempo.py` (ou compléter `matchup.py`) pour générer les nouvelles colonnes `PACE_RATIO_*`, `PACE_TREND`, `SEASONAL_PACE_INDEX`.
2. Mettre à jour les steps silver/Feast pour inclure ces features (via `DEFAULT_FEATURE_PLAN`).
3. Ajouter un script/utilité pour calculer l’`AVAILABILITY_IMPACT` lors de la construction bronze → silver.
4. Recalculer un dataset silver + réentraîner le modèle pour vérifier l’importance des nouvelles colonnes.

## Étapes suivantes
1. Implémenter les features tempo/trend + seasonal index.
2. **Réécrire le module availability** pour calculer l’impact pondéré des absences (pipeline bronze → silver → Feast).
3. Relancer `make model-point-total` (ou tuning léger) et vérifier l’importance + RMSE.
4. Mettre à jour la calibration/stratégie si les probas s’améliorent.
