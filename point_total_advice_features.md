Grosso modo, pour prédire combien de points vont être marqués dans un match (total ou par équipe), tu peux tout ramener à **3 grands blocs** :

1. **Volume de possessions**
2. **Qualité de l’attaque**
3. **Qualité (et style) de la défense**

Je te détaille ça avec les trucs qui, en pratique, pèsent le plus dans les modèles.

---

## 1. Volume de possessions (le nerf de la guerre)

Plus il y a de possessions, plus il y a de tirs, donc plus de points possibles.

Les facteurs clés :

* **Pace (rythme de jeu de l’équipe)**

  * Équipes qui courent en transition, jouent vite après rebond défensif, remettent en jeu rapidement.
  * Exemples typiques : équipes “run & gun”, beaucoup de drives, peu de système long.

* **Tendance à monter la balle vite / utiliser le chrono**

  * % de possessions qui tirent dans les 8–10 premières secondes.
  * À l’inverse, les équipes qui posent beaucoup de systèmes, milkent l’horloge, vont naturellement “baisser” le total de points même si elles sont efficaces.

* **Turnovers et rebonds offensifs**

  * **Peu de turnovers** = plus de possessions qui se terminent par un tir (et donc potentiellement des points).
  * **Beaucoup de rebonds offensifs** = 2ème chances → augmentation de possessions effectives.

* **Matchup de styles**

  * Deux équipes rapides → match souvent très haut en possessions.
  * Une équipe très lente qui impose son tempo → même une équipe rapide peut se retrouver dans un match “under”.

Pour un modèle, ça se traduit par :

> *Pace équipe A, Pace équipe B, possessions moyennes, % early offense, TO%, ORB% (offensive rebounding %), etc.*

### État actuel (NBA_Predictor)

- `src/features/team.py` produit pour chaque équipe des rollings `ROLL_pace_advanced_{3,5,10,25,50,100,200}` et `ROLL_possessions_advanced_{n}` (en EWM) qui sont ensuite projetés en `HOME_/AWAY_` puis combinés dans `src/features/matchup.py` en `MATCH_PACE_{n}` et `PACE_DIFF_{n}`.
- Les volumes additionnels sont couverts par `ROLL_turnovers_traditional_{n}`, `ROLL_percentageTurnovers_usage_{n}`, `ROLL_reboundsOffensive_traditional_{n}`, `ROLL_percentageReboundsOffensive_usage_{n}` et leurs versions `OPP_`, plus les compteurs de seconde chance / transition (`ROLL_pointsOffTurnovers_misc_{n}`, `ROLL_pointsSecondChance_misc_{n}`, `ROLL_pointsFastBreak_misc_{n}`).
- Les effets calendrier sont représentés par `DAYS_SINCE_LAST_GAME`, `REST_ADVANTAGE`, `ROLL_AWAY/HOME_REST_ADV_{n}` et les différences `DIFF_*` automatiquement générées.

### Gaps & plan concret

1. **Quantifier explicitement les possessions bonus**  
   Étendre `src/features/matchup.py::add_matchup_scoring_features` avec un helper `_add_possession_pressure_features` qui exploite `HOME/ AWAY_ROLL_percentageReboundsOffensive_usage_{n}`, `ROLL_percentageTurnovers_usage_{n}` et `ROLL_pointsFastBreak_misc_{n}` pour produire `MATCH_ORB_EDGE_{n}`, `MATCH_TOV_EDGE_{n}` et `MATCH_EXTRA_POSSESSIONS_{n}` (ORB edge − TOV edge + fastbreak proxy). Ces features seront ajoutées aux `MATCH_ALLOWED_PREFIXES`.
2. **Flags calendrier lisibles par le modèle**  
   Dans `src/features/team.py::_add_rest_features`, dériver `IS_BACK_TO_BACK`, `IS_3IN4`, `IS_5IN7` ainsi que leurs versions différentielles (`MATCH_B2B_DIFF`). Concrètement, ajouter des colonnes booléennes basées sur `DAYS_SINCE_LAST_GAME` puis calculer des rollings courts pour capter la fatigue accumulée.
3. **Approximation d’early offense**  
   Exploiter `ROLL_pointsFastBreak_misc_{n}` et `ROLL_possessions_advanced_{n}` pour créer `MATCH_FASTBREAK_RATE_{n}` et `MATCH_FASTBREAK_DIFF_{n}` dans `matchup.py`. Cela servira de proxy au % de possessions lancées dans les 8–10 premières secondes.

---

## 2. Qualité offensive (efficacité par possession)

Une fois le volume de possessions estimé, il faut savoir **combien de points elles génèrent**.

Les caractéristiques importantes :

### a) Profil de tir

* **Répartition 3pts / 2pts / lancer-francs**

  * 3PA rate (volume de 3 points tentés)
  * Rim vs mid-range vs 3pts
  * Free throw rate (FTr = FTA / FGA)

* **Efficacité sur ces zones**

  * %3PT, % at rim, % mid-range
  * EFG% / TS% (plus stables que le simple FG%)

Une équipe qui :

* prend beaucoup de 3,
* en met à un % correct,
* provoque des fautes

va naturellement générer plus de points par possession que :

* une équipe mid-range lourde
* qui ne provoque pas beaucoup de fautes.

### b) Création de tirs (playmaking)

* **Quality of looks**

  * % de shoots assistés
  * Potentiel de création individuelle (isolation, PnR ball handler, etc.)
* **Turnover rate**

  * Perdre la balle = 0 point sur la possession + souvent transition adverse.

Les joueurs clés :

* Un **créateur élite** (Luka, SGA, etc.) augmente énormément la qualité de l’attaque, surtout en fin de quart/fin de match.
* Un **big** qui finit bien au cercle (big roller, lob threat) améliore beaucoup l’efficience du PnR.

### État actuel (NBA_Predictor)

- Les rollings de scoring et d’efficacité (`ROLL_points_traditional_{n}`, `ROLL_offensiveRating_advanced_{n}`, `ROLL_trueShootingPercentage_advanced_{n}`, `ROLL_effectiveFieldGoalPercentage_advanced_{n}`, `ROLL_freeThrowAttemptRate_fourfactors_{n}`) sont calculés pour chaque équipe puis disponibles sous forme `HOME_/AWAY_/DIFF_`.
- La répartition des tirs est couverte par les colonnes `percentageFieldGoalsAttempted2pt/3pt_scoring`, `percentagePoints3pt/2pt/midrange/paint/free_throw_scoring`, `percentagePointsFastBreak_scoring`, etc., ainsi que leurs variantes `OPP_`, ce qui donne une vision offensive + la vision de ce que la défense concède.
- La création de tirs est représentée via `ROLL_assistPercentage_advanced_{n}`, `ROLL_assistRatio_advanced_{n}`, `ROLL_assistToTurnover_advanced_{n}`, `ROLL_percentageAssists_usage_{n}`, les métriques de turnover (`ROLL_turnoverRatio_advanced_{n}`, `ROLL_percentageTurnovers_usage_{n}`) et les features générées automatiquement `DIFF_*`.

### Gaps & plan concret

1. **Mettre en regard profil offensif vs profil concédé**  
   Ajouter dans `src/features/matchup.py` un helper `_add_shot_profile_gaps` qui combinera `HOME_ROLL_percentageFieldGoalsAttempted3pt_scoring_{n}` avec `AWAY_ROLL_OPP_percentageFieldGoalsAttempted3pt_scoring_{n}` (et réciproquement) pour générer `MATCH_3PT_PROFILE_GAP_{n}`, `MATCH_RIM_PROFILE_GAP_{n}`, `MATCH_MIDRANGE_PROFILE_GAP_{n}` et `MATCH_PACE_LOCATION_RISK_{n}`. Cela explicitera l’alignement style offensif vs faiblesses défensives.
2. **Pression vers la ligne des lancers-francs**  
   Utiliser `ROLL_freeThrowAttemptRate_fourfactors_{n}` et `ROLL_OPP_percentagePersonalFouls_usage_{n}` pour créer `MATCH_FT_PRESSURE_{n}` et `MATCH_FOUL_PRONE_{n}` (toujours dans `matchup.py`). Objectif : quantifier les matchs susceptibles de basculer sur des lancers.
3. **Playmaking gap**  
   Générer `MATCH_CREATION_GAP_{n}` = `HOME_ROLL_assistPercentage_{n}` − `AWAY_ROLL_OPP_percentageAssists_usage_{n}` (et inverse) + un `MATCH_TOV_PRESSURE_{n}` combinant `assistToTurnover` et `OPP_teamTurnoverPercentage_fourfactors_{n}`. Implémentation : nouveau helper dans `matchup.py` + ajout des colonnes à la whitelist.

> ✅ Implémenté : `_add_shot_profile_gaps`, `_add_ft_pressure_features` et `_add_playmaking_features` ont été ajoutés à `src/features/matchup.py` (fenêtres MATCHUP_WINDOWS), générant `MATCH_3PT_PROFILE_GAP`, `MATCH_RIM_PROFILE_GAP`, `MATCH_LOCATION_RISK`, `MATCH_FT_PRESSURE`, `MATCH_FOUL_PRONE`, `MATCH_CREATION_GAP` et `MATCH_TOV_PRESSURE`.

---

## 3. Qualité et style défensifs (ce qui freine ou accélère la sauce)

La défense ne fait pas qu’“empêcher de marquer”, elle **change la forme du match**.

### a) Niveau défensif brut

* **Defensive Rating (points encaissés / 100 possessions)**
* EFG% concédé
* %3PT concédé, protection du cercle (% at rim concédé)

Plus une défense est bonne, plus le total attendu sera bas, mais ce n’est pas si simple :

### b) Style défensif (qui influence le rythme et le profil de tir adverse)

* **Équipes qui blitzent les PnR, trappent, surjouent les lignes de passe**
  → plus de turnovers, plus de transitions → parfois *plus* de points dans le match (surtout si l’adversaire court bien aussi).

* **Équipes qui “pack the paint” et laissent le 3 ouvert**
  → opposants prennent plus de 3, variance plus grande sur le total (feu vert ou brique party).

* **Gestion des fautes**

  * Défense très agressive qui fait beaucoup de fautes → plus de lancers francs → total de points qui monte, surtout sur un match serré.

### État actuel (NBA_Predictor)

- Grâce aux rollings `ROLL_OPP_*`, nous avons déjà une vision de la défense : `ROLL_OPP_defensiveRating_advanced_{n}`, `ROLL_OPP_effectiveFieldGoalPercentage_advanced_{n}`, `ROLL_OPP_trueShootingPercentage_advanced_{n}`, `ROLL_OPP_percentageFieldGoalsAttempted3pt_scoring_{n}`, `ROLL_OPP_percentagePointsPaint_scoring_{n}`, etc.
- `matchup.py` combine les ratings via `MATCH_OFF_DEF_GAP_{n}` et `MATCH_DEF_VS_OPP_OFF_{n}`, et les diff auto `DIFF_*` couvrent les gaps d’événement (steals, blocks, turnovers forcés).
- On dispose également de métriques de fautes (`ROLL_OPP_percentagePersonalFouls_usage_{n}`) et de turnovers forcés (`ROLL_OPP_teamTurnoverPercentage_fourfactors_{n}`), mais elles restent séparées offense/défense.

### Gaps & plan concret

1. **Pression défensive explicite**  
   Ajouter dans `matchup.py` un helper `_add_defense_pressure_features` qui combine `ROLL_OPP_teamTurnoverPercentage_fourfactors_{n}`, `ROLL_OPP_percentageSteals_usage_{n}`, `ROLL_OPP_percentageBlocks_usage_{n}` et les metrics offensives correspondantes (`ROLL_percentageTurnovers_usage_{n}`) pour produire `MATCH_TURNOVER_PRESSURE_{n}` et `MATCH_PROTECTION_EDGE_{n}` (capacité à protéger la balle et le cercle).
2. **Fouliness & bonus hunting**  
   Créer `MATCH_FOUL_RATE_DIFF_{n}` = `HOME_ROLL_OPP_percentagePersonalFouls_usage_{n}` − `AWAY_ROLL_percentagePersonalFouls_usage_{n}` pour quantifier les matchs susceptibles d’aller sur la ligne. Implémenter au même endroit que ci-dessus, en ajoutant aussi un flag binaire `MATCH_BONUS_RISK_{n}` (> certain seuil) pour aider les modèles linéaires.
3. **Variabilité du profil concédé**  
   Coupler `OPP_percentageFieldGoalsAttempted3pt_scoring_{n}` avec la variance observée des 3PA adverses (`ROLL_OPP_percentageFieldGoalsAttempted3pt_scoring_std_{n}` qu’il faudra calculer via une nouvelle fonction `compute_rolling_std_features` dans `feature_builder`). Objectif : identifier les défenses qui laissent énormément tirer de loin (variance élevée → match plus swingy).

> ✅ Implémenté : `_add_variance_rollings` dans `team.py` (rolling std `_STD`) + `_add_defense_pressure_features`/`_add_shot_variance_features` dans `matchup.py` fournissent `MATCH_TURNOVER_PRESSURE`, `MATCH_PROTECTION_EDGE`, `MATCH_FOUL_RATE_DIFF`, `MATCH_BONUS_RISK`, `MATCH_3PT_VARIANCE`, etc.

---

## 4. Rôle des joueurs clés (usage, minutes, profil)

Ensuite, au niveau micro (joueurs) :

* **Usage rate des stars**

  * Un joueur à 30–35% d’usage qui est très efficient va tirer le total vers le haut s’il est là, et l’inverse s’il est absent.

* **Temps de jeu projeté**

  * Blessures / gestion des minutes (back-to-back, load management)
  * Des lineups bench-heavy peuvent être soit catastrophiques en attaque, soit ultra rapides et “run & gun”.

* **Profil des joueurs sur le terrain**

  * 5 shooters → spacing max → drives → layups + kickouts à 3.
  * Lineup avec 2 non-shooters → spacing pourri → plus de mid-range contestés → total points plus bas.

* **On/Off impact**

  * Offensive Rating avec le joueur ON vs OFF.
  * Certains bancs explosent en points encaissés/marqués, ce qui change totalement le script du match.

### État actuel (NBA_Predictor)

- Dans `src/features/availability.py`, on génère `ROLL_top_player_absent_rate_{3,5,10,25}`, `ROLL_num_absent_{n}`, `ROLL_has_top_absent_{n}` ainsi que les versions opposées de ces compteurs (en utilisant uniquement des signaux prematch).
- Les colonnes issues de `feature_aggregation.py` (`player_perf_score_mean`, `player_perf_score_sum`, `top_player_count`, `num_present`, etc.) sont conservées dans le bronze/silver mais ne sont ni rollées ni exploit ées directement par le modèle après le drop des colonnes brutes.
- Pas de projection de minutes ni d’info ON/OFF actuellement, et l’impact d’une absence se limite donc à un flag ou à un ratio de présence.

### Gaps & plan concret

1. **Rollings d’impact joueur**  
   Étendre `src/features/team.py::_add_generic_rollings` (ou ajouter une fonction dédiée) pour appeler `compute_rolling_features` avec `top_player_features_to_roll` (déjà défini dans `src/config.py`) sur les colonnes `player_perf_score_mean/sum`, `top_player_count`, `num_present`, etc. Ensuite, exposer `HOME_ROLL_player_perf_score_sum_{n}`, `MATCH_TOP_PLAYER_FORM_{n}`, etc.
2. **Quantifier les absences en points attendus**  
   Dans `src/feature_aggregation.py`, calculer pour chaque match la somme des `player_perf_score` des joueurs absents (par catégorie) et pousser ces colonnes jusque dans `availability.py`. Après rollings, créer `MATCH_TOP_USAGE_MISSING_{n}` = `ROLL_player_perf_score_sum_{n}` × `ROLL_top_player_absent_rate_{n}` pour donner une estimation continue de la perte d’attaque.
3. **Projection de minutes / rotations**  
   Bâtir un petit module `src/features/rotation.py` qui agrège les minutes jouées par lineup sur les N derniers matchs (`rolling_mean_minutes` par joueur poste) et expose `EXPECTED_MINUTES_TOP5` + `MATCH_BENCH_USAGE_GAP`. On peut partir du CSV `data/raw_last/player_stats` déjà utilisé pour les absences.

> ✅ Implémenté : `_add_top_player_rollings` dans `team.py`, `src/features/rotation.py` (nouveau step `rotation_metrics`) et `_add_player_personnel_features` dans `matchup.py` créent `HOME/AWAY_ROLL_player_perf_*`, `ROLL_AVG_MINUTES_PER_PLAYER_*`, `ROLL_BENCH_USAGE_RATIO_*` et les dérivés `MATCH_PLAYER_PERF_GAP`, `MATCH_ROTATION_DEPTH`, `MATCH_TOP_USAGE_MISSING`, `MATCH_BENCH_USAGE_GAP`.

---

## 5. Contexte du match (facteurs situationnels)

Ce sont des “modulateurs” importants :

* **Pace contextuelle**

  * Back-to-back, 3 matchs en 4 jours → fatigue, pace parfois plus lent, moins d’intensité défensive aussi (ça peut aller dans les deux sens).
* **Écart de niveau**

  * Blowout potentiel → garbage time.
  * Garbage time peut :

    * ralentir (si tout le monde marche)
    * ou se transformer en playground si les bancs veulent se montrer (beaucoup de 3, peu de défense).
* **Importance du match**

  * Match de saison régulière random vs rivalité, course playoffs, etc.
  * Les matchs serrés explodent le total en fin de match : fautes systématiques, lancers, timeouts, possessions rapides.

### État actuel (NBA_Predictor)

- Les signaux de contexte disponibles : `DAYS_SINCE_LAST_GAME`, `REST_ADVANTAGE`, `ROLL_WIN_RATIO_{n}`, `WIN_STREAK`, `ELO_PRE`, `ELO_PRE_SEASON`, `MATCH_ELO_GAP`, `MATCH_ELO_LEVEL_AVG`, les features H2H (`ROLL_HOME_H2H_WINRATE_*`, etc.).
- Les rollings de points/pace servent d’approximation de variance, mais nous n’avons pas encore de `STD` ou d’écarts-types pour quantifier la volatilité d’un match.
- Pas d’info explicite sur l’importance (playoffs, IST, rivalité) ni sur les spreads/totals bookmakers. Les moneylines `HOME_MONEYLINE`/`AWAY_MONEYLINE` ont été explicitement désactivées (cf. `src/datasets/bronze.py` où l’appel à `match_odds_with_dataset` est commenté) faute de données fiables, donc plus aucune feature liée aux odds n’est injectée pour l’instant.

### Gaps & plan concret

1. **Importance et phase de saison**  
   Ajouter dans `src/datasets/bronze.py::_prepare_games_meta` des flags `IS_PLAYOFF`, `IS_IN_SEASON_TOURNAMENT`, `IS_FINAL_WEEK` (via `SEASON` + `GAME_DATE`). Propager ces colonnes jusqu’au silver/gold puis créer dans `team.py` des rollings `ROLL_PLAYOFF_INTENSITY_{n}` pour capturer l’effort contextuel.
2. **Bookmaker spread & total**  
   Faire évoluer `src/feature_builder.match_odds_with_dataset` pour fusionner aussi les colonnes `HOME_SPREAD`, `AWAY_SPREAD`, `TOTAL_POINTS` (elles existent dans `data/odds_history` mais ne sont pas exportées). Une fois alignées, créer `IMPLIED_TOTAL`, `IMPLIED_PACE`, `IMPLIED_EXTRA_POSSESSIONS` et les écarts vs nos features (`MATCH_PACE_{n}`) afin d’exploiter la sagesse du marché. Cette étape dépendra du ré-approvisionnement fiable des odds (pour l’instant désactivés dans `bronze.py`).
3. **Variance / garbage-time risk**  
   Implémenter `compute_rolling_std_features` (ou étendre `compute_rolling_features` avec `method='std'`) pour obtenir `ROLL_points_traditional_std_{n}`, `ROLL_pace_advanced_std_{n}`. Dans `matchup.py`, combiner ces STD avec `MATCH_ELO_GAP` et les moneylines pour créer `MATCH_BLOWOUT_RISK` + `MATCH_CLUTCH_FLAG` (ELO gap faible + variance faible). Cela modélise l’impact garbage-time/OT mentionné ci-dessus.

> ✅ Implémenté : flags `IS_PLAYOFF`/`IS_IN_SEASON_TOURNAMENT`/`IS_FINAL_WEEK` enrichissent désormais le bronze, `_add_context_features` les roll sur `MATCHUP_WINDOWS`, et `matchup.py` génère `MATCH_IMPORTANCE_SCORE`, `MATCH_IS_PLAYOFF`, `MATCH_BLOWOUT_RISK`, `MATCH_CLUTCH_FLAG`, `MATCH_POINTS_VARIANCE`, `MATCH_PACE_VARIANCE`.

---

## 6. Si on schématise pour un modèle

Pour un modèle qui prédit les points d’une équipe, tu peux voir ça comme :

> **Points = Possessions × Points par possession**

Avec des features type :

* **Possessions prévues :**

  * Pace_home, Pace_away
  * TO% home/away
  * ORB% home/away
  * Match tempo context (back-to-back, rest days, etc.)

* **Points par possession (Off vs Def match-up) :**

  * Offensive Rating home, Defensive Rating away
  * Profil de tir (3PA%, FTr, % at rim, % mid-range) de l’attaque
  * Profil de tir concédé par la défense
  * EFG%, TS%, Turnover rate

* **Context & joueurs :**

  * Minutes projetées stars, absences, on/off offensive impact
  * Pace et ORtg en présence de certains lineups
  * Importance du match, spread / ligne du bookmaker (qui encode beaucoup d’infos implicites).
