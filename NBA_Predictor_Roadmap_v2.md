# 🏀 NBA Predictor — Roadmap Professionnelle de Refactorisation MLOps (v2)

> Mises à jour : ajout d’une **stratégie notebooks de debug/exploitation** pour chaque étape du pipeline, et ajout d’un **bloc de features Head‑to‑Head (H2H)** entre équipes, intégré au Feature Store Feast et aux flows.

---

## 1. Objectif et Vision

Le projet **NBA Predictor** vise à prédire le vainqueur d’un match NBA en se basant sur les statistiques historiques des équipes et des joueurs.  
Cette v2 renforce deux axes :
- **Observabilité & debug** via une collection de **notebooks dédiés** par étape (ingestion, features, Feast, entraînement, scoring, monitoring).  
- **Pouvoir prédictif** via l’ajout de **features H2H (Team vs Opponent)**, calculées en batch et servies depuis Feast en **point-in-time correctness**.

Stack inchangée : **Prefect** (orchestration), **Feast** (feature store), **MLflow** (tracking & registry), **Docker** (services), **Python venv** (dev).

---

## 2. Données et Sources

- **Matchs & boxscores** (2000‑01 → 2024‑25 + partielle 2025‑26) via `nba_api`  
- **Équipes (statiques)** : mapping `team_id` stable, métadonnées (nom, abréviation, ville…)  
- **Mise à jour quotidienne automatisée** via flow Prefect  
- **(Optionnel phase 2)** : blessures, cotes de paris, calendrier futur

> Décision : on reste **100% données internes** pour la v2 ; architecture **modulaire** pour brancher des sources externes plus tard.

### 2.1 Stratégie Data Lake (Bronze / Silver / Gold)

- `data/legacy/raw` et `data/legacy/raw_last` : miroirs gelés des CSV existants pour préserver la compatibilité avec les notebooks/scripts actuels (lecture seule).
- `data/bronze/nba_api/<source>/season=<YYYY>/ingest_ts=<timestamp>/` : dumps bruts issus de `nba_api` (games, boxscores `traditional|advanced|...`, disponibilité joueurs). Format unique Parquet, schéma identique à l’API, métadonnées (`ingest_ts`, `source_system`).
- `data/silver/` : tables harmonisées pour la modélisation (`team_game_facts`, `team_boxscores_agg`, `player_availability`, `team_form_windowed`, `matchups_h2h_base`). Clefs normalisées (`TEAM_ID`, `OPP_TEAM_ID`, `GAME_DATE`, `GAME_ID`, `SEASON`), colonnes décrites plus bas, stockage Parquet partitionné `dataset`/`season`.
- `data/gold/` : jeux prêts pour entraînement et scoring (`training_sets`, `scoring_payloads`, `monitoring_snapshots`) contenant les features finales (TeamAggregated + H2H + features diff).
- Flux de transition : le scraper actuel continue d’écrire dans `data/legacy/*` pendant la migration ; un job `bronze_rebuild` copie/convertit ces CSV vers bronze. Une fois les pipelines refactorés, l’écriture se fera directement dans bronze puis déclenchera les jobs silver → gold → Feast.
- Configuration : `NBA_DATA_ROOT` + `settings.data_paths` (cf. §4.1) pilotent dynamiquement les chemins pour éviter les chemins en dur dans le code et les notebooks.

---

## 3. Stack Technique

| Outil | Rôle | Détails |
|---|---|---|
| **Prefect** | Orchestration | Flows : update_data, train_model, predict |
| **Feast** | Feature Store | Entités `team`, `opponent` ; FeatureViews TeamAggregated & TeamVsOppH2H |
| **MLflow** | Tracking & Registry | Runs, métriques, artefacts, promotion de modèles |
| **Docker** | Services | Postgres (Feast/MLflow metadata) + pgAdmin UI ; les jobs Python tournent dans le venv |
| **Python venv** | Dev | Environnement isolé |

---

## 4. Architecture Cible

```
nba-predictor/
│
├── data/
│   ├── legacy/
│   │   ├── raw/
│   │   └── raw_last/
│   ├── bronze/
│   │   └── nba_api/
│   │       ├── games/
│   │       ├── boxscores/
│   │       └── player_availability/
│   ├── silver/
│   │   ├── team_game_facts/
│   │   ├── team_boxscores_agg/
│   │   ├── player_availability/
│   │   ├── team_form_windowed/
│   │   ├── matchups_h2h_base/
│   │   └── feast_offline/
│   └── gold/
│       ├── training_sets/
│       ├── scoring_payloads/
│       └── monitoring_snapshots/
│
├── src/
│   ├── nba_predictor/
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── settings.py
│   │   │   └── logging.py
│   │   ├── data_ingest/
│   │   │   ├── nba_api_client.py
│   │   │   └── bronze_writer.py
│   │   ├── transformations/
│   │   │   ├── team_aggregates.py
│   │   │   ├── team_vs_opp_h2h.py
│   │   │   └── player_availability.py
│   │   ├── pipelines/
│   │   │   ├── update_data.py
│   │   │   ├── train_model.py
│   │   │   └── predict.py
│   │   ├── feature_store/
│   │   │   ├── feature_store.yaml
│   │   │   └── feature_views/
│   │   ├── logging/
│   │   │   └── setup.py
│   │   └── utils/
│   ├── scripts/
│   │   ├── scrape_boxscores.py
│   │   ├── rebuild_bronze.py
│   │   ├── build_silver.py
│   │   └── compat/
│   └── legacy/                               # wrapper pour ne pas casser les notebooks existants
│       └── nba_scrapping.py
│
├── flows/
│   ├── update_data_flow.py
│   ├── train_model_flow.py
│   └── predict_flow.py
│
├── notebooks/
│   ├── 00_data_checks.ipynb
│   ├── 10_update_data_debug.ipynb
│   ├── 20_features_team_debug.ipynb
│   ├── 21_features_h2h_debug.ipynb
│   ├── 30_feast_validation.ipynb
│   ├── 40_train_debug.ipynb
│   ├── 50_predict_debug.ipynb
│   ├── 90_monitoring.ipynb
│   └── legacy/                               # notebooks historiques conservés en lecture
│
├── models/
│   ├── mlruns/
│   └── registry/
│
├── docker-compose.yml
├── Makefile
└── README.md
```

> Compatibilité : jusqu’à la migration complète des notebooks, `src/legacy/nba_scrapping.py` expose les fonctions actuelles (`download_games_for_seasons`, `scrape_boxscores_v3_for_games`, etc.) en s’appuyant sur les nouveaux modules `nba_predictor`. Les notebooks historiques sont déplacés dans `notebooks/legacy/` et consomment exclusivement `data/legacy/*`.

**Politique notebooks**  
- **Pas d’écriture** vers les stores de prod (Feast/MLflow) depuis les notebooks : ils appellent des **fonctions du code** (packages internes) en lecture, ou exécutent des **sous‑ensembles contrôlés**.  
- Chaque notebook commence par un **bloc “Contexte run”** (commit hash, date de coupure `training_data_until`, env vars) et finit par une **checklist**.  
- Conseillé : **Jupytext** (`.py` ↔ `.ipynb`) ou **nbdev** pour versionner proprement.

### 4.1 Modularisation du code

- `nba_predictor.config.settings.Settings` (Pydantic) centralise les chemins (`NBA_DATA_ROOT`, emplacements logs), les secrets (`NBA_API_RATE_LIMIT`) et les options (batch size, N_LIST). Les modules importent `from nba_predictor.config import settings` plutôt que `from src.config import *`.
- `nba_predictor.data_ingest` encapsule l’accès `nba_api` : clients (retry, backoff), orchestrateurs par saison, writer bronze (Parquet + métadonnées), gestion des incréments (`latest_game_date`, `checkpoint`).
- `nba_predictor.transformations` porte la logique de features : agrégations d’équipe (réutilise `cols_to_sum/cols_to_weighted_avg`), disponibilité joueurs, fenêtres roulantes, décorateurs de validation, exports Parquet vers silver.
- `nba_predictor.feature_store` contient `feature_store.yaml`, les FeatureViews et des helpers (`load_repo`, `materialize_incremental`) partagés entre CLI et Prefect.
- `nba_predictor.pipelines` expose les fonctions utilisées par Prefect (`update_data`, `train_model`, `predict`) avec une signature pure (paramètres dataclasses). On peut les appeler depuis les notebooks de debug (lecture seule).
- `nba_predictor.logging.setup` fournit `configure_logging()` (voir §5.3) ; tous les modules utilisent `logger = structlog.get_logger(__name__)`.
- `src/scripts/` héberge des CLI légers (Typer) pour lancer manuellement l’ingestion, reconstruire bronze à partir des CSV legacy ou exécuter une transformation spécifique.
- `src/legacy/` garde un wrapper minimal vers les anciennes fonctions pour ne pas casser les notebooks le temps de la transition. Objectif : retirer ce dossier une fois les notebooks migrés.

---

## 5. Composants MLOps (rappel)

### 5.1 Prefect (flows)
- `update_data_flow` : ingestion matchs + boxscores → recalcul features Team + H2H → apply/materialize Feast.  
- `train_model_flow` : extraction **point‑in‑time** (Team + H2H), split temporel, entraînement (XGB/LGBM), logging MLflow, registry.  
- `predict_flow` : vérification fraicheur, extraction features à date `match_date`, chargement modèle “Production”, scoring, persistance.

### 5.2 MLflow (tracking & registry)
- Logs : hyperparams, métriques (AUC, logloss, Brier, calibration), **tags** (`training_data_until`, `feature_set_version`, `code_sha`).  
- Artefacts : modèle, **snapshot Parquet** du dataset d’entraînement, rapport EDA, matrice de confusion.  
- Registry : promotion `Staging` → `Production` + commentaires de validation.

### 5.3 Logging & observabilité
- Configuration centralisée via `nba_predictor.logging.setup.configure_logging()` (logging stdlib + `structlog`) : format JSON, timestamps UTC, champs (`pipeline`, `run_id`, `team_id`, `game_id`, `component`).
- Sorties : `logs/pipelines/<pipeline>.log` (fichiers) + console ; option future d’expédition vers Prefect/MLflow Artifacts.
- `log_boxscores_scrapping` devient un wrapper vers `logger.info("scrape.boxscores", **context)` pour conserver le comportement actuel tout en homogénéisant le format.
- Prefect capte automatiquement les logs ; les notebooks de debug utilisent `configure_logging()` en lecture seule pour répliquer le même rendu.
- Healthchecks dédiés : script `scripts/monitor_log_volume.py` surveille l’absence de logs (alerting simple via mail/slack dans une phase ultérieure).

---

## 6. Feast — Feature Store (Team + H2H)

### 6.0 Offline store & stockage
- Offline store Parquet : `data/silver/feast_offline/<feature_view>/version=v1/season=<YYYY>/part=<hash>.parquet`.
- Les transformations publient les tables sources dans `data/silver/team_form_windowed/v1/` (TeamAggregated) et `data/silver/matchups_h2h_features/v1/` (H2H). `feast apply` pointe vers ces chemins via `feature_store.yaml`.
- On conserve Postgres (Docker) comme online store léger pour la mise en production ; credentials fournis via `.env` chargé par `Settings`.
- Les snapshots bronze → silver contiennent déjà `created_at` et `ingest_ts`, ce qui permet à Feast d’assurer la point-in-time correctness sans logique spécifique côté FeatureView.

### 6.1 Entités
- `Team`: `team_id` (clé stable)  
- `Opponent`: `opponent_team_id` (clé stable)  
- Timestamps : `event_timestamp = game_date` (UTC ISO), **granularité jour**

### 6.2 FeatureViews
#### A) `TeamAggregatedStats` (existant)
- Fenêtres glissantes : 5, 10, 20 derniers matchs
- Exemples : `avg_points_last5`, `avg_points_last10`, `win_rate_last10`, `current_win_streak`, `season_avg_points`

#### B) `TeamVsOppH2H` (nouveau)
- **Entities** : `[team_id, opponent_team_id]`  
- **Colonnes minimales** : `team_id, opponent_team_id, game_date, target (win_home? win_team?), team_pts, opp_pts, pace, ...`  
- **Fenêtres & pondérations** (configurables) :  
  - *Roulantes* : 3, 5, 10 derniers H2H  
  - *Décroissance temporelle* (exponentielle `alpha`), pour donner plus de poids aux confrontations récentes  
  - *Bornage saisonnier* optionnel (ex : 4 dernières saisons max)

**Features H2H proposées**  
| Feature | Description |
|---|---|
| `h2h_games_lastN` | Nombre de confrontations jouées (échantillon) |
| `h2h_win_rate_lastN` | Taux de victoire de `team` vs `opponent` (N derniers H2H) |
| `h2h_avg_margin_lastN` | Différence moyenne de score (team_pts − opp_pts) |
| `h2h_pts_for_lastN` / `h2h_pts_against_lastN` | Points moyens marqués/encaissés vs opponent |
| `h2h_efg_for_lastN` / `h2h_efg_against_lastN` | eFG% moyens (si dispo) |
| `h2h_turnover_rate_diff_lastN` | Diff. de TO% moyens |
| `h2h_reb_rate_diff_lastN` | Diff. de taux de rebonds |
| `h2h_win_rate_exp_decay` | Win% avec **poids exponentiels** (récent > ancien) |
| `h2h_win_rate_smooth` | **Lissage bayésien** : `(wins + prior * k) / (n + k)` où `prior ≈ win_rate_team_last20` |

**Garde‑fous (data leakage & robustesse)**  
- Point‑in‑time via Feast : on **n’utilise que** des H2H **antérieurs** à `match_date`.  
- **Seuils d’échantillon** : fallback vers des priors génériques si `h2h_games_lastN < min_n`.  
- **Playoffs vs Regular** (option) : drapeau `is_playoff` pour segmenter si utile.  
- **Changements d’identité** : on travaille exclusivement avec `team_id` (mapping maintenance).

### 6.3 Implémentation (batch)
- `build_team_boxscores_agg.py` :  
  - Agrégation par `team_id`/`game_id` des boxscores joueurs (Polars).  
  - Taxonomie d’agrégation :  
    - `SUM` pour les compteurs / volumes (`made`, `attempted`, `points`, `rebounds`, `assists`, `steals`, `blocks`, `turnovers`, `fouls`, `minutes`, `possessions`).  
    - `WEIGHTED_MINUTES` pour les ratios d’usage et part de points (`percentage*`, `usagePercentage_advanced`, `percentagePoints*`, etc.) avec pondération par minutes.  
    - `RECOMPUTE_RATIO` pour les pourcentages (`*_pct`, `*_ratio`, `*_rate`) à partir des totaux agrégés (ex : `team_fg_pct = team_fgm_sum / team_fga_sum`).  
    - `FIRST` (avec validation variance≈0) pour les métriques réellement dupliquées sur chaque ligne (`offensiveRating_advanced`, `pace_advanced`, `possessions_advanced`, `estimatedNetRating_advanced`, `teamTurnoverPercentage_fourfactors`, etc.), avec log d’alerte si divergence.  
    - Possibilité d’enrichir par des ratios dérivés (ex : `net_rating` recalculé via points/possessions) si l’API n’est pas cohérente.  
  - Contrôles de qualité : somme des minutes ≈ 240 (ou 300 en OT), égalité `points_traditional == PTS` provenant de `team_game_facts`, comparaison turnovers/possessions.  
  - Publication Parquet → `data/silver/team_boxscores_agg/season=<YYYY>/`.  
- `build_player_availability.py` :  
  - Récupère `commonteamroster` + injury reports pour la saison (bronze `player_availability`).  
  - Compare la liste des joueurs attendus vs `boxscores` joués (minutes > 0).  
  - Produit des indicateurs : `active_players`, `inactive_players`, `starters_available` (top 5 par minutes moyennes sur les 10 matchs précédents, paramètres `core_top_n` / `core_minutes_window` ajustables), `minutes_share_top3`, `missing_starters`, `injury_reported`, `two_way_players_active`, etc.  
  - Alimente `data/silver/player_availability/season=<YYYY>/` avec `team_id`, `game_id`, `game_date`, `injury_designation`, `availability_flags`.
- `build_team_form_windowed.py` (Polars) lit `data/silver/team_boxscores_agg/` joint à `team_game_facts`, applique les fenêtres `N_LIST` (paramétrables) et génère :
  - moyennes/écarts-types glissants `team_*_avg_lastN` / `team_*_std_lastN`,
  - win rate & nombre de victoires (`win_rate_lastN`, `wins_lastN`),
  - colonnes de contexte (`is_home`, `opponent_team_id`) pour la suite des features diff.
- `build_matchups_h2h_base.py` : joint `team_game_facts` & `team_boxscores_agg` pour obtenir une ligne par `(team_id, opponent_team_id, game_id)` avec margin, stats de pace/possessions et contexte home/away.
- `build_matchups_h2h_features.py` :  
  1) Trie par `(team_id, opponent_team_id, game_date)` et applique les fenêtres `N_LIST`.  
  2) Produit pour chaque fenêtre : moyennes/écarts-types, `h2h_win_rate_lastN`, `h2h_wins_lastN`, `h2h_margin_avg_lastN`.  
  3) Prépare le Parquet final dans `data/silver/matchups_h2h_features/season=<YYYY>/` consommé par Feast.
- `feature_views/team_vs_opp_h2h.py` référence cette table, joint fallback sur `team_form_windowed` pour les features globales, expose les features `h2h_*` et `team_form_*` nécessaires à la modélisation.
- `materialize-incremental` se limite aux matchups dont `game_date` ≥ dernière date disponible dans bronze, en s’appuyant sur le watermark enregistré dans Postgres.

### 6.4 Utilisation à l’entraînement / prédiction
- **Entraînement** : extraire TeamAggregated **et** TeamVsOppH2H pour (home, away) et les **combiner** (features “home”, “away” + **différences** et/ou ratios).  
- **Prédiction** : même extraction à `match_date` (ou `match_date - 1d` selon convention).

### 6.5 Scaffold Feast
- `feature_store/feature_store.yaml` paramètre Postgres (online) + FileSource (offline `data/silver/feast_offline`).  
- `feature_store/feature_views/team_aggregated_stats.py` & `team_vs_opp_h2h.py` exposent des builders capables d’inférer automatiquement les `Field` via Polars (`infer_fields_from_parquet`) en excluant les colonnes de clé/metadata.  
- Les entités (`team`, `opponent`) sont centralisées et réutilisées dans les FeatureViews ; la configuration accepte des chemins overrides via paramètres pour faciliter les tests.
- Une fois les tables silver générées, `PYTHONPATH=src feast apply` matérialisera les FeatureViews sans modifier le code (schéma détecté à la volée).

---

## 7. Plan de Mise en Œuvre — Étapes Détaillées (v2)

### Étape 0 — Migration structure & configuration
- Créer `data/{legacy,bronze,silver,gold}` puis déplacer `data/raw*` dans `data/legacy/`.
- Écrire `scripts/rebuild_bronze.py` : lit les CSV legacy (`raw`, `raw_last`), ajoute métadonnées (`ingest_ts`, `source`) et publie Parquet dans `data/bronze/nba_api/...`.
- Introduire `nba_predictor.config.settings.Settings` (Pydantic) + `settings.toml` : centralise les chemins, API keys, paramètres de features (`N_LIST`, `BATCH_SIZE`) et file le type (dev/prod).
- Mettre à jour `src/config.py` → wrapper vers `Settings`; conserver les constantes nécessaires aux notebooks legacy dans `src/legacy/config_compat.py`.

### Étape 1 — Environnement
```bash
python -m venv venv && source venv/bin/activate
pip install -U pip
pip install prefect feast mlflow pandas numpy polars structlog typer pydantic-settings \
            scikit-learn xgboost lightgbm
```
- Mettre à jour `requirements.txt` + `Makefile` (cibles `lint`, `format`, `test`, `data-update`).

### Étape 2 — Ingestion Bronze & préparation Silver
- Refactor `nba_scrapping.py` en `nba_predictor.data_ingest` : clients `nba_api`, orchestration saison, gestion batch, retry & logging unifiés.
- Writer Bronze : `bronze_writer.py` sérialise en Parquet (partitions `source`, `season`, `ingest_ts`) et maintient un `manifest.json` pour les incréments.
- Générer `team_game_facts` & `team_boxscores_agg` (Polars) à partir des bronze boxscores : règles d’agrégation explicites (somme pour les compteurs, recomputation des pourcentages, extraction du premier enregistrement pour les métriques team-level, contrôles d’équilibre minutes/possession). Publier dans `data/silver/...`.
- Ingestion côté bronze des rosters (`commonteamroster`, `teamgamelog`) et injury reports (`playergamelog`, `injuryreport`) ; persistés dans `data/bronze/nba_api/player_availability/`.
- Construire `player_availability` à partir des bronze boxscores + rosters/injury reports → `data/silver/player_availability/`.
- CLI : `PYTHONPATH=src python3 -m scripts.rebuild_bronze --help` puis `PYTHONPATH=src python3 -m nba_predictor.transformations.team_game_facts` (et modules associés) pour déclencher les builds manuellement, en attendant l’intégration Prefect.
- Définir un mapping de colonnes → stratégie d’agrégation (cf. `nba_predictor/transformations/schema.py`) pour pouvoir étendre facilement la logique au fil des ajouts de colonnes Feast.

### Étape 3 — Feast (TeamAggregated + H2H)
1) Définir `team_aggregated_stats.py` (reuse, pointant sur `team_form_windowed`).  
2) Implémenter `team_vs_opp_h2h.py` (FeatureView) avec schemas complets (`event_timestamp`, `created_at`, entités).  
3) Implémenter `build_team_vs_opp_h2h.py` (voir §6.3) et publier `matchups_h2h_features`.  
4) `feast apply` puis `feast materialize-incremental --end <today>` ; stocker l’état dans Postgres (Docker).

### Étape 4 — Prefect (flows refactor)
- `update_data_flow`: taches `scrape_games`, `scrape_boxscores`, `sync_player_availability`, `rebuild_silver`, `publish_feast`. Variante `legacy_update` appelle encore les scripts d’origine si besoin.
- `train_model_flow`: extraction `get_historical_features` (Team + H2H home/away) → construction dataset diff → entraînement (MLflow) → enregistrement modèle + artefacts.
- `predict_flow`: vérification fraicheur (Feast `last_materialization`) → features `point-in-time` → prédictions → persistance dans `data/gold/scoring_payloads/`.
- Scaffolding `flows/*` déjà en place : tâches Prefect appellent les transformations (bronze optionnel, silver, H2H) et exposent des placeholders pour le training/pred scoring en attendant l’intégration MLflow/Feast.

### Étape 5 — Notebooks & observabilité
- Créer les notebooks de debug listés §4 (lecture seule) + un `00_data_checks.ipynb` branché sur `data/bronze`.
- Migrer progressivement les notebooks historiques : ils deviennent consommateurs des nouvelles fonctions (`from nba_predictor.legacy import scrape_boxscores`).
- Ajouter un notebook `15_logs_observability.ipynb` pour inspecter les logs JSON, courbes de volume et erreurs.

### Étape 6 — Validation & tests
- Tests unitaires :  
  - construction **pairwise** H2H,  
  - fenêtres roulantes & lissage bayésien,  
  - respect de la point-in-time correctness (assert `event_timestamp < match_date`).  
- Tests d’intégration : `scripts/system/update_data.sh` enchaîne bronze → silver → Feast (mode dry-run).  
- Dry-run complet Prefect : `update_data` → `train_model` → `predict` avec enregistrement MLflow (`feature_set_version="team+H2H_v1"`).

---

## 8. Exemples de snippets

### 8.1 Diff features (entraînement / prédiction)
```python
# df_home, df_away: features Team + H2H pour home/away au même timestamp
feat_cols = [c for c in df_home.columns if c.startswith(("avg_", "win_rate", "season_", "h2h_"))]
df_model = (df_home[["match_id"] + feat_cols]
            .merge(df_away[["match_id"] + feat_cols], on="match_id", suffixes=("_home", "_away")))
for c in feat_cols:
    df_model[f"{c}_diff"] = df_model[f"{c}_home"] - df_model[f"{c}_away"]
```

### 8.2 Lissage bayésien (ex. win rate H2H)
```python
def smooth_rate(wins, n, prior, k=10.0):
    # k = force de régularisation (pseudo-comptes), prior ~ win% global récent de team
    return (wins + prior * k) / (n + k)
```

---

## 9. Planification & Exécution

- Prefect schedules (exemple) :  
  - `update_data_flow` → 02:00 CET (post‑jeux)  
  - `predict_flow` → 12:00 CET (après materialize)  
- Prefect Agent + MLflow UI via docker-compose.  
- Conserver un **rapport quotidien** (CSV/Parquet) de features & prédictions pour audit.

### 9.1 Rythme d’adoption (anti-empilement)
1. **Phase 0 — Structure & logging** : mettre en place les répertoires bronze/silver/gold, migrer les constantes (`Settings`), activer le nouveau logger tout en gardant les scripts legacy fonctionnels.
2. **Phase 1 — Bronze stable** : refactor scraper → bronze, valider la parité des données vs legacy (`notebooks/legacy`), publier la documentation d’ingestion.
3. **Phase 2 — Silver & tests** : produire `team_game_facts`, `team_boxscores_agg`, `player_availability`, couvrir par des tests unitaires + notebook `20_features_team_debug`.
4. **Phase 3 — Feast + Postgres** : n’activer `feast apply` qu’une fois les tables silver figées ; déployer Postgres/pgAdmin via Docker.
5. **Phase 4 — Prefect flows** : brancher Prefect sur les fonctions packagées, exécuter un run complet en dry-run, monitorer via logs/MLflow.
6. **Phase 5 — MLflow & notebooks de prod** : activer le tracking, migrer les notebooks de debug en lecture seule, retirer progressivement `src/legacy`.
> Règle : passer à la phase suivante uniquement lorsque la précédente est validée (tests + notebook de debug à jour), afin d’éviter l’empilement simultané de chantiers.

---

## 10. Validation Finale (v2)

- H2H intégré aux datasets d’entraînement & de prédiction.  
- Notebooks disponibles pour chaque étape, non‑intrusifs, facilitant **debug et QA**.  
- Feast garantit la **point‑in‑time correctness** sur Team et H2H.  
- MLflow trace les versions de **feature set** et la date de coupure.

---

## 11. Conclusion

Cette v2 renforce l’**observabilité** (notebooks par étape) et la **puissance prédictive** (features H2H) tout en respectant la rigueur temporelle via Feast.  
La suite logique : tuning (Optuna), ajout de features avancées (ratings/pace), et intégration progressive de données externes (blessures, cotes).
