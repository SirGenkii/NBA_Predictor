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

---

## 3. Stack Technique

| Outil | Rôle | Détails |
|---|---|---|
| **Prefect** | Orchestration | Flows : update_data, train_model, predict |
| **Feast** | Feature Store | Entités `team`, `opponent` ; FeatureViews TeamAggregated & TeamVsOppH2H |
| **MLflow** | Tracking & Registry | Runs, métriques, artefacts, promotion de modèles |
| **Docker** | Services | Prefect, MLflow UI, DB/Parquet |
| **Python venv** | Dev | Environnement isolé |

---

## 4. Architecture Cible

```
nba-predictor/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── parquet/
│
├── features/
│   ├── feature_store.yaml
│   ├── feature_views/
│   │   ├── team_aggregated_stats.py          # TeamAggregatedStats
│   │   └── team_vs_opp_h2h.py                # TeamVsOppH2H (nouveau)
│   └── transformations/
│       ├── build_team_aggregates.py
│       └── build_team_vs_opp_h2h.py          # génération H2H (nouveau)
│
├── flows/
│   ├── update_data_flow.py
│   ├── train_model_flow.py
│   └── predict_flow.py
│
├── notebooks/                                # Notebooks “lecture seule” (debug/exploitation)
│   ├── 00_data_checks.ipynb                  # santé données brutes (comptes, nulls, duplicats)
│   ├── 10_update_data_debug.ipynb            # inspection d’un run d’ingestion (logs, delta matchs)
│   ├── 20_features_team_debug.ipynb          # vérification des features TeamAggregated
│   ├── 21_features_h2h_debug.ipynb           # vérification des features TeamVsOppH2H (nouveau)
│   ├── 30_feast_validation.ipynb             # get_historical_features / online_features
│   ├── 40_train_debug.ipynb                  # jeux d’entraînement, splits, métriques locales
│   ├── 50_predict_debug.ipynb                # features point-in-time + prédictions unitaires
│   └── 90_monitoring.ipynb                   # suivi drift/métriques (Evidently, récap MLflow)
│
├── models/
│   ├── mlruns/
│   └── registry/
│
├── config.py
├── docker-compose.yml
└── README.md
```

**Politique notebooks**  
- **Pas d’écriture** vers les stores de prod (Feast/MLflow) depuis les notebooks : ils appellent des **fonctions du code** (packages internes) en lecture, ou exécutent des **sous‑ensembles contrôlés**.  
- Chaque notebook commence par un **bloc “Contexte run”** (commit hash, date de coupure `training_data_until`, env vars) et finit par une **checklist**.  
- Conseillé : **Jupytext** (`.py` ↔ `.ipynb`) ou **nbdev** pour versionner proprement.

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

---

## 6. Feast — Feature Store (Team + H2H)

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
- Script `features/transformations/build_team_vs_opp_h2h.py` :  
  1) Construire table **pairwise**: une ligne par (team_id, opponent_team_id, game_date).  
  2) Calculer cumul & fenêtres roulantes (Pandas/Polars) + version **exp. decay** (poids `w_t = exp(-lambda * age_days)`).  
  3) Appliquer **lissage bayésien** pour win rate (prior = forme récente globale de `team`).  
  4) Écrire Parquet → source offline Feast.  
- Définir `feature_views/team_vs_opp_h2h.py` puis `feast apply` (+ `materialize-incremental`).

### 6.4 Utilisation à l’entraînement / prédiction
- **Entraînement** : extraire TeamAggregated **et** TeamVsOppH2H pour (home, away) et les **combiner** (features “home”, “away” + **différences** et/ou ratios).  
- **Prédiction** : même extraction à `match_date` (ou `match_date - 1d` selon convention).

---

## 7. Plan de Mise en Œuvre — Étapes Détaillées (v2)

### Étape 1 — Environnement
```bash
python -m venv venv && source venv/bin/activate
pip install prefect feast mlflow pandas numpy polars scikit-learn xgboost lightgbm
```

### Étape 2 — Feast (MAJ)
1) Ajouter entité `Opponent` ; vérifier `feature_store.yaml` offline store (Parquet/SQL).  
2) Implémenter `team_vs_opp_h2h.py` (FeatureView) + config schemas.  
3) Implémenter `build_team_vs_opp_h2h.py` (transfo batch) et l’intégrer au flow d’update.  
4) `feast apply && feast materialize-incremental --end <today>`.

### Étape 3 — Prefect (MAJ flows)
- `update_data_flow`: après ingestion boxscores → **recalcul H2H** (équipes impactées par nouveaux matchs) → push Parquet → `feast apply` (si schéma stable) → `materialize-incremental`.  
- `train_model_flow`: extraction **PII** (Team + H2H) pour chaque match d’entraînement → features home/away + features **diff** (`home - away`) → entraînement + MLflow.  
- `predict_flow`: extraction à `match_date` (Team + H2H home/away) → features diff → scoring → persistance.

### Étape 4 — Notebooks (ajouts clés)
- `21_features_h2h_debug.ipynb` :  
  - Sélection d’un couple (`team_id`, `opponent_id`) et d’une date,  
  - Affichage ligné des H2H historiques,  
  - Recalcul “à la main” d’une feature vs sortie Feast (comparaison tolérance),  
  - Courbes de **poids exponentiels** & effet du **lissage bayésien**.  
- `30_feast_validation.ipynb` : démonstration `get_historical_features()` multi‑entités.  
- Tous les notebooks : **lecture seule**, pas d’`apply`/`materialize`/écritures.

### Étape 5 — Validation & Tests
- Tests unitaires pour :  
  - construction **pairwise** H2H,  
  - fenêtres roulantes,  
  - lissage bayésien (cas limites n=0, n<k),  
  - invariants (pas de valeurs futures).  
- Dry‑run complet : update → train → predict avec H2H inclus.  
- Vérification MLflow : tags `feature_set_version="team+H2H_v1"`.

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
