
# 🏀 NBA Predictor

**NBA Predictor** est une pipeline complète de collecte, traitement, visualisation et modélisation de données issues des matchs NBA. L’objectif est de prédire l’issue d’un match de NBA à partir de données historiques enrichies par du feature engineering et des modèles de machine learning.

---

## 📁 Structure du projet

```
NBA_Predictor-main/
├── 01_scrapping_boxscores.ipynb           # Scraping historique des boxscores de matchs
├── 02_update_last_boxscores.ipynb         # Mise à jour avec les derniers matchs
├── 03_scrapping_players.ipynb             # Scraping des infos joueurs
├── 04_boxscores_feature_enginering.ipynb  # Feature engineering sur les boxscores
├── 05_data_visualization.ipynb            # Analyse exploratoire et visualisation
├── src/
│   ├── config.py                           # Paramètres du projet
│   ├── nba_scraping.py                     # Fonctions de scraping
│   └── utils.py                            # Fonctions utilitaires
├── data_examples/                          # Exemples de données (raw, nettoyées, features)
├── README.md
```

---

## 🔄 Pipeline

1. **Scraping des données** :
   - Extraction des matchs NBA (de 2000 à aujourd’hui).
   - Récupération des statistiques individuelles (boxscores).

2. **Nettoyage & fusion** :
   - Suppression des doublons, harmonisation des formats de dates et identifiants.

3. **Feature Engineering** :
   - Calcul de moyennes mobiles, performances passées, ratios personnalisés.

4. **Visualisation** :
   - Analyse des distributions de performances par équipe/joueur.

5. **Modélisation (non inclus ici)** :
   - Prévu pour intégrer XGBoost, LightGBM, RandomForest, etc.

---

## 📦 Dépendances

- `nba_api`
- `pandas`
- `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `lightgbm`, `catboost`

```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer le projet

```bash
# Étapes principales
1. Exécuter `01_scrapping_boxscores.ipynb` pour télécharger les données historiques
2. Utiliser `02_update_last_boxscores.ipynb` pour mettre à jour les matchs récents
3. Fusionner et traiter avec `04_boxscores_feature_enginering.ipynb`
4. Visualiser avec `05_data_visualization.ipynb`
```

---

## 👀 Visualiser rapidement les paris (mass_prediction_nba)

- Voir un résumé console du dernier run: `make nba-mass-report`
- Forcer un run: `make nba-mass-report REPORT_ARGS="--run-id nba-run-20251113-194436-401517"`
- Filtrer par date: `make nba-mass-report REPORT_ARGS="--date 2025-11-13"`
- L'outil affiche les bets recommandés, les picks safe et les meilleurs edges par match à partir de `data/mass_prediction_nba`.

---

## 🔁 Workflow Feast / Prédictions live

Pour garantir que les features utilisées en prod correspondent exactement à celles du training (Feast), la routine suivante est recommandée avant de lancer le watcher ou une session de predictions manuelle :

1. **Rafraîchir les datasets + Feast**
   ```bash
   python -m src.feast.pipeline --seasons 2025-26 --targets POINT_TOTAL IS_WIN --materialize
   ```
   (ajustez la saison ou utilisez `make feast-refresh`).

2. **Pousser les matchs du jour dans l’online store** (optionnel mais recommandé si les futures rencontres ne sont pas encore matérialisées)
   ```bash
   python -m src.feast.push_online --match-date 2025-11-24
   # ou avec des IDs précis
   python -m src.feast.push_online --match-id 0022500123 --match-id 0022500124
   ```

3. **Lancer le watcher / les prédictions** (`make nba-mass-start`, notebook, etc.). `run_prediction_pipeline` ira automatiquement chercher les features via Feast et ne rejouera le pipeline offline qu’en dernier recours.

4. **Vérifier la parité offline/online** si besoin :
   ```bash
   python -m src.analysis.feature_consistency_cli \
     --run-json data/mass_prediction_nba/runs/2025-11-24/nba-run-20251124-190710-315764.json \
     --limit 5 --tolerance 1e-4
   ```
   Le script affiche, pour les matches fournis, l’écart maximal par colonne entre le dataset d’entraînement (offline) et les features servies par Feast.

---

## 📈 Objectif final

Créer un modèle capable de **prédire le vainqueur d’un match NBA** avec un pipeline automatisé prêt à l’usage dans un contexte de **paris sportifs**.
