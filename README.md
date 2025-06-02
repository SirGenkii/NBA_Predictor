
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
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt

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

## 📈 Objectif final

Créer un modèle capable de **prédire le vainqueur d’un match NBA** avec un pipeline automatisé prêt à l’usage dans un contexte de **paris sportifs**.
