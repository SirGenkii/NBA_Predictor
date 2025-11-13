# NBA Mass Prediction Roadmap

Plan détaillé pour remplacer l’ancienne stack `src/mass_prediction` (héritée du projet ATP) par un moteur dédié aux pivots NBA Over/Under. Le code existant dans `src/mass_prediction` reste comme exemple et **ne doit pas être réutilisé**. La nouvelle implémentation vivra dans **`src/nba_mass_prediction/`** (nom retenu pour bien séparer les deux mondes).

---

## 1. Objectifs Clés
- Supporter un **watcher mass-processing** qui écoute un dossier de screenshots (1 screenshot = 1 match) et déclenche les prédictions dès qu’un fichier arrive.
- **Redémarrer l’engine** = rafraîchir les données NBA (boxscores, Feast materialization) une seule fois, puis servir les prédictions en mémoire sans recalcul complet à chaque screenshot.
- Calculer automatiquement pour chaque pivot: `prob_over`, `prob_under`, cotes justes, edge, Kelly fraction.
- Produire des **artefacts structurés (JSON/CSV)** par run pour des usages futurs (API, Discord, dashboard). Pas de notifications à ce stade.
- Anticiper d’autres sources d’input que l’OCR (JSON direct, API bookmaker) en gardant l’interface générique.

---

## 2. Architecture Cible
```
mass_prediction_watcher.py (entrée serveur)
        │
        ▼
src/nba_mass_prediction/
 ├─ adapters/          # OCR, JSON, etc.
 ├─ resolver/          # mapping équipes + dates
 ├─ engine/            # wrapper Feast + modèle point_total
 ├─ strategy/          # probas, fair odds, Kelly, règles EV
 └─ runner.py          # orchestration screenshot → stratégie → export
```

### Flux
1. **Watcher** détecte un fichier → vérifie settling time.
2. **Adapter** lit la source (ex: OCR OpenAI) → produit un `MatchPayload` (home/away textuels, date, pivots, odds, métadonnées screenshot).
3. **Resolver** mappe les noms équipes → `TEAM_ID`, fixe la date (override possible), vérifie la cohérence.
4. **Engine** (caché en mémoire) appelle `run_prediction_pipeline` pour toutes les `(home_id, away_id, date)` uniques + la liste fusionnée des pivots.
5. **Strategy** calcule proba over/under, fair odds, edge, Kelly, et classe chaque pivot (`bet_over`, `bet_under`, `watch`, `skip`…).
6. **Runner** écrit un fichier JSON (et optionnellement CSV/Parquet) avec OCR brut, données normalisées, sortie modèle, décision stratégique, puis déplace le screenshot dans `processed/` ou `error/`.

---

## 3. Étapes d’Implémentation

### Phase A – Préparation
1. **Créer `src/nba_mass_prediction/`** avec une arborescence vide (`__init__.py`, sous-dossiers `adapters`, `resolver`, `engine`, `strategy`, `runner`, `schemas`).
2. **Nettoyer `src/config.py`** :
   - Ajouter un bloc “NBA Mass Prediction” (voir §5).
   - Laisser les constantes ATP en place pour rétrocompatibilité, mais documenter celles utilisées par le nouveau moteur.
3. **Mettre à jour `.env`** avec les nouvelles variables (chemins, toggles watcher, clés OCR s’il faut les différencier).

### Phase B – Engine & Data Refresh
1. Implémenter `PredictionEngine` (`src/nba_mass_prediction/engine/core.py`) :
   - Hook `refresh()` qui lance : ingestion NBA (scripts existants), Feast materialization, chargement modèle point_total (via `build_point_total_trainer` ou `mlflow`).
   - Méthode `score_batch(requests, pivots)` qui dedup `(home_id, away_id, date)` et renvoie `PredictionOutput` (mean, sigma, probas par pivot).
2. Prévoir un cache thread-safe pour les artefacts (model, features, residual std) afin d’éviter la relecture par screenshot.

### Phase C – Input & Resolver
1. `adapters/ocr_adapter.py` :
   - Prompt OpenAI Vision orienté NBA Unibet (équipes, date, `markets` [{pivot, over_odds, under_odds, label, bookmaker}]).
   - Sortie stricte JSON → convertie en `MatchPayload`.
2. `resolver/team_resolver.py` :
   - Génère un mapping nom → `TEAM_ID` depuis le dernier dataset bronze stocké dans la config sous DATA_TEAMS_DIR.
   - Gère les alias et fuzzy search (`LA Lakers`, `Lakers`, `Los Angeles Lakers`…) + log des cas ambigus.
   - Expose un override CLI/env (`--match-date`, `--home-team-id`, etc.) pour debug.

### Phase D – Strategy & Runner
1. `strategy/pivot_strategy.py` :
   - Fonctions pures: `compute_probabilities(mean, sigma, pivot)`, `kelly_for_side(prob, odds)`.
   - Configurable thresholds (edge min, prob min, odds range, scaling Kelly).
2. `runner.py` :
   - Coller au pattern de l’ancien `runner.process_screenshot` mais en simplifié (pas de DB jobs, pas de multiprocessing):
     - OCR → resolver → engine → strategy → export.
     - Gestion des exceptions (erreur OCR = screenshot → `error/`, log JSON).
   - Ecriture d’un artefact par run (`NBA_MASS_PREDICTION_EXPORT_DIR/run-YYYYMMDD-HHMMSS.json`).

### Phase E – Watcher & CLI
1. Adapter `src/mass_prediction/scripts/mass_prediction_watcher.py` :
   - Importer le nouveau runner (`from src.nba_mass_prediction.runner import process_screenshot`).
   - Créer l’engine NBA au démarrage (mais conserver l’écriture d’état `write_state`).
   - Supprimer les références aux anciens modules (Discord, odds API, DB jobs).
2. Prévoir un mode `--run-once` + boucle infinie similaire à l’actuel.

### Phase F – Tests & Validation
1. Tests unitaires:
   - OCR parsing (snapshots JSON).
   - Resolver (alias connus, erreurs d’équipe).
   - Strategy (probabilité, edge, Kelly) avec valeurs forcées.
   - Runner (mock OCR + engine) → fichier JSON attendu.
2. Tests intégrés:
   - Lancer le watcher en mode `--run-once` sur un dossier fake pour valider tout le flux.

---

## 4. Livrables / Artefacts
- `src/nba_mass_prediction/` complet.
- Nouveaux configs + variables `.env`.
- Scripts Makefile/CLI pour lancer:
  - `python -m src.mass_prediction.scripts.mass_prediction_watcher --run-once`
  - `python -m src.nba_mass_prediction.runner --input path/to/image.png` (optionnel pour debug direct).
- Dossier d’export (ex: `data/mass_prediction_nba/runs/YYYY-MM-DD/`).
- Documentation utilisateur (peut être une section du README principal ou une nouvelle page `docs/nba_mass_prediction.md` après implémentation).

---

## 5. Config & .env à prévoir
Ajouter dans `src/config.py` (sections proposées) :
```python
# NBA mass prediction
NBA_MASS_PREDICTION_DIR = os.path.join(DATA_DIR, "mass_prediction_nba")
NBA_MASS_SCREENSHOT_DIR = os.path.join(NBA_MASS_PREDICTION_DIR, "screenshots")
NBA_MASS_SCREENSHOT_PROCESSED_DIR = os.path.join(NBA_MASS_SCREENSHOT_DIR, "processed")
NBA_MASS_SCREENSHOT_ERROR_DIR = os.path.join(NBA_MASS_SCREENSHOT_DIR, "error")
NBA_MASS_LOG_DIR = os.path.join(NBA_MASS_PREDICTION_DIR, "logs")
NBA_MASS_RUNS_DIR = os.path.join(NBA_MASS_PREDICTION_DIR, "runs")
NBA_MASS_PAYLOAD_DIR = os.path.join(NBA_MASS_PREDICTION_DIR, "payloads")
NBA_MASS_DEFAULT_MAX_WORKERS = int(os.getenv("NBA_MASS_DEFAULT_MAX_WORKERS", 2))

NBA_OCR_OPENAI_MODEL = os.getenv("NBA_OCR_OPENAI_MODEL", "gpt-4o-mini")
NBA_OCR_OPENAI_API_KEY = os.getenv("NBA_OCR_OPENAI_API_KEY")  # fallback sur OPENAI_API_KEY si None

NBA_MASS_EDGE_THRESHOLD = float(os.getenv("NBA_MASS_EDGE_THRESHOLD", 0.01))
NBA_MASS_KELLY_SCALING = float(os.getenv("NBA_MASS_KELLY_SCALING", 0.66))
```
Variables `.env` associées à renseigner:
- `NBA_MASS_DEFAULT_MAX_WORKERS`
- `NBA_OCR_OPENAI_MODEL`
- `NBA_OCR_OPENAI_API_KEY` (ou utiliser `OPENAI_API_KEY`)
- `NBA_MASS_EDGE_THRESHOLD`
- `NBA_MASS_KELLY_SCALING`
- (Optionnel) `NBA_MASS_MATCH_DATE_OVERRIDE`, `NBA_MASS_FORCE_HOME_TEAM_ID`, etc., si des overrides sont nécessaires.

Garder `MASS_PREDICTION_*` pour l’ancien système tant que le code coexiste.

---

## 6. Points Restants à Valider
1. **Source alias équipes** : auto-générer depuis le dataset bronze ou maintenir un YAML versionné ?
2. **Formats odds** : uniquement décimal (Unibet) ou prévoir fractionnel/US si on supportera d’autres bookmakers ?
3. **Stockage résultat** : JSON suffisant ou faut-il append dans une base/CSV pour backtesting immédiat ?
4. **Scaling Kelly** : par défaut `2/3`, mais peut-on exposer plusieurs presets dans la stratégie ?
5. **Input alternatifs** : doit-on spécifier dès maintenant une API JSON, ou se contenter du watcher OCR en v1 ?

Clarifier ces points avant de coder permettra de verrouiller la configuration et la structure finale.

---

Suivre cette roadmap garantit que le nouveau moteur NBA est isolé, extensible et contrôlable via l’actuel watcher, tout en réutilisant le pipeline Feast/point_total existant pour la partie modèle.

