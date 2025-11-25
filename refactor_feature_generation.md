# Refactor feature generation (offline vs. online parity)

## Contexte
- Le dataset `gold` / `silver` utilisé à l'entraînement provient de la pipeline Feast (`DEFAULT_SILVER_FEATURE_STEPS` + `gold_steps_for_target`). Les features rolls (pace/points diff, H2H, rest, Elo…) sont calculées à partir de la vision "team" et repassées sur les match rows.
- Le pipeline "online" actuel (`run_prediction_pipeline`) part du bronze cru (`_load_latest_bronze`) puis rejoue une version simplifiée de ces steps. On injecte uniquement les colonnes `feature_names`. Résultat : des features essentielles (`PACE_DIFF_*`, `POINTS_DIFF_EXPECTED_*`, etc.) sont nulles ou très différentes car on n’a pas les mêmes données intermédiaires.
- Conséquence : les prédictions live n’ont rien à voir avec celles utilisées à l’entraînement et les calibrations « safe » deviennent totalement biaisées.

## Objectif
Garantir que les features consommées par les prédictions live sont exactement celles produites offline. Bonne pratique : basculer la prédiction online sur Feast (feature store) pour utiliser la même définition des features.

## Contraintes techniques / existant
1. Le pipeline offline (silver/gold) alimente déjà Feast : `src/feast/pipeline.py` construit `data/feast_sources/silver_latest.parquet` + `gold_dataset_*.parquet` et appelle `feast apply/materialize`.
2. `FeatureView` `point_total_features` expose l’ensemble des colonnes silver. Le `FeatureService` `point_total_service` pointe dessus ; `fetch_historical_features` et `fetch_online_features` existent.
3. `PredictionRequest` crée encore des lignes synthétiques dans `_build_prediction_rows` puis rejoue `run_pipeline`. Cette partie doit être remplacée par un appel à Feast.

## Plan d’implémentation
1. **Refactor `run_prediction_pipeline`**
   - a. Construire un DataFrame `request_df` contenant `match_id` + `event_timestamp` (timestamp = date du match). `match_id` peut être `PredictionRequest.game_id`.
   - b. Appeler `fetch_online_features(request_df, feature_service="point_total_service")`. Cette requête renvoie exactement la ligne du silver exportée dans Feast.
   - c. Par sécurité, si certaines requêtes ne sont pas disponibles dans l’online store (match du futur), fallback sur `fetch_historical_features` ou sur les features calculées via la pipeline (option B pour la transition).
   - d. Continuer avec `_prepare_prediction_features` mais seulement pour s’assurer que l’ordre des colonnes correspond aux `feature_names` enregistrées dans le modèle.

2. **Alimentation du store pour les matchs futurs**
   - a. Pendant la journée (commande `python -m src.feast.pipeline` ou un script dédié), générer les features silver des matchs à venir (ex : ceux fournis par l’OCR la veille) et les publier dans le store via `FeatureStore.write_to_online_store`.
   - b. Conserver un script utilitaire `python -m src.feast.backfill_online --match-id ...` pour forcer le push manuel d’un match en cas d’urgence.

3. **Notebooks & tests**
   - a. Mettre à jour `11_feature_consistency.ipynb` pour récupérer les features via Feast (historical). Pour simuler le passé, on pourra appeler `fetch_historical_features` avec un `entity_df` limité à `event_timestamp < match_date`. On obtient ainsi les mêmes valeurs qu’en live.
   - b. Ajouter un test simple : pour un match historisé, comparer `collect_offline_features` (dataset gold) et `collect_online_features` (Feast). L’écart doit être < tolérance.

4. **Nettoyage**
   - a. Une fois que les prédictions live utilisent Feast, on pourra supprimer les steps `run_pipeline`/`DEFAULT_SILVER_FEATURE_STEPS` dans `src/prediction/point_total.py` (ou seulement les conserver comme fallback).
   - b. Documenter le besoin de lancer `python -m src.feast.pipeline --materialize` avant la journée pour pousser les matches à venir dans le store.

5. **Impacts sur le watcher/UI**
   - a. Le watcher devient plus simple : `PredictionEngine` n’a plus besoin de recréer le silver ergo moins de CPU/mémoire.
   - b. Les features étant alignées, les calibrations et stratégies (surtout `safe`) deviendront fiables.

## Étapes détaillées (TODO)
- [x] Introduire un helper `fetch_features_for_requests(requests: Sequence[PredictionRequest])` qui interroge Feast et gère les fallback.
- [x] Modifier `run_prediction_pipeline` pour utiliser ce helper (remplacer `_build_prediction_rows` + `run_pipeline`).
- [x] Ajouter un script `src/feast/push_online.py` qui prend un CSV de matchs futurs et écrit les features correspondantes dans l’online store (via `FeatureStore.write_to_online_store`).
- [x] Mettre à jour `feature_consistency.py` pour comparer gold vs Feast-historical.
- [x] Documenter le flux (README / wiki) : “Chaque matin, lancer `python -m src.feast.pipeline --materialize` pour rafraîchir les features futures, puis lancer le watcher.”

## Routine quotidienne proposée
1. `python -m src.feast.pipeline --seasons <SAISON> --materialize` pour régénérer bronze/silver/gold et mettre le Feature Store à jour.
2. `python -m src.feast.push_online --match-date YYYY-MM-DD` (ou `--match-id ...`) pour pousser explicitement les matches du soir dans l’online store si besoin (utile quand les futures rencontres n’ont pas encore été matérialisées par Feast).
3. Lancer le watcher (`make nba-mass-start` / API) : `run_prediction_pipeline` récupère désormais les features directement via Feast, et ne retombe sur la pipeline locale que si une ligne est absente.
