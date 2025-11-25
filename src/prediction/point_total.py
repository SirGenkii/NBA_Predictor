from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import (
    DATA_BRONZE_MATCHES_DIR,
    MLFLOW_POINT_TOTAL_MODEL_NAME,
    MLFLOW_POINT_TOTAL_MODEL_STAGE,
    POINT_TOTAL_PRODUCTION_MODEL_KEY,
)
from src.datasets.base import run_pipeline
from src.datasets.recipes import DEFAULT_SILVER_FEATURE_STEPS
from src.feast.data_sources import FEAST_SILVER_EXPORT
from src.feast.loader import fetch_online_features
from src.feast.online_store import (
    POINT_TOTAL_FEATURE_SERVICE,
    POINT_TOTAL_FEATURE_VIEW,
    point_total_feature_columns,
    write_point_total_online_store,
)
from src.mlflow_utils import find_run_dir_with_artifact
from src.modeling.builders import build_point_total_trainer
from src.modeling.constants import CALIBRATION_ARTIFACT_PATH
from src.modeling.trainer import ModelTrainer
from src.prediction.data_refresh import refresh_recent_boxscores
from src.prediction.schemas import PredictionRequest
from src.utils import get_latest_file


EXPERIMENT_NAME = "point_total_regression"
MODEL_FILTER = f"params.model_key = '{POINT_TOTAL_PRODUCTION_MODEL_KEY}'"
BEST_PARAMS_PATH = Path("artifacts/point_total_best_params.json")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionOutput:
    home_team_id: str
    away_team_id: str
    match_date: str
    prediction: float
    sigma: float
    probabilities: dict
    model_path: Optional[str]
    model_bias: float
    model_uncertainty: float


def run_prediction_pipeline(
    requests: Sequence[PredictionRequest],
    *,
    pivots: Iterable[float],
    refresh_data: bool = True,
) -> List[PredictionOutput]:
    if not requests:
        return []

    seasons = {infer_season(req.match_date) for req in requests}
    if refresh_data:
        refresh_recent_boxscores(seasons)

    training_ref = _training_reference()
    features, entities = _collect_prediction_features(requests, training_ref.feature_names)
    model = training_ref.model
    preds = model.predict(features)
    sigma = training_ref.residual_std
    calibration = training_ref.calibration or {}

    pivots = list(pivots)
    outputs: List[PredictionOutput] = []
    for entity, pred_val in zip(entities, preds):
        request_obj = entity.request
        mean_val = float(pred_val)
        probs: dict[float, float] = {}
        for pivot in pivots:
            raw_prob = float(1 - norm.cdf(pivot, loc=mean_val, scale=sigma))
            probs[float(pivot)] = _apply_calibration(raw_prob, pivot, calibration)
        outputs.append(
            PredictionOutput(
                home_team_id=str(request_obj.home_team_id),
                away_team_id=str(request_obj.away_team_id),
                match_date=request_obj.match_date,
                prediction=mean_val,
                sigma=float(sigma),
                probabilities=probs,
                model_path=training_ref.model_path,
                model_bias=training_ref.residual_mean,
                model_uncertainty=training_ref.residual_std,
            )
        )
    return outputs


@dataclass
class _TrainingReference:
    feature_names: List[str]
    residual_std: float
    residual_mean: float
    model: object
    model_path: Optional[str]
    calibration: Optional[dict] = None


@lru_cache(maxsize=1)
def _training_reference() -> _TrainingReference:
    trainer = build_point_total_trainer(enable_registry=False)
    cache = _load_best_params()
    best_params = cache.get(POINT_TOTAL_PRODUCTION_MODEL_KEY) or cache.get("xgb")
    return _fit_or_load_model(trainer, best_params)


def _prepare_prediction_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    pred = df.reindex(columns=feature_names, fill_value=0.0)
    pred.index = df.index
    return pred


def _apply_calibration(prob: float, pivot: float, calibration: dict) -> float:
    if not calibration:
        return prob
    key = str(float(pivot))
    entry = calibration.get(key)
    if not entry:
        return prob
    xs = entry.get("x")
    ys = entry.get("y")
    if not xs or not ys:
        return prob
    calibrated = float(np.interp(prob, xs, ys))
    if calibrated < 0.0:
        return 0.0
    if calibrated > 1.0:
        return 1.0
    return calibrated


def _download_calibration_artifact(client: MlflowClient, run_id: str) -> Optional[dict]:
    try:
        local_path = Path(client.download_artifacts(run_id, CALIBRATION_ARTIFACT_PATH))
        if local_path.is_dir():
            local_path = local_path / Path(CALIBRATION_ARTIFACT_PATH).name
        if not local_path.exists():
            return None
        return json.loads(local_path.read_text())
    except Exception as exc:
        LOGGER.warning("Unable to download calibration artifact for run %s: %s", run_id, exc)
        return None


def _load_calibration_from_local(run_dir: Path) -> Optional[dict]:
    cal_path = Path(run_dir) / "artifacts" / CALIBRATION_ARTIFACT_PATH
    if not cal_path.exists():
        return None
    try:
        return json.loads(cal_path.read_text())
    except Exception as exc:
        LOGGER.warning("Unable to read calibration artifact at %s: %s", cal_path, exc)
        return None


def infer_season(match_date: str) -> str:
    dt = pd.to_datetime(match_date)
    year = dt.year
    if dt.month >= 10:
        start = year
    else:
        start = year - 1
    end = (start + 1) % 100
    return f"{start}-{end:02d}"


def _load_latest_bronze() -> pd.DataFrame:
    path = get_latest_file(DATA_BRONZE_MATCHES_DIR)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build_prediction_rows(
    base_df: pd.DataFrame,
    requests: Sequence[PredictionRequest],
    *,
    match_ids: Optional[Sequence[str]] = None,
) -> (pd.DataFrame, dict):
    template = {col: np.nan for col in base_df.columns}
    rows = []
    mapping = {}
    overrides = list(match_ids) if match_ids is not None else None
    for idx, req in enumerate(requests):
        season = infer_season(req.match_date)
        if overrides and idx < len(overrides):
            game_id = overrides[idx]
        else:
            game_id = _prediction_game_id(req)
        row = template.copy()
        row.update(
            {
                "GAME_ID": game_id,
                "GAME_DATE": pd.to_datetime(req.match_date),
                "SEASON": season,
                "HOME_TEAM_ID": str(req.home_team_id),
                "AWAY_TEAM_ID": str(req.away_team_id),
                "HOME_IS_WIN": 0,
                "AWAY_IS_WIN": 0,
            }
        )
        rows.append(row)
        mapping[game_id] = req
    return pd.DataFrame(rows), mapping


def _prediction_game_id(req: PredictionRequest) -> str:
    return f"PRED_{req.home_team_id}_{req.away_team_id}_{req.match_date}"


def _build_trainer() -> ModelTrainer:
    # Deprecated alias kept for backward compatibility within this module.
    return build_point_total_trainer(enable_registry=False)


def _load_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        return {}
    cache = json.loads(BEST_PARAMS_PATH.read_text())
    normalized = {}
    for key, entry in cache.items():
        params = entry.get("params", entry)
        if key == "stacking":
            params = _normalize_stacking_params(params)
        normalized[key] = params
    return normalized


def _normalize_stacking_params(params: dict) -> dict:
    """Map short aliases coming from Optuna cache to valid StackingRegressor parameter names."""

    mapping = {
        "stack_alpha": "final_estimator__alpha",
        "stack_l1_ratio": "final_estimator__l1_ratio",
    }
    normalized = {}
    for key, value in params.items():
        normalized_key = mapping.get(key, key)
        normalized[normalized_key] = value
    return normalized


def _fit_or_load_model(trainer: ModelTrainer, overrides: dict | None) -> _TrainingReference:
    model = None
    model_path: Optional[str] = None
    calibration: Optional[dict] = None

    if MLFLOW_POINT_TOTAL_MODEL_NAME:
        registry_uri = f"models:/{MLFLOW_POINT_TOTAL_MODEL_NAME}/{MLFLOW_POINT_TOTAL_MODEL_STAGE}"
        try:
            model = mlflow.sklearn.load_model(registry_uri)
            model_path = registry_uri
            LOGGER.info("Loaded point_total model from MLflow Registry (%s)", registry_uri)
            client = MlflowClient()
            versions = client.get_latest_versions(
                MLFLOW_POINT_TOTAL_MODEL_NAME, [MLFLOW_POINT_TOTAL_MODEL_STAGE]
            )
            if versions:
                calibration = _download_calibration_artifact(client, versions[0].run_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Unable to load MLflow registry model (%s): %s", registry_uri, exc)
            model = None

    if model is None:
        try:
            run_dir = find_run_dir_with_artifact(
                target="POINT_TOTAL",
                experiment_name=EXPERIMENT_NAME,
                filter_string=MODEL_FILTER,
                artifact_subdir="model",
                max_results=100,
            )
            model = mlflow.sklearn.load_model(str(run_dir / "artifacts/model"))
            model_path = str(run_dir)
            calibration = _load_calibration_from_local(run_dir)
            LOGGER.info("Loaded point_total model from MLflow run %s", model_path)
        except Exception as exc:
            LOGGER.warning("Unable to load MLflow point_total model from runs: %s", exc)
            model = None

    if model is None:
        target_key = POINT_TOTAL_PRODUCTION_MODEL_KEY or "stacking"
        try:
            pipeline = trainer._build_model(target_key, overrides=overrides)
        except ValueError:
            pipeline = trainer._build_model("stacking", overrides=overrides)
        pipeline.fit(trainer.X_train, trainer.y_train)
        model = pipeline
        LOGGER.info("Trained point_total %s model locally (no MLflow artifact).", target_key)

    preds = model.predict(trainer.X_train)
    residuals = trainer.y_train - preds
    residual_std = float(np.std(residuals))
    residual_mean = float(np.mean(residuals))
    return _TrainingReference(
        feature_names=trainer._feature_names,
        residual_std=residual_std,
        residual_mean=residual_mean,
        model=model,
        model_path=model_path,
        calibration=calibration,
    )


@dataclass(frozen=True)
class _RequestEntity:
    request: PredictionRequest
    match_id: str
    event_timestamp: pd.Timestamp


def _collect_prediction_features(
    requests: Sequence[PredictionRequest],
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, List[_RequestEntity]]:
    entities = _build_request_entities(requests)
    feature_rows = _fetch_or_materialize_features(entities)
    ordered_rows = []
    for entity in entities:
        row = feature_rows.get(entity.match_id)
        if row is None:
            raise RuntimeError(f"Aucune feature Feast disponible pour le match_id {entity.match_id}")
        ordered_rows.append(row)
    feature_df = pd.DataFrame(ordered_rows)
    feature_df = _prepare_prediction_features(feature_df, list(feature_names))
    return feature_df, entities


def _build_request_entities(requests: Sequence[PredictionRequest]) -> List[_RequestEntity]:
    lookup = _match_lookup_map()
    entities: List[_RequestEntity] = []
    for req in requests:
        match_date = pd.to_datetime(req.match_date)
        key = (int(req.home_team_id), int(req.away_team_id), match_date.date())
        match_id = lookup.get(key) or _prediction_game_id(req)
        entities.append(
            _RequestEntity(
                request=req,
                match_id=str(match_id),
                event_timestamp=match_date,
            )
        )
    return entities


def _fetch_or_materialize_features(
    entities: Sequence[_RequestEntity],
) -> Dict[str, pd.Series]:
    if not entities:
        return {}
    fetched, missing = _fetch_features_from_feast(entities)
    if missing:
        computed = _compute_features_for_entities(missing)
        fetched.update(computed)
    still_missing = [ent.match_id for ent in entities if ent.match_id not in fetched]
    if still_missing:
        raise RuntimeError(f"Impossible de récupérer les features pour {still_missing}")
    return fetched


def _fetch_features_from_feast(
    entities: Sequence[_RequestEntity],
) -> Tuple[Dict[str, pd.Series], List[_RequestEntity]]:
    match_ids = [ent.match_id for ent in entities]
    request_df = pd.DataFrame({"match_id": match_ids})
    online_df = pd.DataFrame()
    try:
        online_df = fetch_online_features(
            request_df=request_df,
            feature_service=POINT_TOTAL_FEATURE_SERVICE,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Echec de récupération des features online Feast: %s", exc)
    rows: Dict[str, pd.Series] = {}
    if not online_df.empty:
        online_df["match_id"] = online_df["match_id"].astype(str)
        feature_cols = [col for col in online_df.columns if col != "match_id"]
        for _, row in online_df.iterrows():
            match_id = row["match_id"]
            data = row[feature_cols]
            if feature_cols and data.isna().all():
                continue
            rows[match_id] = data
    missing = [ent for ent in entities if ent.match_id not in rows]
    return rows, missing


def _compute_features_for_entities(
    entities: Sequence[_RequestEntity],
) -> Dict[str, pd.Series]:
    if not entities:
        return {}
    base_df = _load_latest_bronze()
    base_df["GAME_DATE"] = pd.to_datetime(base_df["GAME_DATE"])
    feature_cols = point_total_feature_columns()
    if not feature_cols:
        LOGGER.warning("FeatureView %s vide: skip materialization.", POINT_TOTAL_FEATURE_VIEW)
        return {}

    grouped: Dict[pd.Timestamp, List[_RequestEntity]] = {}
    for ent in entities:
        grouped.setdefault(ent.event_timestamp.normalize(), []).append(ent)

    rows: Dict[str, pd.Series] = {}
    for event_ts, batch in grouped.items():
        history_mask = base_df["GAME_DATE"] < event_ts
        history = base_df.loc[history_mask].copy()
        match_ids = [ent.match_id for ent in batch]
        requests = [ent.request for ent in batch]
        pred_rows, _ = _build_prediction_rows(base_df, requests, match_ids=match_ids)
        augmented = pd.concat([history, pred_rows], ignore_index=True, sort=False)
        silver = run_pipeline(augmented, DEFAULT_SILVER_FEATURE_STEPS)
        pred_mask = silver["GAME_ID"].astype(str).isin(match_ids)
        pred_silver = silver.loc[pred_mask].copy().reset_index(drop=True)
        if pred_silver.empty:
            LOGGER.warning("Aucune feature calculée pour les matches %s", match_ids)
            continue
        pred_silver["match_id"] = pred_silver["GAME_ID"].astype(str)
        missing_cols = [col for col in feature_cols if col not in pred_silver.columns]
        for col in missing_cols:
            pred_silver[col] = np.nan
        write_point_total_online_store(pred_silver)
        for _, row in pred_silver.iterrows():
            rows[row["match_id"]] = row[feature_cols]
    return rows


@lru_cache(maxsize=1)
def _match_lookup_map() -> Dict[Tuple[int, int, object], str]:
    path = Path(FEAST_SILVER_EXPORT)
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, columns=["match_id", "HOME_TEAM_ID", "AWAY_TEAM_ID", "GAME_DATE"])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Impossible de charger l'export silver Feast: %s", exc)
        return {}
    df["HOME_TEAM_ID"] = pd.to_numeric(df["HOME_TEAM_ID"], errors="coerce").astype("Int64")
    df["AWAY_TEAM_ID"] = pd.to_numeric(df["AWAY_TEAM_ID"], errors="coerce").astype("Int64")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    mapping: Dict[Tuple[int, int, object], str] = {}
    for _, row in df.iterrows():
        home = row["HOME_TEAM_ID"]
        away = row["AWAY_TEAM_ID"]
        date = row["GAME_DATE"]
        match_id = row["match_id"]
        if pd.isna(home) or pd.isna(away) or pd.isna(date) or pd.isna(match_id):
            continue
        mapping[(int(home), int(away), date)] = str(match_id)
    return mapping
