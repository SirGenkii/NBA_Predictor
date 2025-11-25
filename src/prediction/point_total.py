from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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
from src.datasets.recipes import (
    DEFAULT_SILVER_FEATURE_STEPS,
    gold_steps_for_target,
)
from src.mlflow_utils import find_run_dir_with_artifact
from src.modeling.config import DatasetConfig, TrainingConfig
from src.modeling.data import load_dataset, prepare_features
from src.modeling.builders import build_point_total_trainer, point_total_bundle
from src.modeling.constants import CALIBRATION_ARTIFACT_PATH
from src.modeling.trainer import ModelTrainer
from src.prediction.data_refresh import refresh_recent_boxscores
from src.prediction.schemas import PredictionRequest
from src.utils import get_latest_file


EXPERIMENT_NAME = "point_total_regression"
MODEL_FILTER = f"params.model_key = '{POINT_TOTAL_PRODUCTION_MODEL_KEY}'"
BEST_PARAMS_PATH = Path("artifacts/point_total_best_params.json")
PREDICTION_ID_COL = "__prediction_game_id"
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

    base_df = _load_latest_bronze()
    base_df["GAME_DATE"] = pd.to_datetime(base_df["GAME_DATE"])
    pred_rows, request_map = _build_prediction_rows(base_df, requests)
    game_ids = list(request_map.keys())
    augmented = pd.concat([base_df, pred_rows], ignore_index=True, sort=False)

    silver = run_pipeline(augmented, DEFAULT_SILVER_FEATURE_STEPS)
    pred_mask = silver["GAME_ID"].isin(game_ids)
    pred_silver = silver.loc[pred_mask].copy()
    pred_silver = pred_silver.reset_index(drop=True)
    pred_ids = pred_silver["GAME_ID"].astype(str).reset_index(drop=True)

    pred_gold = run_pipeline(pred_silver, gold_steps_for_target("POINT_TOTAL"))
    pred_gold = pred_gold.reset_index(drop=True)
    pred_gold[PREDICTION_ID_COL] = pred_ids

    training_ref = _training_reference()
    features = _prepare_prediction_features(pred_gold, training_ref.feature_names)
    model = training_ref.model
    preds = model.predict(features)
    sigma = training_ref.residual_std
    calibration = training_ref.calibration or {}

    result_map = {}
    pivots = list(pivots)
    for (idx, row), pred_val in zip(pred_gold.iterrows(), preds):
        game_id = row.get(PREDICTION_ID_COL)
        if pd.isna(game_id):
            raise KeyError("Prediction row missing internal GAME_ID reference.")
        request_obj = request_map[game_id]
        mean_val = float(pred_val)
        probs: dict[float, float] = {}
        for pivot in pivots:
            raw_prob = float(1 - norm.cdf(pivot, loc=mean_val, scale=sigma))
            probs[float(pivot)] = _apply_calibration(raw_prob, pivot, calibration)
        result_map[game_id] = PredictionOutput(
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
    return [result_map[_prediction_game_id(req)] for req in requests]


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
    base_df: pd.DataFrame, requests: Sequence[PredictionRequest]
) -> (pd.DataFrame, dict):
    template = {col: np.nan for col in base_df.columns}
    rows = []
    mapping = {}
    for req in requests:
        season = infer_season(req.match_date)
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
