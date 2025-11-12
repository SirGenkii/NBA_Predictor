from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import DATA_BRONZE_MATCHES_DIR
from src.datasets.base import run_pipeline
from src.datasets.recipes import (
    DEFAULT_SILVER_FEATURE_STEPS,
    gold_steps_for_target,
)
from src.mlflow_utils import get_latest_run_dir
from src.modeling.config import DatasetConfig, TrainingConfig
from src.modeling.data import load_dataset, prepare_features
from src.modeling.trainer import ModelTrainer
from src.prediction.data_refresh import refresh_recent_boxscores
from src.prediction.schemas import PredictionRequest
from src.utils import get_latest_file


EXPERIMENT_NAME = "point_total_regression"
STACKING_FILTER = "params.model_key = 'stacking'"
BEST_PARAMS_PATH = Path("artifacts/point_total_best_params.json")
PREDICTION_ID_COL = "__prediction_game_id"


@dataclass(frozen=True)
class PredictionOutput:
    home_team_id: str
    away_team_id: str
    match_date: str
    prediction: float
    sigma: float
    probabilities: dict


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

    result_map = {}
    pivots = list(pivots)
    for (idx, row), pred_val in zip(pred_gold.iterrows(), preds):
        game_id = row.get(PREDICTION_ID_COL)
        if pd.isna(game_id):
            raise KeyError("Prediction row missing internal GAME_ID reference.")
        request_obj = request_map[game_id]
        mean_val = float(pred_val)
        probs = {
            float(pivot): float(1 - norm.cdf(pivot, loc=mean_val, scale=sigma))
            for pivot in pivots
        }
        result_map[game_id] = PredictionOutput(
            home_team_id=str(request_obj.home_team_id),
            away_team_id=str(request_obj.away_team_id),
            match_date=request_obj.match_date,
            prediction=mean_val,
            sigma=float(sigma),
            probabilities=probs,
        )
    return [result_map[_prediction_game_id(req)] for req in requests]


@dataclass
class _TrainingReference:
    feature_names: List[str]
    residual_std: float
    model: object


@lru_cache(maxsize=1)
def _training_reference() -> _TrainingReference:
    trainer = _build_trainer()
    best_params = _load_best_params().get("stacking")
    return _fit_or_load_model(trainer, best_params)


def _prepare_prediction_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    pred = df.reindex(columns=feature_names, fill_value=0.0)
    pred.index = df.index
    return pred


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
    dataset_cfg = DatasetConfig(
        target="POINT_TOTAL",
        gold_pattern="gold_dataset_point_total_*.parquet",
        use_feast=True,
        feast_feature_service="point_total_service",
    )
    training_cfg = TrainingConfig(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri="file:./mlruns",
        test_size=0.2,
        random_state=42,
        stratify=False,
        task_type="regression",
        pivot_value=220.0,
        enable_learning_curve=False,
        pivot_values=[
            218.5,
            219.5,
            220.5,
            221.5,
            222.5,
            223.5,
            224.5,
            225.5,
            226.5,
            227.5,
            228.5,
            229.5,
            230.5,
            231.5,
            232.5,
            233.5,
            234.5,
            235.5,
            236.5,
            237.5,
            238.5,
            239.5,
            240.5,
            241.5,
            242.5,
            243.5,
            244.5,
        ],
        enable_sigma_model=True,
        min_sigma=6.0,
    )
    return ModelTrainer(dataset_cfg, training_cfg)


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
    try:
        run_dir = get_latest_run_dir(
            target="POINT_TOTAL",
            experiment_name=EXPERIMENT_NAME,
            filter_string=STACKING_FILTER,
        )
        model = mlflow.sklearn.load_model(str(run_dir / "artifacts/model"))
    except Exception:
        model = None

    if model is None:
        pipeline = trainer._build_model("stacking", overrides=overrides)
        pipeline.fit(trainer.X_train, trainer.y_train)
        model = pipeline

    preds = model.predict(trainer.X_train)
    residual_std = float(np.std(trainer.y_train - preds))
    return _TrainingReference(
        feature_names=trainer._feature_names,
        residual_std=residual_std,
        model=model,
    )
