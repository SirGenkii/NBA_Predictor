from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import structlog

from nba_predictor.datasets.training import build_scoring_payload

logger = structlog.get_logger("nba_predictor.pipelines.prediction")

DEFAULT_MODEL_URI = os.getenv("NBA_PREDICTOR_MODEL_URI", "models:/nba_predictor/Production")


def predict_matches(
    *,
    match_date: str,
    ingest_ts: str | None = None,
    model_uri: Optional[str] = None,
    payload_path: Optional[str | Path] = None,
) -> Path:
    """
    Load the latest scoring payload, apply the MLflow model and persist predictions to gold/scoring_payloads.
    """

    if payload_path is None:
        payload_path = build_scoring_payload(match_date=match_date, ingest_ts=ingest_ts)
    payload_df = pd.read_parquet(payload_path)

    feature_cols = [c for c in payload_df.columns if c.endswith("_diff")]
    if not feature_cols:
        raise ValueError("No feature columns with '_diff' suffix found in scoring payload.")

    model_uri = model_uri or DEFAULT_MODEL_URI
    logger.info("predict_matches_load_model", model_uri=model_uri)
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    probabilities = model.predict(payload_df[feature_cols])
    if probabilities.ndim > 1:
        probabilities = probabilities[:, 1]

    predictions = payload_df.copy()
    predictions["prediction_home_win_proba"] = probabilities
    predictions["prediction_home_win"] = (probabilities >= 0.5).astype(int)

    output_path = payload_path.with_name(payload_path.stem + "_predictions.parquet")
    predictions.to_parquet(output_path, index=False)

    logger.info(
        "predict_matches_complete",
        rows=len(predictions),
        output=str(output_path),
    )
    return output_path


__all__ = ["predict_matches"]
