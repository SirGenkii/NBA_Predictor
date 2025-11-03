from __future__ import annotations

from datetime import datetime
from typing import Optional

from prefect import flow, task, get_run_logger

from nba_predictor.datasets.training import build_scoring_payload
from nba_predictor.pipelines.prediction import predict_matches


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@task
def build_payload_task(match_date: str, ingest_ts: str) -> str:
    logger = get_run_logger()
    logger.info("Building scoring payload", match_date=match_date, ingest_ts=ingest_ts)
    payload_path = build_scoring_payload(match_date=match_date, ingest_ts=ingest_ts)
    return str(payload_path)


@task
def score_matches_task(match_date: str, payload_path: str, model_uri: Optional[str]) -> str:
    logger = get_run_logger()
    logger.info("Scoring matches", match_date=match_date, payload=payload_path, model_uri=model_uri)
    output = predict_matches(match_date=match_date, payload_path=payload_path, model_uri=model_uri)
    logger.info("Predictions stored", path=str(output))
    return str(output)


@flow(name="predict_flow")
def predict_flow(*, match_date: Optional[str] = None, model_uri: Optional[str] = None) -> None:
    match_date = match_date or datetime.utcnow().date().isoformat()
    ingest_ts = _timestamp()
    payload_future = build_payload_task.submit(match_date, ingest_ts)
    score_matches_task.submit(match_date, payload_future, model_uri)


__all__ = ["predict_flow"]
