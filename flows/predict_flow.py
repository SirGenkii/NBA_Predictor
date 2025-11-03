from __future__ import annotations

from datetime import datetime
from typing import Optional

from prefect import flow, task, get_run_logger


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@task
def load_latest_model() -> None:
    get_run_logger().info("Loading latest production model")
    # Placeholder: pull model from MLflow registry.


@task
def fetch_features_task(match_date: str) -> None:
    get_run_logger().info("Fetching features for predictions", match_date=match_date)
    # Placeholder: query Feast or offline store for upcoming matches.


@task
def score_matches_task() -> None:
    get_run_logger().info("Scoring matches")
    # Placeholder: apply model to features and persist predictions.


@flow(name="predict_flow")
def predict_flow(*, match_date: Optional[str] = None) -> None:
    match_date = match_date or datetime.utcnow().date().isoformat()
    model = load_latest_model.submit()
    features = fetch_features_task.submit(match_date, wait_for=[model])
    score_matches_task.submit(wait_for=[features])


__all__ = ["predict_flow"]
