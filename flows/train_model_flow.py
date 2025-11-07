from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from prefect import flow, task, get_run_logger

from nba_predictor.datasets.training import build_training_dataset
from nba_predictor.pipelines.training import TrainConfig, train_model


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@task
def materialize_training_dataset(ts: str) -> str:
    logger = get_run_logger()
    logger.info("Materializing training dataset", extra={"ingest_ts": ts})
    dataset_path = build_training_dataset(ingest_ts=ts)
    logger.info("Training dataset ready", extra={"path": str(dataset_path), "ingest_ts": ts})
    return str(dataset_path)


@task
def train_model_task(ts: str, dataset_path: str) -> str:
    logger = get_run_logger()
    logger.info("Training model", extra={"ingest_ts": ts, "dataset": dataset_path})
    config = TrainConfig(ingest_ts=ts, dataset_path=Path(dataset_path))
    run_id = train_model(config)
    logger.info("Training complete", extra={"run_id": run_id, "ingest_ts": ts})
    return run_id


@flow(name="train_model_flow")
def train_model_flow(*, ingest_ts: Optional[str] = None, dataset_path: Optional[str] = None) -> None:
    ts = ingest_ts or _timestamp()
    if dataset_path:
        train_model_task.submit(ts, dataset_path)
    else:
        dataset_future = materialize_training_dataset.submit(ts)
        train_model_task.submit(ts, dataset_future)


__all__ = ["train_model_flow"]
