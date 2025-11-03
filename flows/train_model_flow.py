from __future__ import annotations

from datetime import datetime
from typing import Optional

from prefect import flow, task, get_run_logger


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@task
def materialize_training_dataset(ts: str) -> None:
    get_run_logger().info("Materializing training dataset", ingest_ts=ts)
    # Placeholder: load features via Feast or offline parquet when dataset ready.


@task
def train_model_task(ts: str) -> None:
    get_run_logger().info("Training model", ingest_ts=ts)
    # Placeholder: implement training pipeline (split, train, MLflow logging).


@task
def register_model_task() -> None:
    get_run_logger().info("Registering model to MLflow registry")
    # Placeholder: promote latest run to registry stage.


@flow(name="train_model_flow")
def train_model_flow(*, ingest_ts: Optional[str] = None) -> None:
    ts = ingest_ts or _timestamp()
    dataset = materialize_training_dataset.submit(ts)
    model = train_model_task.submit(ts, wait_for=[dataset])
    register_model_task.submit(wait_for=[model])


__all__ = ["train_model_flow"]
