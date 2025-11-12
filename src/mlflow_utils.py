from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_URI = "file:./mlruns"


def _resolve_tracking_path(tracking_uri: str) -> Path:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"Unsupported tracking URI for filesystem path: {tracking_uri}")
    path = parsed.path or "./mlruns"
    return Path(path)


def _iter_experiments(client: MlflowClient, name: Optional[str]) -> Iterable[Experiment]:
    if name:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise ValueError(f"Experiment '{name}' not found at {client.tracking_uri}.")
        return [experiment]
    return client.list_experiments()


def get_latest_run(
    *,
    target: str,
    experiment_name: Optional[str] = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    filter_string: Optional[str] = None,
) -> Run:
    """
    Return the most recent MLflow run for a given target and experiment.

    Args:
        target: Value stored in the `params.target` field.
        experiment_name: MLflow experiment name.
        tracking_uri: Tracking URI (defaults to local ./mlruns store).
        filter_string: Optional extra MLflow filter to AND with the target condition.

    Raises:
        ValueError if the experiment or matching run is not found.
    """

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    filters = [f"params.target = '{target}'"]
    if filter_string:
        filters.append(filter_string)
    filter_expr = " and ".join(filters)

    latest_run: Optional[Run] = None
    experiments = _iter_experiments(client, experiment_name)
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_expr,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            continue
        run = runs[0]
        if latest_run is None or run.info.start_time > latest_run.info.start_time:
            latest_run = run

    if latest_run is None:
        exp_msg = experiment_name or "any experiment"
        raise ValueError(f"No run found for target='{target}' in {exp_msg}.")

    return latest_run


def get_latest_run_dir(
    *,
    target: str,
    experiment_name: Optional[str] = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    filter_string: Optional[str] = None,
) -> Path:
    """
    Resolve the filesystem directory containing the most recent run artifacts.
    """

    run = get_latest_run(
        target=target,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        filter_string=filter_string,
    )
    tracking_path = _resolve_tracking_path(tracking_uri)
    return tracking_path / run.info.experiment_id / run.info.run_id
