from __future__ import annotations

from datetime import datetime
from typing import Optional

from prefect import flow, task, get_run_logger

from nba_predictor.transformations import (
    build_matchups_h2h_base,
    build_matchups_h2h_features,
    build_player_availability,
    build_team_boxscores_agg,
    build_team_form_windowed,
    build_team_game_facts,
)
from scripts import rebuild_bronze


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@task
def rebuild_bronze_task(ingest_ts: str) -> None:
    get_run_logger().info("Rebuilding bronze datasets", ingest_ts=ingest_ts)
    rebuild_bronze.rebuild_all(
        ingest_ts=ingest_ts,
        include_raw=True,
        include_raw_last=True,
        normalise_columns=True,
    )


@task
def team_game_facts_task(ingest_ts: str) -> None:
    build_team_game_facts(ingest_ts=ingest_ts)


@task
def team_boxscores_agg_task(ingest_ts: str) -> None:
    build_team_boxscores_agg(ingest_ts=ingest_ts)


@task
def player_availability_task(ingest_ts: str) -> None:
    build_player_availability(ingest_ts=ingest_ts)


@task
def team_form_windowed_task(ingest_ts: str) -> None:
    build_team_form_windowed(ingest_ts=ingest_ts)


@task
def matchups_h2h_base_task(ingest_ts: str) -> None:
    build_matchups_h2h_base(ingest_ts=ingest_ts)


@task
def matchups_h2h_features_task(ingest_ts: str) -> None:
    build_matchups_h2h_features(ingest_ts=ingest_ts)


@flow(name="update_data_flow")
def update_data_flow(*, ingest_ts: Optional[str] = None, rebuild_bronze_data: bool = False) -> None:
    ts = ingest_ts or _timestamp()
    if rebuild_bronze_data:
        rebuild_bronze_task(ingest_ts=ts)

    team_game_facts_task(ingest_ts=ts)
    team_boxscores_agg_task(ingest_ts=ts)
    player_availability_task(ingest_ts=ts)
    team_form_windowed_task(ingest_ts=ts)
    matchups_h2h_base_task(ingest_ts=ts)
    matchups_h2h_features_task(ingest_ts=ts)


__all__ = ["update_data_flow"]
