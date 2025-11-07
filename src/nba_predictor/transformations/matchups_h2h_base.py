from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Sequence

import polars as pl
import structlog

from nba_predictor import settings
from nba_predictor.logging import configure_logging
from nba_predictor.transformations.utils import write_partitioned

logger = structlog.get_logger("nba_predictor.transformations.matchups_h2h_base")

TEAM_AGG_ROOT = settings.data_paths.silver_team_boxscores_agg
TEAM_FACTS_ROOT = settings.data_paths.silver_team_game_facts
OUTPUT_ROOT = settings.data_paths.silver_matchups_h2h_base

EXCLUDE_AGG_COLUMNS = {
    "team_id",
    "game_id",
    "season",
    "bronze_ingest_ts",
    "bronze_source_snapshot",
    "bronze_source_file",
}


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _glob_parquet(root: Path) -> list[str]:
    pattern = root / "**/*.parquet"
    matches = glob(str(pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No parquet files found in {root}")
    return matches


def build_matchups_h2h_base(*, ingest_ts: str | None = None) -> Path:
    """
    Prepare base head-to-head dataset per (team, opponent, game).
    """

    configure_logging(settings.data_paths.logs_root, pipeline="build_matchups_h2h_base")

    ingest_ts = ingest_ts or _default_ingest_ts()
    logger.info("matchups_h2h_base_start", ingest_ts=ingest_ts)

    team_facts = pl.concat(
        [pl.scan_parquet(path) for path in _glob_parquet(TEAM_FACTS_ROOT)],
        how="diagonal_relaxed",
    )
    team_agg = pl.concat(
        [pl.scan_parquet(path) for path in _glob_parquet(TEAM_AGG_ROOT)],
        how="diagonal_relaxed",
    )

    agg_columns = [col for col in team_agg.schema if col not in EXCLUDE_AGG_COLUMNS]

    joined = (
        team_facts.join(
            team_agg.select(["game_id", "team_id", *agg_columns]),
            on=["game_id", "team_id"],
            how="left",
        )
        .with_columns(
            [
                pl.col("game_date").cast(pl.Date),
                (pl.col("team_points") - pl.col("opponent_points")).alias("point_margin"),
                (pl.col("team_plus_minus")).alias("plus_minus"),
            ]
        )
        .sort(["team_id", "game_date"])
    )

    materialized = joined.collect(streaming=True)

    write_partitioned(
        materialized,
        OUTPUT_ROOT,
        ingest_ts,
        partition_cols=["season"],
    )

    logger.info(
        "matchups_h2h_base_complete",
        rows=materialized.height,
    )
    return OUTPUT_ROOT
