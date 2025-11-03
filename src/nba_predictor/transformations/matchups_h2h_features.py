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

logger = structlog.get_logger("nba_predictor.transformations.matchups_h2h_features")

H2H_BASE_ROOT = settings.data_paths.silver_matchups_h2h_base
OUTPUT_ROOT = settings.data_paths.silver_matchups_h2h_features
WINDOW_SIZES = settings.feature_windows

EXCLUDE_COLUMNS = {
    "season",
    "game_date",
    "game_id",
    "team_id",
    "opponent_team_id",
    "is_home",
    "is_win",
}


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _glob_parquet(root: Path) -> list[str]:
    pattern = root / "**/*.parquet"
    matches = glob(str(pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No parquet files found in {root}")
    return matches


def _feature_columns(schema: Dict[str, pl.DataType]) -> list[str]:
    columns: list[str] = []
    for name, dtype in schema.items():
        if name in EXCLUDE_COLUMNS:
            continue
        if isinstance(dtype, (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)):
            columns.append(name)
    return columns


def build_matchups_h2h_features(
    *,
    ingest_ts: str | None = None,
    window_sizes: Sequence[int] | None = None,
    decay_lambda: float | None = None,
) -> Path:
    """
    Build head-to-head rolling features for team vs opponent.
    """

    configure_logging(settings.data_paths.logs_root, pipeline="build_matchups_h2h_features")

    ingest_ts = ingest_ts or _default_ingest_ts()
    window_sizes = tuple(window_sizes or WINDOW_SIZES)
    logger.info("matchups_h2h_features_start", ingest_ts=ingest_ts, window_sizes=window_sizes, decay_lambda=decay_lambda)

    base = pl.concat([pl.scan_parquet(path) for path in _glob_parquet(H2H_BASE_ROOT)])

    base = base.with_columns(
        [
            pl.col("game_date").cast(pl.Date),
            pl.col("is_win").cast(pl.Int8).alias("is_win_int"),
            (pl.col("point_margin")).alias("point_margin"),
        ]
    ).sort(["team_id", "opponent_team_id", "game_date"])

    schema = base.schema
    feature_cols = _feature_columns(schema)

    rolling_exprs: list[pl.Expr] = []
    for window in window_sizes:
        suffix = f"last{window}"
        for column in feature_cols:
            rolling_exprs.append(
                pl.col(column)
                .rolling_mean(window_size=window, min_periods=1)
                .over(["team_id", "opponent_team_id"])
                .alias(f"{column}_avg_{suffix}")
            )
            rolling_exprs.append(
                pl.col(column)
                .rolling_std(window_size=window, min_periods=1)
                .over(["team_id", "opponent_team_id"])
                .alias(f"{column}_std_{suffix}")
            )
        rolling_exprs.append(
            pl.col("is_win_int")
            .rolling_mean(window_size=window, min_periods=1)
            .over(["team_id", "opponent_team_id"])
            .alias(f"h2h_win_rate_{suffix}")
        )
        rolling_exprs.append(
            pl.col("is_win_int")
            .rolling_sum(window_size=window, min_periods=1)
            .over(["team_id", "opponent_team_id"])
            .alias(f"h2h_wins_{suffix}")
        )
        rolling_exprs.append(
            pl.col("point_margin")
            .rolling_mean(window_size=window, min_periods=1)
            .over(["team_id", "opponent_team_id"])
            .alias(f"h2h_margin_avg_{suffix}")
        )

    enriched = base.with_columns(rolling_exprs).drop(["is_win_int"])

    output_columns = [
        "season",
        "game_date",
        "game_id",
        "team_id",
        "opponent_team_id",
        "is_home",
        "is_win",
    ] + [
        col
        for col in enriched.columns
        if col not in {"season", "game_date", "game_id", "team_id", "opponent_team_id", "is_home", "is_win"}
    ]

    materialized = enriched.select(output_columns).collect()

    write_partitioned(
        materialized,
        OUTPUT_ROOT,
        ingest_ts,
        partition_cols=["season"],
    )

    logger.info(
        "matchups_h2h_features_complete",
        rows=materialized.height,
    )
    return OUTPUT_ROOT
