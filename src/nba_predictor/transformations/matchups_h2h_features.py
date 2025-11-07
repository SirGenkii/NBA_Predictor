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
STD_MAX_WINDOW = 10

EXCLUDE_COLUMNS = {
    "season",
    "game_date",
    "game_id",
    "team_id",
    "opponent_team_id",
    "is_home",
    "is_win",
    "bronze_ingest_ts",
    "bronze_source_snapshot",
    "bronze_source_file",
    "team_name",
    "team_abbreviation",
    "team_tricode",
}
OUTPUT_EXCLUDE_COLUMNS = EXCLUDE_COLUMNS


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

    # Load base data
    base_all = pl.concat(
        [pl.scan_parquet(path) for path in _glob_parquet(H2H_BASE_ROOT)],
        how="diagonal_relaxed",
    )

    # Get seasons FIRST to process one at a time (critical for memory)
    seasons_df = base_all.select(pl.col("season").unique()).collect(streaming=True)
    seasons = seasons_df["season"].to_list() if "season" in seasons_df.columns else []
    
    if not seasons:
        raise ValueError("No seasons found in h2h_base")

    logger.info("matchups_h2h_features_seasons", seasons_count=len(seasons), seasons=sorted(seasons))

    total_rows = 0
    
    # Process each season separately to avoid memory explosion
    for season in seasons:
        logger.info("matchups_h2h_features_season_start", season=season)
        
        # Filter by season BEFORE processing (reduces memory)
        base = (
            base_all.filter(pl.col("season") == season)
            .with_columns(
                [
                    pl.col("game_date").cast(pl.Date),
                    pl.col("is_win").cast(pl.Int8).alias("is_win_int"),
                    (pl.col("point_margin")).alias("point_margin"),
                ]
            )
            .sort(["team_id", "opponent_team_id", "game_date"])
        )

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
                if window <= STD_MAX_WINDOW:
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
            if col
            not in OUTPUT_EXCLUDE_COLUMNS
        ]

        # Collect and write THIS season only (releases memory after each season)
        season_materialized = enriched.select(output_columns).collect(streaming=True)
        float_cols = [
            name for name, dtype in season_materialized.schema.items() if isinstance(dtype, (pl.Float64,))
        ]
        if float_cols:
            season_materialized = season_materialized.with_columns(pl.col(float_cols).cast(pl.Float32))

        # Write this season directly
        write_partitioned(
            season_materialized,
            OUTPUT_ROOT,
            ingest_ts,
            partition_cols=["season"],
        )
        
        total_rows += season_materialized.height
        logger.info(
            "matchups_h2h_features_season_complete",
            season=season,
            rows=season_materialized.height,
        )

    logger.info(
        "matchups_h2h_features_complete",
        total_rows=total_rows,
    )
    return OUTPUT_ROOT
