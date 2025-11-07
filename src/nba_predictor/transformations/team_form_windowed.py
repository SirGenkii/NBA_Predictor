from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Sequence

import polars as pl
import structlog

from nba_predictor import settings
from nba_predictor.logging import configure_logging
from nba_predictor.transformations.utils import write_partitioned

logger = structlog.get_logger("nba_predictor.transformations.team_form_windowed")

BRONZE_SUBDIR = settings.data_paths.bronze_root / "nba_api"
TEAM_AGG_ROOT = settings.data_paths.silver_team_boxscores_agg
TEAM_FACTS_ROOT = settings.data_paths.silver_team_game_facts
OUTPUT_ROOT = settings.data_paths.silver_team_form_windowed

WINDOW_SIZES = settings.feature_windows
STD_MAX_WINDOW = 10

BASE_COLUMNS = [
    "season",
    "game_date",
    "game_id",
    "team_id",
    "opponent_team_id",
    "is_home",
    "is_win",
]

EXCLUDE_FEATURE_COLUMNS = {
    "team_id",
    "game_id",
    "season",
    "game_date",
    "opponent_team_id",
    "is_home",
    "is_win",
    "bronze_ingest_ts",
    "bronze_source_snapshot",
    "bronze_source_file",
    "team_name",
    "team_tricode",
    "team_abbreviation",
}
OUTPUT_EXCLUDE_COLUMNS = {
    "bronze_source_snapshot",
    "bronze_source_file",
    "bronze_ingest_ts",
    "team_name",
    "team_tricode",
    "team_abbreviation",
}


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _collect_latest_parquet(root: Path) -> list[str]:
    pattern = root / "**/*.parquet"
    matches = glob(str(pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No parquet files found under {root}.")
    return matches


def _load_team_boxscores() -> pl.LazyFrame:
    paths = _collect_latest_parquet(TEAM_AGG_ROOT)
    return pl.concat([pl.scan_parquet(path) for path in paths], how="diagonal_relaxed")


def _load_team_game_facts() -> pl.LazyFrame:
    paths = _collect_latest_parquet(TEAM_FACTS_ROOT)
    return pl.concat([pl.scan_parquet(path) for path in paths], how="diagonal_relaxed")


def _identify_feature_columns(schema: Dict[str, pl.DataType]) -> list[str]:
    columns: list[str] = []
    for name, dtype in schema.items():
        if name in EXCLUDE_FEATURE_COLUMNS:
            continue
        if isinstance(dtype, (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)):
            columns.append(name)
    return columns


def build_team_form_windowed(
    *,
    ingest_ts: str | None = None,
    window_sizes: Sequence[int] | None = None,
) -> Path:
    """
    Build rolling team-form features over multiple window sizes.
    """

    configure_logging(settings.data_paths.logs_root, pipeline="build_team_form_windowed")

    ingest_ts = ingest_ts or _default_ingest_ts()
    window_sizes = tuple(window_sizes or WINDOW_SIZES)
    logger.info("team_form_windowed_start", ingest_ts=ingest_ts, window_sizes=window_sizes)

    team_boxscores = _load_team_boxscores()
    team_facts = _load_team_game_facts()

    # Get seasons FIRST to process one at a time (critical for memory)
    seasons_df = team_facts.select(pl.col("season").unique()).collect(streaming=True)
    seasons = seasons_df["season"].to_list() if "season" in seasons_df.columns else []
    
    if not seasons:
        raise ValueError("No seasons found in team_game_facts")

    logger.info("team_form_windowed_seasons", seasons_count=len(seasons), seasons=sorted(seasons))

    total_rows = 0
    
    # Process each season separately to avoid memory explosion
    for season in seasons:
        logger.info("team_form_windowed_season_start", season=season)
        
        # Filter by season BEFORE joining (reduces memory)
        season_boxscores = team_boxscores.filter(pl.col("season") == season)
        season_facts = team_facts.filter(pl.col("season") == season)

        joined = (
            season_boxscores.join(
                season_facts.select(
                    "season",
                    "game_date",
                    "game_id",
                    "team_id",
                    "opponent_team_id",
                    "is_home",
                    "is_win",
                ),
                on=["game_id", "team_id"],
                how="inner",
            )
            .with_columns(
                [
                    pl.col("game_date").cast(pl.Date),
                    pl.col("is_win").cast(pl.Int8).alias("is_win_int"),
                ]
            )
            .sort(["team_id", "game_date"])
        )

        schema = joined.schema
        feature_columns = _identify_feature_columns(schema)

        rolling_exprs: list[pl.Expr] = []
        for window in window_sizes:
            suffix = f"last{window}"
            for column in feature_columns:
                rolling_exprs.append(
                    pl.col(column)
                    .rolling_mean(window_size=window, min_periods=1)
                    .over("team_id")
                    .alias(f"{column}_avg_{suffix}")
                )
                if window <= STD_MAX_WINDOW:
                    rolling_exprs.append(
                        pl.col(column)
                        .rolling_std(window_size=window, min_periods=1)
                        .over("team_id")
                        .alias(f"{column}_std_{suffix}")
                    )
            rolling_exprs.append(
                pl.col("is_win_int")
                .rolling_mean(window_size=window, min_periods=1)
                .over("team_id")
                .alias(f"win_rate_{suffix}")
            )
            rolling_exprs.append(
                pl.col("is_win_int")
                .rolling_sum(window_size=window, min_periods=1)
                .over("team_id")
                .alias(f"wins_{suffix}")
            )

        enriched = joined.with_columns(rolling_exprs).drop("is_win_int")

        output_columns = BASE_COLUMNS + [
            col for col in enriched.columns if col not in BASE_COLUMNS and col not in OUTPUT_EXCLUDE_COLUMNS
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
            "team_form_windowed_season_complete",
            season=season,
            rows=season_materialized.height,
        )

    logger.info(
        "team_form_windowed_complete",
        total_rows=total_rows,
        window_sizes=window_sizes,
    )
    return OUTPUT_ROOT


__all__ = ["build_team_form_windowed"]
