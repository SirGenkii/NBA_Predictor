from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Sequence

import polars as pl
import structlog

from nba_predictor import settings
from nba_predictor.logging import configure_logging
from nba_predictor.transformations.utils import minutes_to_float, write_partitioned

logger = structlog.get_logger("nba_predictor.transformations.player_availability")

BRONZE_SUBDIR = settings.data_paths.bronze_root / "nba_api"
OUTPUT_ROOT = settings.data_paths.silver_player_availability


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


CORE_TOP_N_DEFAULT = 5
CORE_MINUTES_WINDOW = 10


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return pl.when(pl.col(denominator) > 0).then(pl.col(numerator) / pl.col(denominator)).otherwise(None).alias(alias)


def _load_boxscores(snapshot_labels: Sequence[str]) -> pl.LazyFrame:
    scans = []
    for label in snapshot_labels:
        pattern = BRONZE_SUBDIR / label / "**" / "*boxscores*.parquet"
        pattern_str = str(pattern)
        if glob(pattern_str, recursive=True):
            scans.append(pl.scan_parquet(pattern_str))
    if not scans:
        raise FileNotFoundError("No bronze boxscores parquet files found. Run `rebuild_bronze.py` first.")
    return pl.concat(scans)


def _load_games(snapshot_labels: Sequence[str]) -> pl.LazyFrame:
    scans = []
    for label in snapshot_labels:
        for pattern in (
            BRONZE_SUBDIR / label / "**" / "*games*.parquet",
            BRONZE_SUBDIR / label / "**" / "*games_merged*.parquet",
        ):
            pattern_str = str(pattern)
            if glob(pattern_str, recursive=True):
                scans.append(pl.scan_parquet(pattern_str))
    if not scans:
        raise FileNotFoundError("No bronze games parquet files found. Run `rebuild_bronze.py` first.")
    return pl.concat(scans)


def _load_roster(snapshot_labels: Sequence[str]) -> Optional[pl.DataFrame]:
    frames = []
    patterns = ("*commonteamroster*.parquet", "*team_roster*.parquet")
    for label in snapshot_labels:
        for pattern in patterns:
            pattern_str = str(BRONZE_SUBDIR / label / "**" / pattern)
            for path in glob(pattern_str, recursive=True):
                frames.append(pl.read_parquet(path))
    if not frames:
        return None
    roster = pl.concat(frames, how="vertical_relaxed")
    columns = roster.columns
    rename_map: Dict[str, str] = {}
    if "teamid" in columns:
        rename_map["teamid"] = "team_id"
    if "team_id" not in columns and "team" in columns:
        rename_map["team"] = "team_id"
    if "player_id" not in columns and "playerid" in columns:
        rename_map["playerid"] = "player_id"
    if "player" in columns and "player_name" not in columns:
        rename_map["player"] = "player_name"
    if rename_map:
        roster = roster.rename(rename_map)
    expected_columns = {"team_id", "player_id"}
    if not expected_columns.issubset(set(roster.columns)):
        logger.warning("roster_missing_expected_columns", columns=roster.columns)
        return None
    if "season" not in roster.columns and "season_start_year" in roster.columns:
        roster = roster.with_columns(
            pl.concat_str(
                pl.col("season_start_year").cast(pl.Int64).cast(pl.Utf8),
                pl.lit("-"),
                ((pl.col("season_start_year").cast(pl.Int64) + 1) % 100)
                .cast(pl.Int64)
                .cast(pl.Utf8)
                .str.zfill(2),
            ).alias("season")
        )
    return roster.with_columns(
        [
            pl.col("team_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64),
        ]
    )


def build_player_availability(
    *,
    snapshot_labels: Sequence[str] = ("raw", "raw_last"),
    ingest_ts: str | None = None,
    core_top_n: int = CORE_TOP_N_DEFAULT,
    core_minutes_window: int = CORE_MINUTES_WINDOW,
) -> Path:
    """
    Build the silver player_availability dataset combining boxscores with roster context.
    """

    configure_logging(settings.data_paths.logs_root, pipeline="build_player_availability")

    ingest_ts = ingest_ts or _default_ingest_ts()
    logger.info("player_availability_start", ingest_ts=ingest_ts, snapshots=list(snapshot_labels))

    boxscores = _load_boxscores(snapshot_labels)
    games = _load_games(snapshot_labels)

    games_df = games.select(
        pl.col("game_id"),
        pl.col("team_id").cast(pl.Int64),
        pl.col("game_date").cast(pl.Date),
        pl.col("season").alias("season_label"),
        pl.col("season_id"),
    ).with_columns(
        pl.when(pl.col("season_label").is_not_null())
        .then(pl.col("season_label"))
        .otherwise(
            pl.concat_str(
                (pl.col("season_id").cast(pl.Int64) % 10000).cast(pl.Utf8),
                pl.lit("-"),
                (((pl.col("season_id").cast(pl.Int64) % 10000) + 1) % 100).cast(pl.Utf8).str.zfill(2),
            )
        )
        .alias("season")
    )

    boxscores_df = (
        boxscores.select(
            pl.col("game_id"),
            pl.col("team_id").cast(pl.Int64),
            pl.col("person_id").cast(pl.Int64),
            pl.col("minutes"),
            pl.col("source_snapshot"),
            pl.col("ingest_ts").alias("bronze_ingest_ts"),
        )
        .with_columns(minutes_to_float(pl.col("minutes")).alias("minutes_float"))
        .join(games_df.select("game_id", "team_id", "game_date", "season"), on=["game_id", "team_id"], how="left")
    )

    player_minutes = (
        boxscores_df.sort(["season", "team_id", "person_id", "game_date"])
        .with_columns(
            [
                pl.col("minutes_float")
                .rolling_mean(core_minutes_window, min_periods=1)
                .over(["season", "team_id", "person_id"])
                .shift(1)
                .alias("avg_minutes_prev_window"),
                pl.col("minutes_float").gt(0).alias("is_active"),
            ]
        )
        .with_columns(
            pl.when(pl.col("avg_minutes_prev_window").is_not_null())
            .then(
                pl.col("avg_minutes_prev_window")
                .rank("dense", descending=True)
                .over(["season", "team_id", "game_id"])
            )
            .otherwise(None)
            .alias("core_rank")
        )
        .with_columns(
            (
                pl.col("core_rank").is_not_null() & (pl.col("core_rank") <= core_top_n)
            ).alias("is_expected_core")
        )
    )

    availability = (
        player_minutes.groupby(["season", "game_date", "game_id", "team_id"])
        .agg(
            [
                pl.col("bronze_ingest_ts").first(),
                pl.col("source_snapshot").first(),
                pl.col("is_active").cast(pl.Int64).sum().alias("active_players"),
                pl.col("minutes_float").sum().alias("total_minutes"),
                pl.col("minutes_float")
                .filter(pl.col("minutes_float") > 0)
                .sort(descending=True)
                .head(3)
                .list.sum()
                .alias("top3_minutes"),
                pl.col("minutes_float")
                .filter(pl.col("minutes_float") > 0)
                .sort(descending=True)
                .head(5)
                .list.sum()
                .alias("top5_minutes"),
                pl.col("is_expected_core").cast(pl.Int64).sum().alias("core_players_expected"),
                (
                    pl.col("is_expected_core").cast(pl.Int64) * pl.col("is_active").cast(pl.Int64)
                )
                .sum()
                .alias("core_players_active"),
            ]
        )
        .with_columns(
            [
                _safe_ratio("top3_minutes", "total_minutes", "top3_minutes_share"),
                _safe_ratio("top5_minutes", "total_minutes", "top5_minutes_share"),
            ]
        )
        .with_columns(
            [
                (1.0 - pl.col("top5_minutes_share")).alias("bench_minutes_share"),
                (pl.col("total_minutes") - 240).alias("minutes_delta"),
                (pl.col("core_players_expected") - pl.col("core_players_active")).clip_min(0).alias("core_players_missing"),
            ]
        )
    )

    availability = availability.with_columns(
        [
            pl.when(pl.col("core_players_expected") == 0)
            .then(pl.col("active_players").clip_max(core_top_n))
            .otherwise(pl.col("core_players_expected"))
            .alias("core_players_expected"),
            pl.col("core_players_active").fill_null(0),
        ]
    )

    roster = _load_roster(snapshot_labels)
    if roster is not None:
        if "season" in roster.columns:
            roster_counts = roster.groupby(["season", "team_id"]).agg(pl.count().alias("roster_players"))
            availability = availability.join(roster_counts, on=["season", "team_id"], how="left")
        else:
            roster_counts = roster.groupby("team_id").agg(pl.count().alias("roster_players"))
            availability = availability.join(roster_counts, on="team_id", how="left")
        availability = availability.with_columns(
            (
                pl.col("roster_players")
                .fill_null(pl.col("active_players"))
                .alias("roster_players")
            )
        ).with_columns(
            (
                (pl.col("roster_players") - pl.col("active_players"))
                .clip_min(0)
                .alias("inactive_players")
            )
        )
    else:
        availability = availability.with_columns(
            [
                pl.lit(None).alias("roster_players"),
                pl.lit(None).alias("inactive_players"),
            ]
        )

    materialized = availability.collect()

    if materialized.is_empty():
        raise RuntimeError("player_availability: no rows produced. Check bronze inputs.")

    write_partitioned(
        materialized,
        OUTPUT_ROOT,
        ingest_ts,
        partition_cols=["season"],
    )

    logger.info(
        "player_availability_complete",
        rows=materialized.height,
    )
    return OUTPUT_ROOT
