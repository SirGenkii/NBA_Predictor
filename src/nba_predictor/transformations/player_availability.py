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
    paths: list[Path] = []
    for label in snapshot_labels:
        pattern = BRONZE_SUBDIR / label / "**" / "*boxscores*.parquet"
        for path_str in glob(str(pattern), recursive=True):
            paths.append(Path(path_str))
    if not paths:
        raise FileNotFoundError("No bronze boxscores parquet files found. Run `rebuild_bronze.py` first.")

    frames: list[pl.DataFrame] = []
    for path in paths:
        df = pl.read_parquet(path)

        for col_name, alias in [
            ("ingest_ts", "bronze_ingest_ts"),
            ("source_snapshot", "bronze_source_snapshot"),
            ("source_file", "bronze_source_file"),
        ]:
            if col_name in df.columns:
                if alias in df.columns:
                    df = df.drop(alias)
                df = df.rename({col_name: alias})
            elif alias not in df.columns:
                df = df.with_columns(pl.lit(None).alias(alias))

        drops = [col for col in ("ingest_ts", "source_snapshot", "source_file") if col in df.columns]
        if drops:
            df = df.drop(drops)

        df = df.with_columns(
            [
                pl.col("game_id").cast(pl.Utf8, strict=False).alias("game_id"),
                pl.col("team_id").cast(pl.Utf8, strict=False).alias("team_id"),
                pl.col("person_id").cast(pl.Utf8, strict=False).alias("person_id"),
            ]
        )

        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed")
    return combined.lazy()


def _load_games(snapshot_labels: Sequence[str]) -> pl.LazyFrame:
    paths: list[Path] = []
    for label in snapshot_labels:
        for pattern in (
            BRONZE_SUBDIR / label / "**" / "*games*.parquet",
            BRONZE_SUBDIR / label / "**" / "*games_merged*.parquet",
        ):
            for path_str in glob(str(pattern), recursive=True):
                paths.append(Path(path_str))
    if not paths:
        raise FileNotFoundError("No bronze games parquet files found. Run `rebuild_bronze.py` first.")

    frames: list[pl.DataFrame] = []
    for path in paths:
        df = pl.read_parquet(path)

        for col_name, alias in [
            ("ingest_ts", "bronze_ingest_ts"),
            ("source_snapshot", "bronze_source_snapshot"),
            ("source_file", "bronze_source_file"),
        ]:
            if col_name in df.columns:
                if alias in df.columns:
                    df = df.drop(alias)
                df = df.rename({col_name: alias})
            elif alias not in df.columns:
                df = df.with_columns(pl.lit(None).alias(alias))

        rename_map = {}
        if "fg3_m" in df.columns and "fg3m" not in df.columns:
            rename_map["fg3_m"] = "fg3m"
        if "fg3_a" in df.columns and "fg3a" not in df.columns:
            rename_map["fg3_a"] = "fg3a"
        if rename_map:
            df = df.rename(rename_map)

        if "team_tricode" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("team_tricode"))

        df = df.with_columns(
            [
                pl.col("game_id").cast(pl.Utf8, strict=False).alias("game_id"),
                pl.col("team_id").cast(pl.Utf8, strict=False).alias("team_id"),
            ]
        )

        drops = [col for col in ("ingest_ts", "source_snapshot", "source_file") if col in df.columns]
        if drops:
            df = df.drop(drops)

        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed")
    return combined.lazy()


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
            pl.col("team_id").cast(pl.Utf8, strict=False),
            pl.col("player_id").cast(pl.Utf8, strict=False),
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
        pl.col("game_id").cast(pl.Utf8, strict=False).alias("game_id"),
        pl.col("team_id").cast(pl.Utf8, strict=False).alias("team_id"),
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
            pl.col("game_id").cast(pl.Utf8, strict=False).alias("game_id"),
            pl.col("team_id").cast(pl.Utf8, strict=False).alias("team_id"),
            pl.col("person_id").cast(pl.Utf8, strict=False).alias("person_id"),
            pl.col("minutes"),
            pl.col("bronze_source_snapshot"),
            pl.col("bronze_source_file"),
            pl.col("bronze_ingest_ts"),
        )
        .with_columns(minutes_to_float(pl.col("minutes")).alias("minutes_float"))
        .join(games_df.select("game_id", "team_id", "game_date", "season"), on=["game_id", "team_id"], how="left")
    )

    seasons_df = games_df.select(pl.col("season").unique()).collect()
    seasons = seasons_df["season"].to_list() if "season" in seasons_df.columns else []

    logger.info(
        "player_availability_season_list",
        seasons_count=len(seasons),
        seasons=sorted(seasons),
    )

    roster = _load_roster(snapshot_labels)

    total_rows = 0
    written_seasons = 0
    for season in seasons:
        season_boxscores = boxscores_df.filter(pl.col("season") == pl.lit(season))

        season_minutes = (
            season_boxscores.sort(["season", "team_id", "person_id", "game_date"])
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

        season_availability = (
            season_minutes.group_by(["season", "game_date", "game_id", "team_id"])
            .agg(
                [
                    pl.col("bronze_ingest_ts").first().alias("bronze_ingest_ts"),
                    pl.col("bronze_source_snapshot").first().alias("bronze_source_snapshot"),
                    pl.col("bronze_source_file").first().alias("bronze_source_file"),
                    pl.col("is_active").cast(pl.Int64).sum().alias("active_players"),
                    pl.col("minutes_float").sum().alias("total_minutes"),
                    pl.col("minutes_float")
                    .filter(pl.col("minutes_float") > 0)
                    .sort(descending=True)
                    .head(3)
                    .sum()
                    .alias("top3_minutes"),
                    pl.col("minutes_float")
                    .filter(pl.col("minutes_float") > 0)
                    .sort(descending=True)
                    .head(5)
                    .sum()
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
                    (pl.col("core_players_expected") - pl.col("core_players_active"))
                    .clip(lower_bound=0)
                    .alias("core_players_missing"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("core_players_expected") == 0)
                    .then(pl.col("active_players").clip(upper_bound=core_top_n))
                    .otherwise(pl.col("core_players_expected"))
                    .alias("core_players_expected"),
                    pl.col("core_players_active").fill_null(0),
                ]
            )
        )

        season_df = season_availability.collect(streaming=True)
        if season_df.is_empty():
            continue

        for meta_col in ("bronze_ingest_ts", "bronze_source_snapshot", "bronze_source_file"):
            if meta_col not in season_df.columns:
                season_df = season_df.with_columns(pl.lit(None).alias(meta_col))

        if roster is not None:
            if "season" in roster.columns:
                roster_counts = roster.filter(pl.col("season") == pl.lit(season)).group_by(["season", "team_id"]).agg(pl.count().alias("roster_players"))
                season_df = season_df.join(roster_counts, on=["season", "team_id"], how="left")
            else:
                roster_counts = roster.group_by("team_id").agg(pl.count().alias("roster_players"))
                season_df = season_df.join(roster_counts, on="team_id", how="left")
            season_df = season_df.with_columns(
                (
                    pl.col("roster_players")
                    .fill_null(pl.col("active_players"))
                    .alias("roster_players")
                )
            ).with_columns(
                (
                    (pl.col("roster_players") - pl.col("active_players"))
                    .clip(lower_bound=0)
                    .alias("inactive_players")
                )
            )
        else:
            season_df = season_df.with_columns(
                [
                    pl.lit(None).alias("roster_players"),
                    pl.lit(None).alias("inactive_players"),
                ]
            )

        write_partitioned(
            season_df,
            OUTPUT_ROOT,
            ingest_ts,
            partition_cols=["season"],
        )

        total_rows += season_df.height
        written_seasons += 1

    if written_seasons == 0:
        raise RuntimeError("player_availability: no rows produced. Check bronze inputs.")

    logger.info(
        "player_availability_complete",
        rows=total_rows,
        seasons=written_seasons,
    )
    return OUTPUT_ROOT
