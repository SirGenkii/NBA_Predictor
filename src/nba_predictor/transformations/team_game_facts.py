from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
import structlog

from nba_predictor import settings
from nba_predictor.logging import configure_logging
from nba_predictor.transformations.utils import season_label_from_id, write_partitioned

logger = structlog.get_logger("nba_predictor.transformations.team_game_facts")

BRONZE_SUBDIR = settings.data_paths.bronze_root / "nba_api"
OUTPUT_ROOT = settings.data_paths.silver_team_game_facts

GROUP_KEYS = ["game_id", "team_id"]


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_games(snapshot_labels: Sequence[str]) -> pl.LazyFrame:
    paths: list[Path] = []
    for label in snapshot_labels:
        base_pattern = BRONZE_SUBDIR / label / "**" / "*games*.parquet"
        merged_pattern = BRONZE_SUBDIR / label / "**" / "*games_merged*.parquet"
        for pattern in (base_pattern, merged_pattern):
            pattern_str = str(pattern)
            for path_str in glob(pattern_str, recursive=True):
                paths.append(Path(path_str))
    if not paths:
        raise FileNotFoundError("No bronze games parquet files found. Run `rebuild_bronze.py` first.")

    frames: list[pl.DataFrame] = []
    for path in paths:
        df = pl.read_parquet(path)

        rename_map = {}
        if "fg3_m" in df.columns and "fg3m" not in df.columns:
            rename_map["fg3_m"] = "fg3m"
        if "fg3_a" in df.columns and "fg3a" not in df.columns:
            rename_map["fg3_a"] = "fg3a"
        if rename_map:
            df = df.rename(rename_map)

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

        if "team_tricode" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("team_tricode"))

        df = df.with_columns(
            [
                pl.col("game_id")
                .cast(pl.Utf8, strict=False)
                .alias("game_id"),
                pl.col("team_id").cast(pl.Utf8, strict=False).alias("team_id"),
            ]
        )

        drops = [col for col in ("ingest_ts", "source_snapshot", "source_file") if col in df.columns]
        if drops:
            df = df.drop(drops)

        if "bronze_source_file" not in df.columns:
            df = df.with_columns(pl.lit(str(path)).alias("bronze_source_file"))

        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed")
    return combined.lazy()


def _derive_opponent_abbreviation(matchup: pl.Expr) -> pl.Expr:
    return matchup.str.extract(r"(?:vs\.?|@) ([A-Z]{2,3})$", 1).alias("opponent_abbreviation")


def _derive_is_home(matchup: pl.Expr) -> pl.Expr:
    return matchup.str.contains(r"vs\.?").alias("is_home")


def _derive_is_win(wl: pl.Expr) -> pl.Expr:
    return wl.str.to_uppercase().eq("W").alias("is_win")


def _validate_team_game_facts(df: pl.DataFrame) -> None:
    df = df.with_columns(pl.col("game_id").cast(pl.Utf8, strict=False))

    mismatched_team_counts = (
        df.group_by("game_id")
        .agg(pl.n_unique("team_id").alias("team_count"))
        .filter(pl.col("team_count") != 2)
    )
    if mismatched_team_counts.height > 0:
        logger.warning(
            "team_game_facts_team_count_mismatch",
            game_ids=mismatched_team_counts["game_id"].to_list(),
        )

    missing_opponents = df.filter(pl.col("opponent_team_id").is_null())
    if missing_opponents.height > 0:
        logger.warning(
            "team_game_facts_missing_opponent",
            rows=missing_opponents.height,
        )

    opponent_points_map = df.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opponent_team_id"),
        pl.col("team_points").alias("actual_opponent_points"),
    )
    symmetry_checks = (
        df.join(opponent_points_map, on=["game_id", "opponent_team_id"], how="left")
        .filter(
            pl.col("actual_opponent_points").is_not_null()
            & (pl.col("actual_opponent_points") != pl.col("opponent_points"))
        )
        .select("game_id", "team_id", "team_points", "opponent_points", "actual_opponent_points")
    )
    if symmetry_checks.height > 0:
        logger.warning(
            "team_game_facts_points_asymmetry",
            rows=symmetry_checks.height,
        )


def build_team_game_facts(
    *,
    snapshot_labels: Sequence[str] = ("raw", "raw_last"),
    ingest_ts: str | None = None,
) -> Path:
    """
    Build the silver team_game_facts dataset from bronze games parquet files.
    """

    configure_logging(settings.data_paths.logs_root, pipeline="build_team_game_facts")

    ingest_ts = ingest_ts or _default_ingest_ts()
    logger.info("team_game_facts_start", ingest_ts=ingest_ts, snapshots=list(snapshot_labels))

    games = _load_games(snapshot_labels)

    seasons_df = games.select(pl.col("season").unique()).collect()
    seasons = seasons_df["season"].to_list() if "season" in seasons_df.columns else []

    total_rows = 0
    written_seasons = 0
    for season in seasons:
        season_games = games.filter(pl.col("season") == pl.lit(season))

        base = (
            season_games.select(
                pl.col("game_id"),
                pl.col("team_id"),
                pl.col("team_abbreviation"),
                pl.col("team_name"),
                pl.coalesce([pl.col("team_tricode"), pl.col("team_abbreviation")]).alias("team_tricode"),
                pl.col("game_date").cast(pl.Date),
                pl.col("matchup"),
                pl.col("wl"),
                pl.col("min").alias("team_minutes"),
                pl.col("pts").alias("team_points"),
                pl.col("fgm").alias("team_fgm"),
                pl.col("fga").alias("team_fga"),
                pl.col("fg_pct").alias("team_fg_pct"),
                pl.col("fg3m").alias("team_fg3m"),
                pl.col("fg3a").alias("team_fg3a"),
                pl.col("fg3_pct").alias("team_fg3_pct"),
                pl.col("ftm").alias("team_ftm"),
                pl.col("fta").alias("team_fta"),
                pl.col("ft_pct").alias("team_ft_pct"),
                pl.col("oreb").alias("team_oreb"),
                pl.col("dreb").alias("team_dreb"),
                pl.col("reb").alias("team_reb"),
                pl.col("ast").alias("team_ast"),
                pl.col("stl").alias("team_stl"),
                pl.col("blk").alias("team_blk"),
                pl.col("tov").alias("team_tov"),
                pl.col("pf").alias("team_pf"),
                pl.col("plus_minus").alias("team_plus_minus"),
                pl.col("season").alias("season_label"),
                pl.col("season_id"),
            )
            .with_columns(
                [
                    _derive_opponent_abbreviation(pl.col("matchup")),
                    _derive_is_home(pl.col("matchup")),
                    _derive_is_win(pl.col("wl")),
                    season_label_from_id(pl.col("season_id"), fallback=pl.col("season_label")).alias("season"),
                ]
            )
            .drop("season_label")
        )

        opponent_lookup = base.select(
            pl.col("game_id"),
            pl.col("team_abbreviation").alias("opponent_abbreviation"),
            pl.col("team_id").alias("opponent_team_id"),
            pl.col("team_points").alias("opponent_points"),
            pl.col("team_plus_minus").alias("opponent_plus_minus"),
        ).unique()

        enriched = base.join(
            opponent_lookup,
            on=["game_id", "opponent_abbreviation"],
            how="left",
        )

        season_df = (
            enriched.with_columns(
                pl.col("game_date").cast(pl.Date),
            )
            .drop("matchup", "wl")
            .select(
                [
                    "season",
                    "game_date",
                    "game_id",
                    "team_id",
                    "team_abbreviation",
                    "team_name",
                    "team_tricode",
                    "is_home",
                    "is_win",
                    "team_minutes",
                    "team_points",
                    "team_fgm",
                    "team_fga",
                    "team_fg_pct",
                    "team_fg3m",
                    "team_fg3a",
                    "team_fg3_pct",
                    "team_ftm",
                    "team_fta",
                    "team_ft_pct",
                    "team_oreb",
                    "team_dreb",
                    "team_reb",
                    "team_ast",
                    "team_stl",
                    "team_blk",
                    "team_tov",
                    "team_pf",
                    "team_plus_minus",
                    "opponent_team_id",
                    "opponent_points",
                    "opponent_plus_minus",
                ]
            )
            .collect(streaming=True)
            .with_columns(
                [
                    pl.col("game_id").cast(pl.Utf8, strict=False),
                    pl.col("team_id").cast(pl.Utf8, strict=False),
                    pl.col("opponent_team_id").cast(pl.Utf8, strict=False),
                ]
            )
        )

        if season_df.is_empty():
            continue

        for meta_col in ("bronze_ingest_ts", "bronze_source_snapshot", "bronze_source_file"):
            if meta_col not in season_df.columns:
                season_df = season_df.with_columns(pl.lit(None).alias(meta_col))

        _validate_team_game_facts(season_df)

        write_partitioned(
            season_df,
            OUTPUT_ROOT,
            ingest_ts,
            partition_cols=["season"],
        )

        total_rows += season_df.height
        written_seasons += 1

    if written_seasons == 0:
        raise RuntimeError("team_game_facts: no rows produced. Check bronze inputs.")

    logger.info(
        "team_game_facts_complete",
        rows=total_rows,
        seasons=written_seasons,
    )
    return OUTPUT_ROOT
