from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import polars as pl
import structlog

from nba_predictor import settings
from nba_predictor.logging import configure_logging
from nba_predictor.transformations.schema import FieldRule, build_rules_from_schema
from nba_predictor.transformations.utils import minutes_to_float, write_partitioned

logger = structlog.get_logger("nba_predictor.transformations.team_boxscores_agg")

BRONZE_SUBDIR = settings.data_paths.bronze_root / "nba_api"
OUTPUT_ROOT = settings.data_paths.silver_team_boxscores_agg

GROUP_KEYS = ["game_id", "team_id"]
PLAYER_ID_COLUMNS = {
    "person_id",
    "first_name",
    "family_name",
    "name_i",
    "player_slug",
    "position",
    "comment",
    "jersey_num",
}
METADATA_COLUMNS = {"bronze_source_snapshot", "bronze_source_file", "bronze_ingest_ts"}


SUM_KEYWORDS = (
    "made",
    "attempted",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "minutes",
    "possessions",
    "wins",
    "losses",
    "pts",
    "reb",
    "ast",
    "blk",
    "stl",
    "tov",
    "pf",
)

DERIVED_RATIOS = (
    ("team_field_goals_made_traditional", "team_field_goals_attempted_traditional", "team_fg_pct"),
    ("team_three_pointers_made_traditional", "team_three_pointers_attempted_traditional", "team_fg3_pct"),
    ("team_free_throws_made_traditional", "team_free_throws_attempted_traditional", "team_ft_pct"),
)


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_boxscores(snapshot_labels: Sequence[str]) -> pl.LazyFrame:
    """
    Load boxscores using lazy evaluation to avoid loading all files in memory.
    This is a critical optimization to prevent RAM explosion.
    """
    paths: list[Path] = []
    for label in snapshot_labels:
        pattern = BRONZE_SUBDIR / label / "**" / "*boxscores*.parquet"
        for path_str in glob(str(pattern), recursive=True):
            paths.append(Path(path_str))
    if not paths:
        raise FileNotFoundError("No bronze boxscores parquet files found. Run `rebuild_bronze.py` first.")

    # Use lazy evaluation: scan parquet files and apply transformations lazily
    lazy_frames: list[pl.LazyFrame] = []
    for path in paths:
        lazy_df = pl.scan_parquet(str(path))
        
        # Handle column aliases and additions
        for col_name, alias in [
            ("ingest_ts", "bronze_ingest_ts"),
            ("source_snapshot", "bronze_source_snapshot"),
            ("source_file", "bronze_source_file"),
        ]:
            if col_name in lazy_df.schema:
                if alias in lazy_df.schema:
                    lazy_df = lazy_df.drop(alias)
                lazy_df = lazy_df.rename({col_name: alias})
            elif alias not in lazy_df.schema:
                lazy_df = lazy_df.with_columns(pl.lit(None).alias(alias))
        
        # Drop columns if they exist
        drops = [col for col in ("ingest_ts", "source_snapshot", "source_file") if col in lazy_df.schema]
        if drops:
            lazy_df = lazy_df.drop(drops)
        
        # Cast columns
        lazy_df = lazy_df.with_columns([
            pl.col("game_id").cast(pl.Utf8, strict=False).alias("game_id"),
            pl.col("team_id").cast(pl.Utf8, strict=False).alias("team_id"),
            pl.col("person_id").cast(pl.Utf8, strict=False).alias("person_id"),
        ])
        
        lazy_frames.append(lazy_df)

    # Concatenate lazy frames (much more memory efficient)
    combined = pl.concat(lazy_frames, how="diagonal_relaxed")
    return combined


NUMERIC_TYPES = {
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Decimal,
}


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return any(isinstance(dtype, numeric_type) for numeric_type in NUMERIC_TYPES)


def _team_alias(name: str) -> str:
    if name in METADATA_COLUMNS:
        return name
    return name if name.startswith("team_") else f"team_{name}"


def _build_aggregations(
    schema: Dict[str, pl.DataType],
    field_rules: Dict[str, FieldRule],
) -> Tuple[List[pl.Expr], List[pl.Expr], List[Tuple[str, str, str]]]:
    sum_exprs: List[pl.Expr] = []
    first_exprs: List[pl.Expr] = []
    weighted_specs: List[Tuple[str, str, str]] = []
    aggregated_aliases: set[str] = set()

    sum_exprs.append(pl.col("minutes_float").sum().alias("team_minutes_total"))
    first_exprs.append(pl.count().alias("player_count"))
    aggregated_aliases.add("team_minutes_total")

    for name, dtype in schema.items():
        if name in GROUP_KEYS or name in PLAYER_ID_COLUMNS or name in {"minutes", "minutes_float"}:
            continue
        alias = _team_alias(name)
        rule = field_rules.get(name)
        if not rule:
            rule = FieldRule(strategy="first")

        if rule.strategy == "sum" and _is_numeric_dtype(dtype):
            if alias not in aggregated_aliases:
                sum_exprs.append(pl.col(name).sum().alias(alias))
                aggregated_aliases.add(alias)
        elif rule.strategy == "weighted_minutes" and _is_numeric_dtype(dtype):
            weight_col = rule.weight_col or "minutes_float"
            weight_alias = (
                "team_minutes_total" if weight_col == "minutes_float" else _team_alias(weight_col)
            )
            if weight_alias not in aggregated_aliases:
                sum_exprs.append(pl.col(weight_col).sum().alias(weight_alias))
                aggregated_aliases.add(weight_alias)
            temp_alias = f"__weighted_sum__{alias}"
            sum_exprs.append((pl.col(name) * pl.col(weight_col)).sum().alias(temp_alias))
            weighted_specs.append((alias, temp_alias, weight_alias))
        else:
            first_exprs.append(pl.col(name).first().alias(alias))

    return sum_exprs, first_exprs, weighted_specs


def _season_from_game_id(game_id: pl.Expr) -> pl.Expr:
    season_code = game_id.cast(pl.Utf8, strict=False).str.zfill(10).str.slice(3, 2).cast(pl.Int64)
    start_year = pl.when(season_code >= 50).then(season_code + 1900).otherwise(season_code + 2000)
    end_suffix = ((start_year + 1) % 100).cast(pl.Int64)
    return (
        pl.concat_str(
            start_year.cast(pl.Utf8),
            pl.lit("-"),
            end_suffix.cast(pl.Utf8).str.zfill(2),
        ).alias("season")
    )


def build_team_boxscores_agg(
    *,
    snapshot_labels: Sequence[str] = ("raw", "raw_last"),
    ingest_ts: str | None = None,
) -> Path:
    """
    Build the silver team_boxscores_agg dataset by aggregating player-level boxscores.
    """

    configure_logging(settings.data_paths.logs_root, pipeline="build_team_boxscores_agg")

    ingest_ts = ingest_ts or _default_ingest_ts()
    logger.info("team_boxscores_agg_start", ingest_ts=ingest_ts, snapshots=list(snapshot_labels))

    boxscores = _load_boxscores(snapshot_labels)

    df = boxscores.with_columns(
        [
            pl.col("game_id").cast(pl.Utf8),
            pl.col("team_id").cast(pl.Utf8),
            pl.col("person_id").cast(pl.Utf8, strict=False),
            minutes_to_float(pl.col("minutes")).alias("minutes_float"),
            _season_from_game_id(pl.col("game_id")).alias("season"),
        ]
    )

    schema = df.schema
    field_rules = build_rules_from_schema(schema.keys())
    sum_exprs, first_exprs, weighted_specs = _build_aggregations(schema, field_rules)

    seasons_df = df.select(pl.col("season").unique()).collect(streaming=True)
    seasons = seasons_df["season"].to_list() if "season" in seasons_df.columns else []

    logger.info(
        "team_boxscores_agg_season_list",
        seasons_count=len(seasons),
        seasons=sorted(seasons),
    )

    total_rows = 0
    written_seasons = 0
    for season in seasons:
        season_lazy = (
            df.filter(pl.col("season") == pl.lit(season))
            .group_by(GROUP_KEYS)
            .agg(sum_exprs + first_exprs)
            .with_columns(pl.lit(season).alias("season"))
        )

        extra_weight_aliases: set[str] = set()
        for alias, temp_alias, weight_alias in weighted_specs:
            season_lazy = season_lazy.with_columns(
                pl.when(pl.col(weight_alias) > 0)
                .then(pl.col(temp_alias) / pl.col(weight_alias))
                .otherwise(None)
                .alias(alias)
            ).drop(temp_alias)
            if weight_alias not in {"team_minutes_total"}:
                extra_weight_aliases.add(weight_alias)

        if extra_weight_aliases:
            season_lazy = season_lazy.drop(list(extra_weight_aliases))

        for numerator, denominator, alias in DERIVED_RATIOS:
            season_lazy = season_lazy.with_columns(
                pl.when(pl.col(denominator) > 0)
                .then(pl.col(numerator) / pl.col(denominator))
                .otherwise(None)
                .alias(alias)
            )

        season_lazy = season_lazy.with_columns(pl.col("team_minutes_total").round(3))

        season_df = season_lazy.collect(streaming=True).rename({"team_minutes_total": "team_minutes"})
        if season_df.is_empty():
            continue

        for meta_col in ("bronze_ingest_ts", "bronze_source_snapshot", "bronze_source_file"):
            if meta_col not in season_df.columns:
                season_df = season_df.with_columns(pl.lit(None).alias(meta_col))

        write_partitioned(
            season_df,
            OUTPUT_ROOT,
            ingest_ts,
            partition_cols=["season"],
        )
        total_rows += season_df.height
        written_seasons += 1

    if written_seasons == 0:
        raise RuntimeError("team_boxscores_agg: no rows produced. Check bronze inputs.")

    logger.info(
        "team_boxscores_agg_complete",
        rows=total_rows,
        seasons=written_seasons,
    )
    return OUTPUT_ROOT
