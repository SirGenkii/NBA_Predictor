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
METADATA_COLUMNS = {"source_snapshot", "bronze_ingest_ts"}


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
    scans = []
    for label in snapshot_labels:
        base_pattern = BRONZE_SUBDIR / label / "**" / "*boxscores*.parquet"
        pattern_str = str(base_pattern)
        if glob(pattern_str, recursive=True):
            scans.append(pl.scan_parquet(pattern_str))
    if not scans:
        raise FileNotFoundError("No bronze boxscores parquet files found. Run `rebuild_bronze.py` first.")
    return pl.concat(scans)


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

    for meta in METADATA_COLUMNS:
        if meta in schema:
            first_exprs.append(pl.col(meta).first().alias(meta))

    return sum_exprs, first_exprs, weighted_specs


def _season_from_game_id(game_id: pl.Expr) -> pl.Expr:
    season_code = game_id.cast(pl.Utf8).str.slice(3, 2).cast(pl.Int64)
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

    df = (
        boxscores.select(pl.all())
        .with_columns(
            [
                pl.col("game_id").cast(pl.Utf8),
                pl.col("team_id").cast(pl.Int64),
                minutes_to_float(pl.col("minutes")).alias("minutes_float"),
                pl.col("ingest_ts").alias("bronze_ingest_ts"),
            ]
        )
        .drop("ingest_ts")
    )

    schema = df.schema
    field_rules = build_rules_from_schema(schema.keys())
    sum_exprs, first_exprs, weighted_specs = _build_aggregations(schema, field_rules)

    aggregated = (
        df.groupby(GROUP_KEYS)
        .agg(sum_exprs + first_exprs)
        .with_columns(_season_from_game_id(pl.col("game_id")))
    )

    extra_weight_aliases: set[str] = set()

    for alias, temp_alias, weight_alias in weighted_specs:
        aggregated = aggregated.with_columns(
            pl.when(pl.col(weight_alias) > 0)
            .then(pl.col(temp_alias) / pl.col(weight_alias))
            .otherwise(None)
            .alias(alias)
        ).drop(temp_alias)
        if weight_alias not in {"team_minutes_total"}:
            extra_weight_aliases.add(weight_alias)

    if extra_weight_aliases:
        aggregated = aggregated.drop(list(extra_weight_aliases))

    for numerator, denominator, alias in DERIVED_RATIOS:
        if numerator in aggregated.columns and denominator in aggregated.columns:
            aggregated = aggregated.with_columns(
                pl.when(pl.col(denominator) > 0)
                .then(pl.col(numerator) / pl.col(denominator))
                .otherwise(None)
                .alias(alias)
            )

    aggregated = aggregated.with_columns(pl.col("team_minutes_total").round(3))

    materialized = aggregated.collect()

    if materialized.is_empty():
        raise RuntimeError("team_boxscores_agg: no rows produced. Check bronze inputs.")

    drop_cols = [col for col in ("source_file", "ingest_ts") if col in materialized.columns]
    if drop_cols:
        materialized = materialized.drop(drop_cols)

    if "bronze_ingest_ts" not in materialized.columns and "source_snapshot" in materialized.columns:
        materialized = materialized.with_columns(pl.lit(None).alias("bronze_ingest_ts"))

    materialized = materialized.rename({"team_minutes_total": "team_minutes"})

    write_partitioned(
        materialized,
        OUTPUT_ROOT,
        ingest_ts,
        partition_cols=["season"],
    )

    logger.info(
        "team_boxscores_agg_complete",
        rows=materialized.height,
    )
    return OUTPUT_ROOT
