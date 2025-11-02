from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import polars as pl


def season_label_from_id(season_id: pl.Expr, fallback: pl.Expr | None = None) -> pl.Expr:
    start_year = season_id.cast(pl.Int64) % 10000
    end_suffix = ((start_year + 1) % 100).cast(pl.Int64)
    formatted = pl.concat_str(
        start_year.cast(pl.Utf8),
        pl.lit("-"),
        end_suffix.cast(pl.Utf8).str.zfill(2),
    )
    if fallback is None:
        return formatted
    return pl.when(season_id.is_not_null()).then(formatted).otherwise(fallback)


def minutes_to_float(column: pl.Expr) -> pl.Expr:
    iso_pattern = r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?"

    def _parse_iso(expr: pl.Expr) -> pl.Expr:
        minutes = expr.str.extract(iso_pattern, group_index=1).cast(pl.Float64).fill_null(0.0)
        seconds = expr.str.extract(iso_pattern, group_index=2).cast(pl.Float64).fill_null(0.0)
        return minutes + seconds / 60.0

    def _parse_mm_ss(expr: pl.Expr) -> pl.Expr:
        parts = expr.str.split(":")
        minutes = parts.list.get(0).cast(pl.Float64).fill_null(0.0)
        seconds = parts.list.get(1).cast(pl.Float64).fill_null(0.0)
        return minutes + seconds / 60.0

    return (
        pl.when(column.is_null() | (column == "") | (column == "0"))
        .then(0.0)
        .when(column.str.starts_with("PT"))
        .then(_parse_iso(column))
        .when(column.str.contains(":"))
        .then(_parse_mm_ss(column))
        .otherwise(column.cast(pl.Float64))
        .alias(column.meta.output_name())
    )


def write_partitioned(
    df: pl.DataFrame,
    output_root: Path,
    ingest_ts: str,
    partition_cols: Sequence[str],
    *,
    compression: str = "zstd",
) -> None:
    if df.is_empty():
        return

    unique_partitions = df.select([pl.col(col) for col in partition_cols]).unique()

    for row in unique_partitions.iter_rows(named=True):
        filters = [pl.col(col) == value for col, value in row.items()]
        subset = df.filter(filters[0] if len(filters) == 1 else pl.all(filters))
        partition_dir = output_root
        for col, value in row.items():
            partition_dir = partition_dir / f"{col}={value}"
        partition_dir = partition_dir / f"ingest_ts={ingest_ts}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"part-{uuid4().hex}.parquet"
        subset.write_parquet(partition_dir / file_name, compression=compression)
