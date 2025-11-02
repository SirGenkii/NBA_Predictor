from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from feast import Field
from feast.types import Bool, Float32, Float64, Int32, Int64, String

_TYPE_MAPPING = {
    pl.Int32: Int32,
    pl.Int64: Int64,
    pl.UInt32: Int64,
    pl.UInt64: Int64,
    pl.Float32: Float32,
    pl.Float64: Float64,
    pl.Boolean: Bool,
}


def _dtype_to_feast(dtype: pl.DataType) -> type:
    for polars_type, feast_type in _TYPE_MAPPING.items():
        if isinstance(dtype, polars_type):
            return feast_type
    return String


def infer_fields_from_parquet(
    parquet_path: str | Path,
    *,
    exclude: Iterable[str] | None = None,
) -> list[Field]:
    path = Path(parquet_path)
    exclude_set = set(exclude or [])
    try:
        schema = pl.scan_parquet(str(path)).schema
    except FileNotFoundError as exc:  # pragma: no cover
        raise FileNotFoundError(
            f"Parquet path '{path}' not found. Generate silver data first."
        ) from exc

    fields: list[Field] = []
    for name, dtype in schema.items():
        if name in exclude_set:
            continue
        feast_type = _dtype_to_feast(dtype)
        fields.append(Field(name=name, dtype=feast_type))
    return fields
