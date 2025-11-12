from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd
from feast import Field
from feast.types import Bool, Float32, Int64

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_EXPORT_PATH = PROJECT_ROOT / "data" / "feast_sources" / "silver_latest.parquet"

DEFAULT_EXCLUDE_COLUMNS: Set[str] = {
    "match_id",
    "GAME_ID",
    "GAME_DATE",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "POINT_TOTAL",
    "POINT_DIFF",
    "IS_WIN",
    "POINTS_FOR",
    "POINTS_AGAINST",
}


def _dtype_to_feast_type(dtype) -> Optional[type]:
    if pd.api.types.is_bool_dtype(dtype):
        return Bool
    if pd.api.types.is_integer_dtype(dtype):
        return Int64
    if pd.api.types.is_float_dtype(dtype):
        return Float32
    return None


def load_silver_dataframe(path: Path = SILVER_EXPORT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feast silver export not found at {path}. Run "
            "`python -m src.feast.pipeline ...` to generate it before applying Feast."
        )
    return pd.read_parquet(path)


def infer_feature_fields(
    df: pd.DataFrame,
    *,
    exclude_columns: Optional[Sequence[str]] = None,
) -> List[Field]:
    exclude = set(DEFAULT_EXCLUDE_COLUMNS)
    if exclude_columns:
        exclude.update(exclude_columns)

    fields: List[Field] = []
    for col, dtype in df.dtypes.items():
        if col in exclude:
            continue
        feast_type = _dtype_to_feast_type(dtype)
        if feast_type is None:
            # Skip categorical/object columns for now.
            continue
        fields.append(Field(name=col, dtype=feast_type))
    return fields


def load_silver_feature_fields(
    *,
    path: Path = SILVER_EXPORT_PATH,
    exclude_columns: Optional[Sequence[str]] = None,
) -> List[Field]:
    df = load_silver_dataframe(path)
    return infer_feature_fields(df, exclude_columns=exclude_columns)
