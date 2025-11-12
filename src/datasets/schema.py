from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

SCHEMA_PREFIXES: List[str] = [
    "HOME_",
    "AWAY_",
    "DIFF_",
    "MATCH_",
    "TOTAL_",
    "ODDS_",
    "IMPLIED_",
]


def infer_prefix(column: str) -> str:
    for prefix in SCHEMA_PREFIXES:
        if column.startswith(prefix):
            return prefix.rstrip("_")
    return "BASE"


def build_schema(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "stage": stage,
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "null_pct": float(series.isna().mean() * 100),
                "prefix": infer_prefix(column),
            }
        )
    return pd.DataFrame(rows)


def save_schema(report: pd.DataFrame, path: Path, fmt: str = "csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        report.to_csv(path, index=False)
    elif fmt in {"md", "markdown"}:
        path.write_text(report.to_markdown(index=False))
    else:
        raise ValueError(f"Unsupported schema format '{fmt}'")
    return path
