from __future__ import annotations

import pandas as pd

from src.config import (
    COLS_TO_DROP_TARGET_IS_WIN,
    COLS_TO_DROP_TARGET_POINT_DIFF,
    MATCH_SIDE_PREFIXES,
    RESULT_BASE_COLUMNS,
)


def add_point_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure POINT_DIFF / POINT_TOTAL targets exist based on the match representation."""

    if "HOME_POINTS_FOR" not in df.columns or "AWAY_POINTS_FOR" not in df.columns:
        raise KeyError("Expected HOME_POINTS_FOR and AWAY_POINTS_FOR columns to derive targets.")

    result = df.copy()
    result["POINTS_FOR"] = result["HOME_POINTS_FOR"]
    result["POINTS_AGAINST"] = result["AWAY_POINTS_FOR"]
    result["POINT_DIFF"] = result["POINTS_FOR"] - result["POINTS_AGAINST"]
    result["POINT_TOTAL"] = result["POINTS_FOR"] + result["POINTS_AGAINST"]
    result["IS_WIN"] = result["POINT_DIFF"].gt(0).astype(int)

    drop_cols = []
    for base in RESULT_BASE_COLUMNS:
        for side in MATCH_SIDE_PREFIXES:
            drop_cols.append(f"{side}_{base}")
        drop_cols.append(f"DIFF_{base}")
    result = result.drop(columns=drop_cols, errors="ignore")
    return result


def drop_for_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop identifiers/targets that should not be seen when training a given head."""

    target = target.upper()
    target_map = {
        "IS_WIN": COLS_TO_DROP_TARGET_IS_WIN,
        "POINT_DIFF": COLS_TO_DROP_TARGET_POINT_DIFF,
        "POINT_TOTAL": COLS_TO_DROP_TARGET_POINT_DIFF + ["POINT_DIFF"],
    }
    cols = target_map.get(target, [])
    return df.drop(columns=cols, errors="ignore")
