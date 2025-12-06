from __future__ import annotations

from typing import Iterable, List, Set

import numpy as np
import pandas as pd

from src.config import (
    MATCH_IDENTIFIER_COLUMNS,
    PRUNE_DROP_ZERO_VARIANCE,
    PRUNE_FEATURES_ENABLED,
    PRUNE_MIN_NON_MISSING,
    PRUNE_MISSING_THRESHOLD,
)


def prune_low_quality_features(
    df: pd.DataFrame,
    *,
    enabled: bool = PRUNE_FEATURES_ENABLED,
    missing_threshold: float = PRUNE_MISSING_THRESHOLD,
    min_non_missing: int = PRUNE_MIN_NON_MISSING,
    drop_zero_variance: bool = PRUNE_DROP_ZERO_VARIANCE,
    protect: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Drop feature columns with too many missing values or zero variance.

    missing_threshold: drop if fraction of missing > threshold.
    min_non_missing: drop if count of non-missing < min_non_missing.
    drop_zero_variance: drop numeric columns with zero variance (ignoring NaN).
    protect: columns that must never be dropped.
    """

    if not enabled:
        return df

    protect_set: Set[str] = set(protect or [])
    protect_set.update(MATCH_IDENTIFIER_COLUMNS)
    protect_set.update(
        {
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
            "POINT_TOTAL",
            "POINT_DIFF",
            "POINTS_FOR",
            "POINTS_AGAINST",
            "HOME_POINTS_FOR",
            "AWAY_POINTS_FOR",
            "HOME_IS_WIN",
            "AWAY_IS_WIN",
            "IS_WIN",
            "SEASON",
            "GAME_DATE",
        }
    )

    drop_cols: List[str] = []
    total = len(df)
    if total == 0:
        return df

    non_missing = df.notna().sum()
    missing_frac = 1 - (non_missing / total)

    for col in df.columns:
        if col in protect_set:
            continue
        # Missingness pruning
        if missing_frac[col] > missing_threshold or non_missing[col] < min_non_missing:
            drop_cols.append(col)
            continue
        # Zero variance pruning
        if drop_zero_variance and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if series.empty:
                drop_cols.append(col)
            else:
                if np.nan_to_num(series.std(ddof=0)) == 0:
                    drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df
