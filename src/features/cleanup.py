from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd

from src.config import (
    BASE_TEAM_FEATURE_COLUMNS,
    MATCH_ALLOWED_BASE_COLUMNS,
    MATCH_ALLOWED_PREFIXES,
    cols_player_stats,
    cols_to_sum,
    cols_to_weighted_avg,
)


SYMMETRIC_BASES = [
    "ELO_DIFF",
    "ELO_DIFF_SEASON",
    "ELO_MEAN",
    "ELO_MEAN_SEASON",
    "ROLL_WIN_RATIO_3",
    "ROLL_WIN_RATIO_5",
    "ROLL_WIN_RATIO_10",
    "ROLL_WIN_RATIO_25",
    "ROLL_WIN_RATIO_50",
    "ROLL_WIN_RATIO_100",
    "ROLL_WIN_RATIO_200",
]


def drop_raw_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove per-game raw stats once rolling features have been materialized."""

    drop_cols: List[str] = []
    extra_bases = {"ELO_DIFF", "ELO_DIFF_SEASON", "ELO_MEAN", "ELO_MEAN_SEASON"}
    raw_inputs = set(cols_to_sum + cols_to_weighted_avg + cols_player_stats + BASE_TEAM_FEATURE_COLUMNS)
    raw_inputs.update(extra_bases)
    for col in raw_inputs:
        drop_cols.extend([f"HOME_{col}", f"AWAY_{col}", f"DIFF_{col}"])

    for base in SYMMETRIC_BASES:
        drop_cols.extend([f"DIFF_{base}", f"AWAY_{base}"])

    return df.drop(columns=drop_cols, errors="ignore")


def drop_helper_columns(df: pd.DataFrame, helpers: Sequence[str] | None = None) -> pd.DataFrame:
    """Drop intermediate helper columns such as shifted targets."""

    helpers = helpers or ["IS_WIN_SHIFTED"]
    drop_cols: List[str] = []
    for helper in helpers:
        drop_cols.extend([f"HOME_{helper}", f"AWAY_{helper}", f"DIFF_{helper}"])
    return df.drop(columns=drop_cols, errors="ignore")


def enforce_feature_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ID columns + prefixed engineered fields."""

    allowed = set(MATCH_ALLOWED_BASE_COLUMNS)
    keep = [col for col in df.columns if col in allowed or col.startswith(MATCH_ALLOWED_PREFIXES)]
    return df[keep]
