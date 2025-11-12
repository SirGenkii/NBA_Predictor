from __future__ import annotations

from typing import List

import pandas as pd

from src.config import MATCHUP_WINDOWS


def add_matchup_scoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute matchup-level aggregates combining HOME_/AWAY_ rolling stats."""

    result = df.copy()

    for window in MATCHUP_WINDOWS:
        _add_pace_features(result, window)
        _add_scoring_features(result, window)
        _add_rating_gaps(result, window)
        _add_availability_gaps(result, window)

    _add_elo_matchup(result)

    return result


def _add_pace_features(df: pd.DataFrame, window: int) -> None:
    home_col = f"HOME_ROLL_pace_advanced_{window}"
    away_col = f"AWAY_ROLL_pace_advanced_{window}"
    if home_col in df.columns and away_col in df.columns:
        df[f"MATCH_PACE_{window}"] = (df[home_col] + df[away_col]) / 2
        df[f"PACE_DIFF_{window}"] = df[home_col] - df[away_col]


def _add_scoring_features(df: pd.DataFrame, window: int) -> None:
    home_pts = f"HOME_ROLL_points_traditional_{window}"
    away_pts = f"AWAY_ROLL_points_traditional_{window}"

    if home_pts in df.columns and away_pts in df.columns:
        df[f"TOTAL_POINTS_EXPECTED_{window}"] = df[home_pts] + df[away_pts]
        df[f"POINTS_DIFF_EXPECTED_{window}"] = df[home_pts] - df[away_pts]


def _add_rating_gaps(df: pd.DataFrame, window: int) -> None:
    home_off = f"HOME_ROLL_offensiveRating_advanced_{window}"
    away_def = f"AWAY_ROLL_defensiveRating_advanced_{window}"
    away_off = f"AWAY_ROLL_offensiveRating_advanced_{window}"
    home_def = f"HOME_ROLL_defensiveRating_advanced_{window}"

    if home_off in df.columns and away_def in df.columns:
        df[f"MATCH_OFF_DEF_GAP_{window}"] = df[home_off] - df[away_def]

    if away_off in df.columns and home_def in df.columns:
        df[f"MATCH_DEF_VS_OPP_OFF_{window}"] = df[home_def] - df[away_off]


def _add_elo_matchup(df: pd.DataFrame) -> None:
    if "HOME_ELO_DIFF" in df.columns:
        df["MATCH_ELO_GAP"] = df["HOME_ELO_DIFF"]

    if "HOME_ELO_DIFF_SEASON" in df.columns:
        df["MATCH_ELO_GAP_SEASON"] = df["HOME_ELO_DIFF_SEASON"]

    if "HOME_ELO_MEAN" in df.columns and "AWAY_ELO_MEAN" in df.columns:
        df["MATCH_ELO_LEVEL_AVG"] = (df["HOME_ELO_MEAN"] + df["AWAY_ELO_MEAN"]) / 2


def _add_availability_gaps(df: pd.DataFrame, window: int) -> None:
    rate = f"ROLL_top_player_absent_rate_{window}"
    home_col = f"HOME_{rate}"
    away_col = f"AWAY_{rate}"
    if home_col in df.columns and away_col in df.columns:
        df[f"MATCH_AVAILABILITY_GAP_{window}"] = df[home_col] - df[away_col]
