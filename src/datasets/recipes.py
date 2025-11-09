from __future__ import annotations

from typing import Callable, List

import pandas as pd

from src.config import (
    COLS_MATCH_REAL,
    COLS_ODDS,
    COLS_TO_DROP_TARGET_IS_WIN,
    COLS_TO_DROP_TARGET_POINT_DIFF,
    features_to_roll,
    top_player_features_to_roll,
)
from src.feature_builder import (
    compute_elo,
    compute_elo_season,
    compute_h2h,
    compute_h2h_pts_margin,
    compute_h2h_season,
    compute_h2h_streak,
    compute_home_away_pts,
    compute_rest_days,
    compute_rolling_features,
    compute_rolling_rest_advantage,
    compute_win_ratio,
    compute_win_streak,
    convert_elos_to_elo_diff,
    rename_pts_against_columns,
)


# ---------------------------------------------------------------------------
# Feature steps
# ---------------------------------------------------------------------------

def add_shifted_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create leak-safe shifted columns (win/result of previous game)."""

    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    df["IS_WIN_SHIFTED"] = df.groupby("TEAM_ID")["IS_WIN"].shift(1).fillna(0).astype(int)
    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw rest days and rolling rest advantage."""

    df = df.copy()
    df["DAYS_SINCE_LAST_GAME"] = compute_rest_days(df, "GAME_DATE", "TEAM_ID")
    df["OPP_DAYS_SINCE_LAST_GAME"] = compute_rest_days(df, "GAME_DATE", "OPP_TEAM_ID")
    df["REST_ADVANTAGE"] = df["DAYS_SINCE_LAST_GAME"] - df["OPP_DAYS_SINCE_LAST_GAME"]
    df = compute_rolling_rest_advantage(df, "TEAM_ID", "IS_HOME", "REST_ADVANTAGE", windows=[3, 5, 10])
    return df


def add_win_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add win ratios and streak features based on shifted results."""

    df = df.copy()
    df = compute_win_ratio(df, "TEAM_ID", "IS_WIN", windows=[3, 5, 10, 25, 50, 100, 200])
    df["WIN_STREAK"] = compute_win_streak(df, "TEAM_ID", "IS_WIN_SHIFTED")
    return df


def add_home_away_pts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling scoring features split by home/away."""

    df = compute_home_away_pts(
        df,
        group_col="TEAM_ID",
        is_home_col="IS_HOME",
        pts_col="points_traditional",
        opp_pts_col="OPP_points_traditional",
        windows=[3, 5, 10, 25, 50, 100, 200],
    )
    return rename_pts_against_columns(df)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling/ewm statistics to team & opponent boxscore aggregates."""

    df = compute_rolling_features(
        df,
        group_col="TEAM_ID",
        sort_cols=["TEAM_ID", "GAME_DATE"],
        value_cols=features_to_roll,
        windows=[3, 5, 10, 25, 50, 100, 200],
        method="ewm",
    )
    return df


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head aggregates across various windows + season stats."""

    df = compute_h2h(df, windows=[3, 5, 10, 25, 50])
    df = compute_h2h_pts_margin(df, windows=[3, 5, 10, 25, 50])
    df = compute_h2h_season(df)
    df = compute_h2h_streak(df)
    return df


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings (global + season) and convert to diffs."""

    df = compute_elo(df)
    df = compute_elo_season(df)
    df = convert_elos_to_elo_diff(df)
    return df


def add_point_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create explicit scoring targets from raw PTS columns (or existing point fields)."""

    df = df.copy()
    pts_for = (
        df["PTS"]
        if "PTS" in df.columns
        else df["points_traditional"]
        if "points_traditional" in df.columns
        else df.get("POINTS_FOR")
    )
    pts_against = (
        df["OPP_PTS"]
        if "OPP_PTS" in df.columns
        else df["OPP_points_traditional"]
        if "OPP_points_traditional" in df.columns
        else df.get("POINTS_AGAINST")
    )

    if pts_for is None or pts_against is None:
        raise KeyError("Unable to derive POINTS_FOR/AGAINST (missing PTS columns in dataset).")

    df["POINTS_FOR"] = pts_for
    df["POINTS_AGAINST"] = pts_against
    df["POINT_DIFF"] = df["POINTS_FOR"] - df["POINTS_AGAINST"]
    df["POINT_TOTAL"] = df["POINTS_FOR"] + df["POINTS_AGAINST"]
    return df


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that contain post-match info or engineered history."""

    drop_cols = set(COLS_MATCH_REAL + features_to_roll + top_player_features_to_roll)
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def drop_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bookmaker odds columns when preparing gold datasets."""

    return df.drop(columns=COLS_ODDS, errors="ignore")


def drop_for_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop identifiers/targets that should not be seen when training a given head."""

    target_map = {
        "IS_WIN": COLS_TO_DROP_TARGET_IS_WIN,
        "POINT_DIFF": COLS_TO_DROP_TARGET_POINT_DIFF,
        "POINT_TOTAL": COLS_TO_DROP_TARGET_POINT_DIFF + ["POINT_DIFF"],
    }
    cols = target_map.get(target.upper())
    if not cols:
        return df
    return df.drop(columns=cols, errors="ignore")


# ---------------------------------------------------------------------------
# Default step collections
# ---------------------------------------------------------------------------

DEFAULT_SILVER_FEATURE_STEPS: List[Callable[[pd.DataFrame], pd.DataFrame]] = [
    add_shifted_targets,
    add_rest_features,
    add_win_features,
    add_home_away_pts,
    add_rolling_features,
    add_h2h_features,
    add_elo_features,
]

DEFAULT_SILVER_TARGET_STEPS: List[Callable[[pd.DataFrame], pd.DataFrame]] = [
    add_point_targets,
]

def gold_steps_for_target(target: str) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    """Factory returning the default cleaning steps for a given target."""

    def _drop_target_columns(df: pd.DataFrame) -> pd.DataFrame:
        return drop_for_target(df, target)

    return [
        drop_odds_columns,
        drop_leakage_columns,
        _drop_target_columns,
    ]
