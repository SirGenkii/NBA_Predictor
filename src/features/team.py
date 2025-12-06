from __future__ import annotations

import pandas as pd

from src.config import BASE_TEAM_FEATURE_COLUMNS, features_to_roll
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
from .availability import apply_availability_features
from .reshaping import attach_team_features, match_rows_to_team_rows


def apply_team_history_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Run the team-history pipeline (rollings, Elo, streaks) and project back to match rows."""

    team_view = match_rows_to_team_rows(match_df, include_cols=BASE_TEAM_FEATURE_COLUMNS)
    base_columns = set(team_view.columns)

    team_view = _add_shifted_targets(team_view)
    team_view = _add_rest_features(team_view)
    team_view = _add_win_features(team_view)
    team_view = _add_scoring_rollups(team_view)
    team_view = _add_generic_rollings(team_view)
    team_view = _add_h2h_features(team_view)
    team_view = _add_elo_features(team_view)
    feature_cols = [col for col in team_view.columns if col not in base_columns]
    match_df = attach_team_features(match_df, team_view, feature_cols=feature_cols)
    match_df = apply_availability_features(match_df)
    return match_df


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _add_shifted_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    result["IS_WIN_SHIFTED"] = result.groupby("TEAM_ID")["IS_WIN"].shift(1).fillna(0).astype(int)
    return result


def _add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["DAYS_SINCE_LAST_GAME"] = compute_rest_days(result, "GAME_DATE", "TEAM_ID")
    result["OPP_DAYS_SINCE_LAST_GAME"] = compute_rest_days(result, "GAME_DATE", "OPP_TEAM_ID")
    result["REST_ADVANTAGE"] = result["DAYS_SINCE_LAST_GAME"] - result["OPP_DAYS_SINCE_LAST_GAME"]
    # Fatigue flags
    result["IS_B2B"] = (result["DAYS_SINCE_LAST_GAME"] <= 1).astype(int)
    prev_3 = result.groupby("TEAM_ID")["GAME_DATE"].shift(3)
    result["IS_3IN4"] = ((result["GAME_DATE"] - prev_3).dt.days <= 4).astype(int).fillna(0)
    for n in [3, 5, 10]:
        result[f"ROLL_B2B_RATE_{n}"] = (
            result.groupby("TEAM_ID")["IS_B2B"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
            .fillna(0)
        )
        result[f"ROLL_3IN4_RATE_{n}"] = (
            result.groupby("TEAM_ID")["IS_3IN4"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
            .fillna(0)
        )
    result = compute_rolling_rest_advantage(result, "TEAM_ID", "IS_HOME", "REST_ADVANTAGE", windows=[3, 5, 10])
    return result


def _add_win_features(df: pd.DataFrame) -> pd.DataFrame:
    result = compute_win_ratio(df, "TEAM_ID", "IS_WIN", windows=[3, 5, 10, 25, 50, 100, 200])
    result["WIN_STREAK"] = compute_win_streak(result, "TEAM_ID", "IS_WIN_SHIFTED")
    return result


def _add_scoring_rollups(df: pd.DataFrame) -> pd.DataFrame:
    result = compute_home_away_pts(
        df,
        group_col="TEAM_ID",
        is_home_col="IS_HOME",
        pts_col="points_traditional",
        opp_pts_col="OPP_points_traditional",
        windows=[3, 5, 10, 25, 50, 100, 200],
    )
    return rename_pts_against_columns(result)


def _add_generic_rollings(df: pd.DataFrame) -> pd.DataFrame:
    return compute_rolling_features(
        df,
        group_col="TEAM_ID",
        sort_cols=["TEAM_ID", "GAME_DATE"],
        value_cols=features_to_roll,
        windows=[3, 5, 10, 25, 50, 100, 200],
        method="ewm",
    )


def _add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    result = compute_h2h(df, windows=[3, 5, 10, 25, 50])
    result = compute_h2h_pts_margin(result, windows=[3, 5, 10, 25, 50])
    result = compute_h2h_season(result)
    result = compute_h2h_streak(result)
    return result


def _add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    result = compute_elo(df)
    result = compute_elo_season(result)
    result = convert_elos_to_elo_diff(result)
    return result
