from __future__ import annotations

import pandas as pd

from src.config import (
    BASE_TEAM_FEATURE_COLUMNS,
    MATCHUP_WINDOWS,
    MATCH_CONTEXT_FLAGS,
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
from .reshaping import attach_team_features, match_rows_to_team_rows


def apply_team_history_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Run the team-history pipeline (rollings, Elo, streaks) and project back to match rows."""

    team_view = match_rows_to_team_rows(match_df, include_cols=BASE_TEAM_FEATURE_COLUMNS)
    base_columns = set(team_view.columns)

    team_view = _add_shifted_targets(team_view)
    team_view = _add_rest_features(team_view)
    team_view = _add_context_features(team_view)
    team_view = _add_win_features(team_view)
    team_view = _add_scoring_rollups(team_view)
    team_view = _add_generic_rollings(team_view)
    team_view = _add_top_player_rollings(team_view)
    team_view = _add_variance_rollings(team_view)
    team_view = _add_h2h_features(team_view)
    team_view = _add_elo_features(team_view)
    feature_cols = [col for col in team_view.columns if col not in base_columns]
    match_df = attach_team_features(match_df, team_view, feature_cols=feature_cols)
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
    result = _add_schedule_flags(result)
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


def _add_schedule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive schedule congestion indicators and short-term rollings."""

    result = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    result["IS_BACK_TO_BACK"] = (result["DAYS_SINCE_LAST_GAME"] <= 1).astype(int)

    group = result.groupby("TEAM_ID")
    two_games_back = group["GAME_DATE"].shift(2)
    four_games_back = group["GAME_DATE"].shift(4)

    result["IS_3IN4"] = (
        (result["GAME_DATE"] - two_games_back) <= pd.Timedelta(days=4)
    ).fillna(False).astype(int)
    result["IS_5IN7"] = (
        (result["GAME_DATE"] - four_games_back) <= pd.Timedelta(days=7)
    ).fillna(False).astype(int)

    for col in ("IS_BACK_TO_BACK", "IS_3IN4", "IS_5IN7"):
        for window in (5, 10):
            result[f"ROLL_{col}_{window}"] = group[col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    return result


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Roll tournament/playoff flags to quantify recent importance."""

    result = df.copy()
    flags = [flag for flag in MATCH_CONTEXT_FLAGS if flag in result.columns]
    if not flags:
        return result

    group = result.groupby("TEAM_ID")
    for flag in flags:
        for window in MATCHUP_WINDOWS:
            result[f"ROLL_{flag}_{window}"] = group[flag].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    return result


def _add_variance_rollings(df: pd.DataFrame) -> pd.DataFrame:
    """Capture volatility of shot profiles to gauge matchup variance."""

    variance_cols = [
        "percentageFieldGoalsAttempted3pt_scoring",
        "OPP_percentageFieldGoalsAttempted3pt_scoring",
        "percentagePoints3pt_scoring",
        "OPP_percentagePoints3pt_scoring",
        "points_traditional",
        "OPP_points_traditional",
        "pace_advanced",
        "OPP_pace_advanced",
        "possessions_advanced",
        "OPP_possessions_advanced",
    ]
    available = [col for col in variance_cols if col in df.columns]
    if not available:
        return df

    return compute_rolling_features(
        df,
        group_col="TEAM_ID",
        sort_cols=["TEAM_ID", "GAME_DATE"],
        value_cols=available,
        windows=[5, 10, 25],
        method="std",
        suffix="_STD",
    )


def _add_top_player_rollings(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in top_player_features_to_roll if col in df.columns]
    if not available:
        return df

    return compute_rolling_features(
        df,
        group_col="TEAM_ID",
        sort_cols=["TEAM_ID", "GAME_DATE"],
        value_cols=available,
        windows=[3, 5, 10, 25],
        method="ewm",
    )
