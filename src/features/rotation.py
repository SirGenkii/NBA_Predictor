from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MATCHUP_WINDOWS
from src.feature_builder import compute_rolling_features
from .reshaping import attach_team_features, match_rows_to_team_rows


def apply_rotation_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate rotation/bench usage metrics and attach them to match rows."""

    include_cols = [
        "MINUTES_PLAYED",
        "num_present",
        "top_player_count",
        "player_perf_score_sum",
        "player_perf_score_mean",
    ]
    team_view = match_rows_to_team_rows(match_df, include_cols=include_cols)
    base_columns = set(team_view.columns)

    team_view = _add_rotation_metrics(team_view)
    team_view = _add_rotation_rollings(team_view)
    team_view = team_view.drop(
        columns=[
            "AVG_MINUTES_PER_PLAYER",
            "BENCH_PLAYERS",
            "BENCH_USAGE_RATIO",
            "TOP_PLAYER_SHARE",
            "TOP_USAGE_SCORE",
            "BENCH_USAGE_SCORE",
        ],
        errors="ignore",
    )

    feature_cols = [col for col in team_view.columns if col not in base_columns]
    if not feature_cols:
        return match_df

    return attach_team_features(match_df, team_view, feature_cols=feature_cols)


def _add_rotation_metrics(team_df: pd.DataFrame) -> pd.DataFrame:
    result = team_df.copy()
    players = result["num_present"].replace(0, np.nan)
    top_players = result["top_player_count"].replace(0, np.nan)

    result["AVG_MINUTES_PER_PLAYER"] = result["MINUTES_PLAYED"] / players
    result["BENCH_PLAYERS"] = result["num_present"] - result["top_player_count"]
    result["BENCH_USAGE_RATIO"] = result["BENCH_PLAYERS"] / players
    result["TOP_PLAYER_SHARE"] = top_players / players
    result["TOP_USAGE_SCORE"] = result["player_perf_score_sum"].fillna(0) * result["TOP_PLAYER_SHARE"].fillna(0)
    result["BENCH_USAGE_SCORE"] = result["player_perf_score_sum"].fillna(0) * result["BENCH_USAGE_RATIO"].fillna(0)

    fill_cols = [
        "AVG_MINUTES_PER_PLAYER",
        "BENCH_PLAYERS",
        "BENCH_USAGE_RATIO",
        "TOP_PLAYER_SHARE",
        "TOP_USAGE_SCORE",
        "BENCH_USAGE_SCORE",
    ]
    result[fill_cols] = result[fill_cols].fillna(0)
    return result


def _add_rotation_rollings(team_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "AVG_MINUTES_PER_PLAYER",
        "BENCH_PLAYERS",
        "BENCH_USAGE_RATIO",
        "TOP_PLAYER_SHARE",
        "TOP_USAGE_SCORE",
        "BENCH_USAGE_SCORE",
    ]
    return compute_rolling_features(
        team_df,
        group_col="TEAM_ID",
        sort_cols=["TEAM_ID", "GAME_DATE"],
        value_cols=value_cols,
        windows=MATCHUP_WINDOWS,
        method="ewm",
    )
