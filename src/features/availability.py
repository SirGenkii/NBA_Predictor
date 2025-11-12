from __future__ import annotations

import pandas as pd

from src.config import PLAYER_AVAILABILITY_BASE, PLAYER_AVAILABILITY_WINDOWS
from .reshaping import attach_team_features, match_rows_to_team_rows


def apply_availability_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-safe rolling availability stats and project them back to match rows."""

    team_view = match_rows_to_team_rows(match_df, include_cols=PLAYER_AVAILABILITY_BASE)
    base_columns = set(team_view.columns)
    team_view = _add_availability_rollups(team_view)
    feature_cols = [col for col in team_view.columns if col not in base_columns]
    if not feature_cols:
        return match_df
    return attach_team_features(match_df, team_view, feature_cols=feature_cols)


def _add_availability_rollups(team_df: pd.DataFrame) -> pd.DataFrame:
    result = team_df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    numeric_cols = [col for col in PLAYER_AVAILABILITY_BASE if col in result.columns]

    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        group = result.groupby("TEAM_ID")[col]
        for window in PLAYER_AVAILABILITY_WINDOWS:
            result[f"ROLL_{col}_{window}"] = group.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    drop_cols = numeric_cols + [f"OPP_{col}" for col in numeric_cols]
    result = result.drop(columns=drop_cols, errors="ignore")
    return result
