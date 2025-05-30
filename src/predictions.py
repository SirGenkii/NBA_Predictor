import pandas as pd
import numpy as np
from datetime import datetime
from src.feature_builder import *
from src.config import FEATURES_TO_ROLL, DATA_ODDS_HISTORY_DIR, N_LIST
from src.utils import merge_odds_csv_files

def build_prediction_rows(home_team_id: int, away_team_id: int, home_odds: float, away_odds: float ,dataset: pd.DataFrame, match_date: datetime = None) -> pd.DataFrame:

    dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])
    temp_df = dataset.copy()

    if match_date is None:
        next_game_date = pd.to_datetime(temp_df['GAME_DATE'].max()) + pd.Timedelta(days=1)
    else:
        next_game_date = pd.to_datetime(match_date)

    next_season = temp_df.loc[temp_df['GAME_DATE'].idxmax(), 'SEASON']

    base_rows = [
            {
                "TEAM_ID": home_team_id,
                "OPP_TEAM_ID": away_team_id,
                "GAME_DATE": next_game_date,
                "SEASON": next_season,
                "IS_HOME": 1,
                "ODDS": home_odds,
                "OPP_ODDS": away_odds,
                "IS_WIN": 0
            },
            {
                "TEAM_ID": away_team_id,
                "OPP_TEAM_ID": home_team_id,
                "GAME_DATE": next_game_date,
                "SEASON": next_season,
                "IS_HOME": 0,
                "ODDS": away_odds,
                "OPP_ODDS": home_odds,
                "IS_WIN": 0
            }
        ]

    prediction_df = pd.DataFrame(base_rows)

    temp_df = pd.concat([temp_df, prediction_df], ignore_index=True)
    

    # Ajouter les features avancées aux stats de match
    temp_df = add_advanced_boxscore_features(temp_df)

    # Calcul des features glissantes shiftées
    temp_df = compute_rolling_features(temp_df, "TEAM_ID", ["TEAM_ID", "GAME_DATE"], FEATURES_TO_ROLL, N_LIST, method="ewm")


    temp_df = compute_winrates(temp_df, "TEAM_ID", "IS_WIN", "IS_HOME", N_LIST)
    temp_df = compute_win_ratio(temp_df, "TEAM_ID", "IS_WIN", N_LIST)

    temp_df["IS_WIN_SHIFTED"] = temp_df.groupby("TEAM_ID")["IS_WIN"].shift(1).fillna(0).astype(int)
    temp_df["WIN_STREAK"] = temp_df.groupby("TEAM_ID").apply(
        lambda x: compute_win_streak(x, "TEAM_ID", "IS_WIN_SHIFTED")).reset_index(level=0, drop=True)
    temp_df = compute_side_win_streak(temp_df)

    temp_df["DAYS_SINCE_LAST_GAME"] = compute_rest_days(temp_df, "GAME_DATE", "TEAM_ID")
    temp_df["OPP_DAYS_SINCE_LAST_GAME"] = compute_rest_days(temp_df, "GAME_DATE", "OPP_TEAM_ID")
    temp_df["REST_ADVANTAGE"] = temp_df["DAYS_SINCE_LAST_GAME"] - temp_df["OPP_DAYS_SINCE_LAST_GAME"]

    temp_df = compute_rolling_rest_advantage(temp_df, "TEAM_ID", "IS_HOME", "REST_ADVANTAGE", N_LIST)
    temp_df = compute_home_away_pts(temp_df, "TEAM_ID", "IS_HOME", "PTS", "OPP_PTS", N_LIST)
    # temp_df = rename_pts_against_columns(temp_df)

    temp_df = compute_h2h(temp_df, N_LIST)
    # temp_df = compute_h2h_pts_margin(temp_df, N_LIST)
    temp_df = compute_h2h_season(temp_df)
    temp_df = compute_h2h_streak(temp_df)

    temp_df = compute_elo(temp_df)
    temp_df = compute_elo_season(temp_df)

    temp_df = temp_df.drop(columns=[
        'ELO_PRE', 'OPP_ELO_PRE', 'ELO_PRE_SEASON', 'OPP_ELO_PRE_SEASON',
        'ELO_PRE_x', 'OPP_ELO_PRE_x', 'ELO_PRE_SEASON_x', 'OPP_ELO_PRE_SEASON_x',
        'ELO_PRE_y', 'OPP_ELO_PRE_y', 'ELO_PRE_SEASON_y', 'OPP_ELO_PRE_SEASON_y'
    ], errors='ignore')

    temp_df = compute_elo(temp_df)
    temp_df = compute_elo_season(temp_df)

    return temp_df.tail(2)
