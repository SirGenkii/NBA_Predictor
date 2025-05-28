import pandas as pd
from datetime import datetime
from src.feature_builder import *



def build_prediction_rows(home_team_id: int, away_team_id: int, dataset: pd.DataFrame) -> pd.DataFrame:

    N_LIST = [3, 5, 10, 25, 50, 100, 200]

    # Création du jeu de données tronqué pour éviter toute fuite
    
    dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])  # 🔧 Force conversion ici

    
    temp_df = dataset.copy()
    next_game_date = pd.to_datetime(temp_df['GAME_DATE'].max()) + pd.Timedelta(days=1)
    next_season = temp_df.loc[temp_df['GAME_DATE'].idxmax(), 'SEASON']

    base_rows = []
    for team_id, opp_id, is_home in [(home_team_id, away_team_id, 1), (away_team_id, home_team_id, 0)]:
        last_row = temp_df[temp_df['TEAM_ID'] == team_id].sort_values("GAME_DATE").iloc[-1].to_dict()
        row = {
            "TEAM_ID": team_id,
            "OPP_TEAM_ID": opp_id,
            "GAME_DATE": next_game_date,
            "SEASON": next_season,
            "IS_HOME": is_home,
            "IS_WIN": 0,  # Valeur fictive
        }
        base_rows.append(row)

    prediction_df = pd.DataFrame(base_rows)
    temp_df = pd.concat([temp_df, prediction_df], ignore_index=True)

    # Recalcul de toutes les features restantes
    temp_df["IS_WIN_SHIFTED"] = temp_df.groupby("TEAM_ID")["IS_WIN"].shift(1).fillna(0).astype(int)
    temp_df["WIN_STREAK"] = temp_df.groupby("TEAM_ID").apply(lambda x: compute_win_streak(x, "TEAM_ID", "IS_WIN_SHIFTED")).reset_index(level=0, drop=True)
    temp_df = compute_side_win_streak(temp_df)

    temp_df["DAYS_SINCE_LAST_GAME"] = compute_rest_days(temp_df, "GAME_DATE", "TEAM_ID")
    temp_df["OPP_DAYS_SINCE_LAST_GAME"] = compute_rest_days(temp_df, "GAME_DATE", "OPP_TEAM_ID")
    temp_df["REST_ADVANTAGE"] = temp_df["DAYS_SINCE_LAST_GAME"] - temp_df["OPP_DAYS_SINCE_LAST_GAME"]
    temp_df = compute_rolling_rest_advantage(temp_df, "TEAM_ID", "IS_HOME", "REST_ADVANTAGE", N_LIST)

    temp_df = compute_winrates(temp_df, "TEAM_ID", "IS_WIN", "IS_HOME", N_LIST)
    temp_df = compute_win_ratio(temp_df, "TEAM_ID", "IS_WIN", N_LIST)

    temp_df = compute_h2h(temp_df, N_LIST)
    temp_df = compute_h2h_season(temp_df)
    temp_df = compute_h2h_streak(temp_df)


    # 🧹 Supprimer les colonnes ELO avant de les recalculer
    temp_df = temp_df.drop(columns=[
        'ELO_PRE', 'OPP_ELO_PRE', 'ELO_PRE_SEASON', 'OPP_ELO_PRE_SEASON',
        'ELO_PRE_x', 'OPP_ELO_PRE_x', 'ELO_PRE_SEASON_x', 'OPP_ELO_PRE_SEASON_x',
        'ELO_PRE_y', 'OPP_ELO_PRE_y', 'ELO_PRE_SEASON_y', 'OPP_ELO_PRE_SEASON_y'
    ], errors='ignore')


    temp_df = compute_elo(temp_df)
    temp_df = compute_elo_season(temp_df)

    #drop_cols = ['GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'SEASON', 'GAME_DATE','IS_WIN']
    #temp_df = temp_df.drop(columns=drop_cols, errors='ignore')

    return temp_df.tail(2)
