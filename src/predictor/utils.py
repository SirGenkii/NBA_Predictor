import pandas as pd
from datetime import datetime
from src.feature_builder import *

def build_prediction_rows(home_team_id: int, away_team_id: int, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les deux lignes de features pour un match donné à partir des données historiques.
    """

    dataset = dataset.copy()
    dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])  # 🔧 Force conversion ici

    # 📅 Date du prochain match (on simule comme si c'était le jour après le dernier match)
    next_game_date = pd.to_datetime(dataset['GAME_DATE'].max()) + pd.Timedelta(days=1)
    next_season = dataset.loc[dataset['GAME_DATE'].idxmax(), 'SEASON']
    
    # 🧱 Créer les deux lignes "raw" du match
    rows = []
    for team_id, is_home, opp_id in [(home_team_id, 1, away_team_id), (away_team_id, 0, home_team_id)]:
        rows.append({
            "TEAM_ID": team_id,
            "OPP_TEAM_ID": opp_id,
            "GAME_DATE": next_game_date,
            "SEASON": next_season,
            "IS_HOME": is_home,
            "IS_WIN": 0  # placeholder temporaire
        })

    prediction_df = pd.DataFrame(rows)

    # 🧪 On concatène au dataset existant pour que les features soient bien calculées dans le contexte historique
    temp_df = pd.concat([dataset, prediction_df], ignore_index=True).sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # 🛠 Recalcul de toutes les features (comme dans notebook 04)
    N_LIST = [3, 5, 10, 25, 50, 100, 200]

    temp_df = compute_rolling_features(temp_df, group_col="TEAM_ID", sort_cols=["TEAM_ID", "GAME_DATE"],
                                       value_cols=['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'PLUS_MINUS'], windows=N_LIST)

    temp_df = compute_winrates(temp_df, "TEAM_ID", "IS_WIN", "IS_HOME", N_LIST)
    temp_df = compute_win_ratio(temp_df, "TEAM_ID", "IS_WIN", N_LIST)
    temp_df["IS_WIN_SHIFTED"] = temp_df.groupby("TEAM_ID")["IS_WIN"].shift(1).fillna(0).astype(int)
    temp_df["WIN_STREAK"] = temp_df.groupby("TEAM_ID").apply(lambda x: compute_win_streak(x, "TEAM_ID", "IS_WIN_SHIFTED")).reset_index(level=0, drop=True)
    temp_df = compute_side_win_streak(temp_df)
    temp_df["DAYS_SINCE_LAST_GAME"] = compute_rest_days(temp_df, "GAME_DATE", "TEAM_ID")
    temp_df["OPP_DAYS_SINCE_LAST_GAME"] = compute_rest_days(temp_df, "GAME_DATE", "OPP_TEAM_ID")
    temp_df["REST_ADVANTAGE"] = temp_df["DAYS_SINCE_LAST_GAME"] - temp_df["OPP_DAYS_SINCE_LAST_GAME"]
    temp_df = compute_rolling_rest_advantage(temp_df, "TEAM_ID", "IS_HOME", "REST_ADVANTAGE", N_LIST)
    temp_df = compute_home_away_pts(temp_df, "TEAM_ID", "IS_HOME", "PTS", "OPP_PTS", N_LIST)
    temp_df = rename_pts_against_columns(temp_df)
    temp_df = compute_h2h(temp_df, N_LIST)
    temp_df = compute_h2h_pts_margin(temp_df, N_LIST)
    temp_df = compute_h2h_season(temp_df)
    temp_df = compute_h2h_streak(temp_df)
    temp_df = compute_elo(temp_df)
    temp_df = compute_elo_season(temp_df)

    # 🎯 Récupérer uniquement les deux dernières lignes ajoutées (match de prédiction)
    result = temp_df.tail(2).copy()

    # 🧹 On retire les colonnes inutiles pour la prédiction
    drop_cols = ['TEAM_ID', 'OPP_TEAM_ID', 'SEASON', 'GAME_DATE', 'IS_WIN']
    prediction_ready = result.drop(columns=drop_cols, errors='ignore')

    return prediction_ready
