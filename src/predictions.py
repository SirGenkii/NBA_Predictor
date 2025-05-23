import pandas as pd
import joblib
import os
from datetime import datetime
from src.config import *
from src.utils import get_latest_file

def get_last_match_date(df, debug=False):
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        if debug:
            print("Last match date:", df["GAME_DATE"].max())
        return df["GAME_DATE"].max()
    if debug:
        print("GAME_DATE column not found in DataFrame.")
    return None

def get_team_stats_before_date(df, team_id, date):
    return df[(df["TEAM_ID"] == team_id) & (df["GAME_DATE"] < date)].sort_values("GAME_DATE", ascending=False).head(1)

def get_h2h_stats_before_date(df, team_id, opp_id, date):
    return df[(df["GAME_DATE"] < date) & (df["TEAM_ID"] == team_id) & (df["OPP_TEAM_ID"] == opp_id)].sort_values("GAME_DATE", ascending=False).head(1)

def generate_features(home_team_id, away_team_id, df):
    last_date = get_last_match_date(df, debug=True)

    # Récupération des lignes de chaque équipe
    home_raw = get_team_stats_before_date(df, home_team_id, last_date).copy()
    away_raw = get_team_stats_before_date(df, away_team_id, last_date).copy()

    # Définir qui est à domicile
    home_raw["IS_HOME"] = 1
    away_raw["IS_HOME"] = 0

    # Fusionner les features OPP de l'équipe adverse
    for col in home_raw.columns:
        if col.startswith("ELO_PRE") or col.startswith("ROLL") or col.startswith("WIN_STREAK") or col.startswith("H2H"):
            home_raw[f"OPP_{col}"] = away_raw[col].values[0] if col in away_raw.columns else None
            away_raw[f"OPP_{col}"] = home_raw[col].values[0] if col in home_raw.columns else None

    # Récupération des stats H2H spécifiques
    h2h_home = get_h2h_stats_before_date(df, home_team_id, away_team_id, last_date)
    h2h_away = get_h2h_stats_before_date(df, away_team_id, home_team_id, last_date)

    for col in h2h_home.columns:
        if col.startswith("H2H"):
            home_raw[col] = h2h_home[col].values[0] if col in h2h_home.columns else None
    for col in h2h_away.columns:
        if col.startswith("H2H"):
            away_raw[col] = h2h_away[col].values[0] if col in h2h_away.columns else None

    return pd.concat([home_raw, away_raw], ignore_index=True)

def predict_match(home_team_id, away_team_id):
    dataset_file = get_latest_file(DATA_FINAL_CLEANED_DATASET_DIR)
    df = pd.read_csv(dataset_file, dtype={'GAME_ID': str})
    match_df = generate_features(home_team_id, away_team_id, df)

    # Nettoyage : garder uniquement les colonnes utilisées à l'entraînement
    model_input_cols = df.drop(columns=["IS_WIN"], errors='ignore').columns
    match_df = match_df[model_input_cols]

    # Chargement du scaler et du modèle
    # scaler = joblib.load(SCALER_PATH)
    # model = joblib.load(MODEL_PATH)

    # X_scaled = scaler.transform(match_df)
    # probs = model.predict_proba(X_scaled)

    # return {
    #     "home_team_id": home_team_id,
    #     "away_team_id": away_team_id,
    #     "home_win_proba": probs[0][1],
    #     "away_win_proba": probs[1][1]
    # }
    
    return match_df
