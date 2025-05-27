import pandas as pd
import numpy as np
from datetime import timedelta

def clean_team_name(name):
    return name.strip().lower() if isinstance(name, str) else name

def match_odds_with_dataset(odds_df, nba_df, team_id_map):
    nba_df = nba_df.copy()
    odds_df = odds_df.copy()

    # Conversion explicite des ID en str
    nba_df["TEAM_ID"] = nba_df["TEAM_ID"].astype(str)
    nba_df["OPP_TEAM_ID"] = nba_df["OPP_TEAM_ID"].astype(str)
    team_id_map = {str(k): v for k, v in team_id_map.items()}

    nba_df["TEAM_NAME"] = nba_df["TEAM_ID"].map(team_id_map).apply(clean_team_name)
    nba_df["OPPONENT_NAME"] = nba_df["OPP_TEAM_ID"].map(team_id_map).apply(clean_team_name)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"]).dt.date

    odds_df["home_team"] = odds_df["home_team"].apply(clean_team_name)
    odds_df["away_team"] = odds_df["away_team"].apply(clean_team_name)
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.date

    nba_df["match_key"] = nba_df.apply(
        lambda row: (row["GAME_DATE"], row["TEAM_NAME"], row["OPPONENT_NAME"])
        if row["IS_HOME"] == 1
        else (row["GAME_DATE"], row["OPPONENT_NAME"], row["TEAM_NAME"]),
        axis=1,
    )

    keys_full = []
    for shift in [-1, 0, 1]:
        shifted = odds_df.copy()
        shifted["match_key"] = shifted.apply(
            lambda row: (row["date"] + timedelta(days=shift), row["home_team"], row["away_team"]),
            axis=1
        )
        keys_full.append(shifted)

    odds_full = pd.concat(keys_full, ignore_index=True)
    odds_full = odds_full.drop_duplicates(subset=["match_key"])

    print("\nExemples de clés de match dans odds_df (tolérance date):")
    print(odds_full["match_key"].drop_duplicates().head())
    print("\nExemples de clés de match dans nba_df:")
    print(nba_df["match_key"].drop_duplicates().head())


    merged = pd.merge(odds_full, nba_df, on="match_key", how="left")
    
    # Attribution claire des cotes à chaque ligne équipe
    merged["ODDS"] = merged.apply(
        lambda row: row["home_odds"] if row["IS_HOME"] == 1 else row["away_odds"], axis=1
    )
    merged["OPP_ODDS"] = merged.apply(
        lambda row: row["away_odds"] if row["IS_HOME"] == 1 else row["home_odds"], axis=1
    )
    
    print(f"\nNombre de lignes fusionnées: {len(merged)}")
    print(f"Nombre de correspondances réussies: {merged['TEAM_ID'].notna().sum()}")
    print(f"Nombre de correspondances échouées: {merged['TEAM_ID'].isna().sum()}")

    print("\nIDs manquants TEAM_ID:", nba_df[~nba_df["TEAM_ID"].isin(team_id_map.keys())]["TEAM_ID"].unique())
    print("IDs manquants OPP_TEAM_ID:", nba_df[~nba_df["OPP_TEAM_ID"].isin(team_id_map.keys())]["OPP_TEAM_ID"].unique())

    return merged

def clean_merged_matches(df):
    df = df.copy()
    df = df[df['TEAM_ID'].notna() & df['OPP_TEAM_ID'].notna()]
    df = df.drop_duplicates(subset=["match_key", "TEAM_ID"])
    
    
    
    # Suppression des colonnes originales de odds
    df.drop(columns=["date","match_key","home_team", "away_team", "home_odds", "away_odds", "home_score", "away_score"], inplace=True, errors='ignore')

    
    return df.reset_index(drop=True)

def calculate_ev(prob, odds):
    return prob * (odds - 1) - (1 - prob)

def kelly_criterion(prob, odds):
    b = odds - 1
    q = 1 - prob
    kelly = (b * prob - q) / b if b != 0 else 0
    return max(kelly, 0)

def simulate_bets(merged_df, model_pipeline, min_ev=0.05, bankroll=1000, max_risk=0.05):
    bets = []
    current_bankroll = bankroll

    for idx, row in merged_df.iterrows():
        for team_type in ["home", "away"]:
            team_col = f"{team_type}_team"
            odds_col = f"{team_type}_odds"

            is_home = 1 if team_type == "home" else 0
            pred_row = row.copy()
            pred_row = pred_row.drop(["TEAM_ID", "OPP_TEAM_ID", "SEASON", "GAME_DATE", "IS_WIN"], errors='ignore')
            pred_row = pred_row.dropna()

            if pred_row.empty:
                continue

            pred_input = pred_row[model_pipeline.named_steps['scaler'].get_feature_names_out()].to_frame().T
            prob = model_pipeline.predict_proba(pred_input)[0][1] if is_home == 1 else model_pipeline.predict_proba(pred_input)[0][0]
            ev = calculate_ev(prob, row[odds_col])

            if ev > min_ev:
                f = kelly_criterion(prob, row[odds_col])
                stake = min(f * current_bankroll, current_bankroll * max_risk)
                won = int(row['IS_HOME'] == is_home and row['IS_WIN'] == 1)
                gain = stake * (row[odds_col] - 1) if won else -stake
                current_bankroll += gain

                bets.append({
                    "date": row["date"],
                    "team": row[team_col],
                    "odds": row[odds_col],
                    "prob": prob,
                    "ev": ev,
                    "stake": stake,
                    "won": won,
                    "gain": gain,
                    "bankroll": current_bankroll
                })

    return pd.DataFrame(bets)

def evaluate_simulation(bets_df):
    total_bets = len(bets_df)
    wins = bets_df["won"].sum()
    roi = bets_df["gain"].sum() / bets_df["stake"].sum() if bets_df["stake"].sum() > 0 else 0
    final_bankroll = bets_df["bankroll"].iloc[-1] if not bets_df.empty else None

    return {
        "total_bets": total_bets,
        "wins": wins,
        "roi": roi,
        "final_bankroll": final_bankroll
    }
