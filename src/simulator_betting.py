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

def simulate_bets(merged_df, model_pipeline, min_ev=0.05, max_ev_for_both=0.15, bankroll=1000, max_risk=0.05):
    bets = []
    current_bankroll = bankroll
    feature_cols = model_pipeline.feature_names_in_

    merged_df = merged_df.sort_values("GAME_DATE")
    #group by GAME_ID
    merged_df = merged_df.groupby("GAME_ID").first().reset_index()
    
    for i in range(0, len(merged_df), 2):
        if i+1 >= len(merged_df):
            break

        rows = merged_df.iloc[i:i+2].copy().reset_index(drop=True)
        if rows.shape[0] != 2:
            continue

        display_info = rows[["TEAM_NAME", "OPPONENT_NAME", "ODDS", "IS_HOME", "IS_WIN"]]
        try:
            pred_rows = rows.drop(columns=["ODDS", "OPP_ODDS", "TEAM_NAME", "OPPONENT_NAME", "date", "match_key"], errors='ignore')
            pred_input = pred_rows[feature_cols]
            probs = model_pipeline.predict_proba(pred_input)

            team_0_prob = probs[0][1] if rows.loc[0, "IS_HOME"] == 1 else probs[0][0]
            team_1_prob = probs[1][1] if rows.loc[1, "IS_HOME"] == 1 else probs[1][0]
            ev_0 = calculate_ev(team_0_prob, rows.loc[0, "ODDS"])
            ev_1 = calculate_ev(team_1_prob, rows.loc[1, "ODDS"])

            print(f"\nMatch {i//2 + 1}: {display_info.loc[0, 'TEAM_NAME']} vs {display_info.loc[0, 'OPPONENT_NAME']} ({rows.loc[0, 'GAME_DATE']})")
            print(f"  {display_info.loc[0, 'TEAM_NAME']} - Prob: {team_0_prob:.2f}, EV: {ev_0:.2f}, Odds: {rows.loc[0, 'ODDS']}")
            print(f"  {display_info.loc[1, 'TEAM_NAME']} - Prob: {team_1_prob:.2f}, EV: {ev_1:.2f}, Odds: {rows.loc[1, 'ODDS']}")

            if ev_0 > min_ev and ev_1 > min_ev:
                if max(ev_0, ev_1) < max_ev_for_both:
                    print("  Match trop serré, pas de pari.")
                    continue

            best_idx = 0 if ev_0 > ev_1 else 1
            row = rows.loc[best_idx]
            prob = team_0_prob if best_idx == 0 else team_1_prob
            ev = ev_0 if best_idx == 0 else ev_1

            f = kelly_criterion(prob, row["ODDS"])
            stake = min(f * current_bankroll, current_bankroll * max_risk)
            won = int(row["IS_WIN"] == 1)
            gain = stake * (row["ODDS"] - 1) if won else -stake
            current_bankroll += gain

            print(f"  Pari sur {row['TEAM_NAME']} ! Mise: {stake:.2f}, {'GAGNÉ' if won else 'PERDU'}, Bankroll: {current_bankroll:.2f}")

            bets.append({
                "date": row.get("date", None),
                "team": row["TEAM_NAME"],
                "odds": row["ODDS"],
                "prob": prob,
                "ev": ev,
                "stake": stake,
                "won": won,
                "gain": gain,
                "bankroll": current_bankroll
            })

        except Exception as e:
            print(f"  Erreur sur match {i//2 + 1} : {e}")
            continue

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
