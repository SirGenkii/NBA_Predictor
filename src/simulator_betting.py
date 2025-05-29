import pandas as pd
import numpy as np
from datetime import timedelta
import json
from datetime import datetime

import joblib
import os
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


from src.feature_builder import *
from src.config import *
from src.utils import get_latest_file, json_serial

from pathlib import Path
from typing import List, Tuple


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

    # print("\nExemples de clés de match dans odds_df (tolérance date):")
    # print(odds_full["match_key"].drop_duplicates().head())
    # print("\nExemples de clés de match dans nba_df:")
    # print(nba_df["match_key"].drop_duplicates().head())


    merged = pd.merge(odds_full, nba_df, on="match_key", how="left")
    
    # Attribution claire des cotes à chaque ligne équipe
    merged["ODDS"] = merged.apply(
        lambda row: row["home_odds"] if row["IS_HOME"] == 1 else row["away_odds"], axis=1
    )
    merged["OPP_ODDS"] = merged.apply(
        lambda row: row["away_odds"] if row["IS_HOME"] == 1 else row["home_odds"], axis=1
    )
    
    # print(f"\nNombre de lignes fusionnées: {len(merged)}")
    # print(f"Nombre de correspondances réussies: {merged['TEAM_ID'].notna().sum()}")
    # print(f"Nombre de correspondances échouées: {merged['TEAM_ID'].isna().sum()}")

    # print("\nIDs manquants TEAM_ID:", nba_df[~nba_df["TEAM_ID"].isin(team_id_map.keys())]["TEAM_ID"].unique())
    # print("IDs manquants OPP_TEAM_ID:", nba_df[~nba_df["OPP_TEAM_ID"].isin(team_id_map.keys())]["OPP_TEAM_ID"].unique())

    return merged

def clean_merged_matches(df):
    df = df.copy()
    df = df[df['TEAM_ID'].notna() & df['OPP_TEAM_ID'].notna()]
    df = df.drop_duplicates(subset=["match_key", "TEAM_ID"])
    
    # Suppression des colonnes originales de odds
    df.drop(columns=["date","match_key","home_team", "away_team", "home_odds", "away_odds", "home_score", "away_score"], inplace=True, errors='ignore')
    
    #df = df.dropna(subset=["home_odds", "away_odds"])
    
    #drop nan for "home_odds", "away_odds" and print the number of rows removed
    initial_rows = len(df)
    df = df.dropna(subset=["ODDS", "OPP_ODDS"])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"------------------ Nombre de lignes supprimées pour cotes manquantes: {removed_rows} ------------------")
    
    return df.reset_index(drop=True)

def calculate_ev(prob, odds):
    return prob * (odds - 1) - (1 - prob)

def kelly_criterion(prob, odds):
    b = odds - 1
    q = 1 - prob
    kelly = (b * prob - q) / b if b != 0 else 0
    return max(kelly, 0)


def simulate_bets(merged_df, model_pipeline, min_ev=0.05, bankroll=1000, max_risk=0.05, skip_first_n=0, log_file_path="betting_simulation_log.json"):
    bets = []
    current_bankroll = bankroll
    feature_cols = model_pipeline.feature_names_in_

    merged_df = merged_df.sort_values("GAME_DATE")
    game_groups = merged_df.groupby("GAME_ID")
    valid_games = [g for _, g in game_groups if len(g) == 2]
    valid_games = valid_games[skip_first_n:]

    for i, game in enumerate(valid_games, 1):
        rows = game.sort_values("IS_HOME", ascending=False).reset_index(drop=True)
        display_info = rows[["TEAM_NAME", "OPPONENT_NAME", "ODDS", "IS_HOME", "IS_WIN"]]
        try:
            pred_rows = rows.drop(columns=["ODDS", "OPP_ODDS", "TEAM_NAME", "OPPONENT_NAME", "date", "match_key"], errors='ignore')
            pred_input = pred_rows[feature_cols].dropna(axis=1, how='any')

            if pred_input.shape[1] != len(feature_cols):
                print(f"  Match {i} ignoré : features incomplètes.")
                continue

            probs = model_pipeline.predict_proba(pred_input)

            team_0_prob = probs[0][1]
            team_1_prob = probs[1][1]
            ev_0 = calculate_ev(team_0_prob, rows.loc[0, "ODDS"])
            ev_1 = calculate_ev(team_1_prob, rows.loc[1, "ODDS"])

            print(f"\nMatch {i}: {display_info.loc[0, 'TEAM_NAME']} vs {display_info.loc[0, 'OPPONENT_NAME']} ({rows.loc[0, 'GAME_DATE']})")
            print(f"  {display_info.loc[0, 'TEAM_NAME']} - Prob: {team_0_prob:.2f}, EV: {ev_0:.2f}, Odds: {rows.loc[0, 'ODDS']}")
            print(f"  {display_info.loc[1, 'TEAM_NAME']} - Prob: {team_1_prob:.2f}, EV: {ev_1:.2f}, Odds: {rows.loc[1, 'ODDS']}")

            prob_diff = abs(team_0_prob - team_1_prob)
            if prob_diff < 0.05:
                print("Match trop serré, pas de pari.")

            if max(ev_0, ev_1) < min_ev:
                print(f"Pas d'EV interessante au dessus de {min_ev}.")
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
                "date": row["GAME_DATE"],
                "game_id": row["GAME_ID"],
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
            print(f"  Erreur sur match {i} : {e}")
            continue

    # Logging
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "min_ev": float(min_ev),
            "bankroll": float(bankroll),
            "max_risk": float(max_risk),
            "skip_first_n": int(skip_first_n)
        },
        "results": evaluate_simulation(pd.DataFrame(bets))
    }
    with open(log_file_path, 'a') as f:
        f.write(json.dumps(log_data, default=json_serial) + "\n")

    return pd.DataFrame(bets)

def simulate_bets_optimized(merged_df, model_pipeline, 
                            min_ev=0.15, 
                            bankroll=1000,
                            max_risk=0.02,
                            prob_diff=0.10,
                            odds_max=2.5, 
                            odds_min=1.2, 
                            ev_diff_min=0.10,
                            streak_limit=5,
                            bankroll_stop=0.5, 
                            skip_first_n=0, 
                            log_file_path="betting_simulation_log_optimized.json"):
    bets = []
    current_bankroll = bankroll
    initial_bankroll = bankroll
    feature_cols = model_pipeline.feature_names_in_
    consecutive_losses = 0

    merged_df = merged_df.sort_values("GAME_DATE")
    game_groups = merged_df.groupby("GAME_ID")
    valid_games = [g for _, g in game_groups if len(g) == 2]
    valid_games = valid_games[skip_first_n:]

    for i, game in enumerate(valid_games, 1):
        if current_bankroll < initial_bankroll * bankroll_stop:
            print(f"Arrêt de la simulation: bankroll trop faible ({current_bankroll:.2f})")
            break
        if consecutive_losses >= streak_limit:
            print(f"Pause après {consecutive_losses} pertes consécutives au match {i}")
            consecutive_losses = 0
            continue

        rows = game.sort_values("IS_HOME", ascending=False).reset_index(drop=True)
        try:
            pred_rows = rows.drop(columns=["ODDS", "OPP_ODDS", "TEAM_NAME", "OPPONENT_NAME", "date", "match_key"], errors='ignore')
            pred_input = pred_rows[feature_cols].dropna(axis=1, how='any')

            if pred_input.shape[1] != len(feature_cols):
                print(f"  Match {i} ignoré : features incomplètes.")
                continue

            probs = model_pipeline.predict_proba(pred_input)
            team_0_prob = probs[0][1]
            team_1_prob = probs[1][1]
            ev_0 = (team_0_prob * rows.loc[0, "ODDS"]) - 1
            ev_1 = (team_1_prob * rows.loc[1, "ODDS"]) - 1
            odds_0 = rows.loc[0, "ODDS"]
            odds_1 = rows.loc[1, "ODDS"]

            best_idx = 0 if ev_0 > ev_1 else 1
            row = rows.loc[best_idx]
            odds = row["ODDS"]
            prob = team_0_prob if best_idx == 0 else team_1_prob
            ev = ev_0 if best_idx == 0 else ev_1
            won = int(row["IS_WIN"])
            comment = ""

            if abs(team_0_prob - team_1_prob) < prob_diff:
                comment = f"prob diff too low ({abs(team_0_prob - team_1_prob):.3f})"
            elif odds > odds_max or odds < odds_min:
                comment = "odds too high" if odds > odds_max else "odds too low"
            elif ev < min_ev:
                comment = f"ev below min ({ev:.3f})"
            elif abs(ev_0 - ev_1) < ev_diff_min:
                comment = f"ev diff too low ({abs(ev_0 - ev_1):.3f})"

            if comment:
                print(f"  Match {i} ignoré : {comment}")
                bets.append({
                    "date": row["GAME_DATE"], "game_id": row["GAME_ID"],
                    "team": row["TEAM_NAME"], "odds": odds,
                    "prob": prob, "ev": ev, "stake": 0,
                    "won": won, "gain": 0, "bankroll": current_bankroll,
                    "comment": comment
                })
                continue

            kelly_fraction = ev / (odds - 1)
            stake = min(kelly_fraction * current_bankroll, current_bankroll * max_risk)
            stake = max(stake, 0)
            gain = stake * (odds - 1) if won else -stake
            current_bankroll += gain
            consecutive_losses = 0 if won else consecutive_losses + 1

            print(f"  Match {i} placed : Pari sur {row['TEAM_NAME']} ! Mise: {stake:.2f}, {'GAGNÉ' if won else 'PERDU'}, Bankroll: {current_bankroll:.2f}")
            bets.append({
                "date": row["GAME_DATE"], "game_id": row["GAME_ID"],
                "team": row["TEAM_NAME"], "odds": odds,
                "prob": prob, "ev": ev, "stake": stake,
                "won": won, "gain": gain, "bankroll": current_bankroll,
                "comment": "bet placed"
            })

        except Exception as e:
            print(f"!!! Erreur lors du traitement du match {i}: {e} !!!")
            bets.append({
                "game_id": rows.iloc[0]["GAME_ID"], "stake": 0, "gain": 0,
                "bankroll": current_bankroll, "comment": f"error: {str(e)}"
            })

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "min_ev": float(min_ev), "bankroll": float(bankroll), "max_risk": float(max_risk),
            "prob_diff": float(prob_diff), "odds_max": float(odds_max), "odds_min": float(odds_min),
            "ev_diff_min": float(ev_diff_min), "streak_limit": int(streak_limit),
            "bankroll_stop": float(bankroll_stop), "skip_first_n": int(skip_first_n)
        },
        "results": {
            "total_bets": len(bets),
            "wins": sum(b.get("won", 0) for b in bets if b.get("stake", 0) > 0),
            "roi": sum(b["gain"] for b in bets) / sum(b["stake"] for b in bets if b["stake"] > 0) if any(b["stake"] > 0 for b in bets) else 0,
            "final_bankroll": current_bankroll
        }
    }

    with open(log_file_path, 'a') as f:
        f.write(json.dumps(log_data, default=str) + "\n")

    return pd.DataFrame(bets)

# Logging avec nouveaux paramètres
def json_serial(obj):
    if isinstance(obj, (datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")
    

def evaluate_simulation(bets_df):
    total_analysed = len(bets_df)
    total_bet_placed = bets_df[bets_df["stake"] > 0]
    wins = bets_df["won"].sum()
    roi = bets_df["gain"].sum() / bets_df["stake"].sum() if bets_df["stake"].sum() > 0 else 0
    final_bankroll = bets_df["bankroll"].iloc[-1] if not bets_df.empty else None

    return {
        "total_analysed": total_analysed,
        "total_bet_placed": len(total_bet_placed),
        "wins": wins,
        "roi": roi,
        "final_bankroll": final_bankroll
    }


def set_model_n_jobs(model, n_jobs):
    try:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            stack = model.named_steps["model"]

            # Fix sur les estimateurs de base
            for name, estimator in stack.estimators:
                if hasattr(estimator, "n_jobs"):
                    estimator.n_jobs = n_jobs

            # Fix sur le final estimator
            if hasattr(stack.final_estimator, "n_jobs"):
                stack.final_estimator.n_jobs = n_jobs

            # Fix sur le StackingClassifier lui-même
            if hasattr(stack, "n_jobs"):
                stack.n_jobs = n_jobs

    except Exception as e:
        print(f"[WARN] set_model_n_jobs failed: {e}")


def train_or_load_model(df, exclude_seasons, model_identifier, season_key, parallel=True):
    os.makedirs(DATA_MODELS_SIMULATIONS_DIR, exist_ok=True)
    model_path = os.path.join(DATA_MODELS_SIMULATIONS_DIR, f"model_{season_key}_{model_identifier}.joblib")

    if parallel:
        n_jobs = -1  # Utiliser tous les cœurs disponibles
    else:
        n_jobs = 1
        
    print(f"Model in parallel mode: {parallel}. n_jobs set to {n_jobs}" )

    if Path(model_path).exists():
        print(f"Chargement du modèle existant pour {season_key} ({model_identifier})")
        model = joblib.load(model_path)
        print(f"Modèle chargé depuis {model_path}")
  
        #set_model_n_jobs(model, n_jobs)
        
        return model

    target = 'IS_WIN'
    drop_cols = ['GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'SEASON', 'GAME_DATE'] + COLS_MATCH_REAL
    features = [col for col in df.columns if col not in drop_cols + [target]]
    df = df.dropna(subset=features)

    filtered_df = df[~df['SEASON'].isin(exclude_seasons)].copy()
    X = filtered_df[features].select_dtypes(include=['number'])
    y = filtered_df[target]

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=n_jobs)),
        ('lgbm', LGBMClassifier(n_estimators=200, random_state=42, n_jobs=n_jobs, verbose=-1)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                              subsample=0.7, colsample_bytree=0.7,
                              random_state=42, eval_metric='logloss',
                              n_jobs=n_jobs, use_label_encoder=False, verbosity=0)),
        ('cat', CatBoostClassifier(n_estimators=200, learning_rate=0.05,
                                   depth=6, verbose=0, random_state=42, thread_count=n_jobs)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, random_state=42))
    ]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', StackingClassifier(
            estimators=estimators,
            cv=5,
            final_estimator=LogisticRegression(),
            passthrough=True,
            n_jobs=n_jobs
        ))
    ])
    model.fit(X, y)

    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"ROC AUC sur données d'entraînement: {auc:.4f}")

    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    return model
# Affichage du code modifié de run_season_simulation incluant le paramètre reset_bankroll_each_season


def run_season_simulation(seasons_to_test: List[str],
                           exclude_future_seasons=True,
                           min_ev=0.1,
                           odds_max=2.8,
                           odds_min=1.15,
                           prob_diff=0.05,
                           ev_diff_min=0.1,
                           bankroll=1000,
                           max_risk=0.02,
                           streak_limit=5,
                           bankroll_stop=0.5,
                           skip_first_n=0,
                           model_identifier="default",
                           reset_bankroll_each_season=False,
                           parallel=True
                           ) -> Tuple[pd.DataFrame, dict]:

    bets_all = []
    current_bankroll = bankroll

    dataset_path = get_latest_file(DATA_FINAL_CLEANED_DATASET_DIR)
    full_df = pd.read_csv(dataset_path)
    team_mapping_file = get_latest_file(DATA_TEAMS_DIR)
    team_mapping = pd.read_csv(team_mapping_file)
    team_id_map = dict(zip(team_mapping["id"].astype(str), team_mapping["full_name"]))

    for season in seasons_to_test:
        print(f"\n--- Saison: {season} ---")
        season_df = full_df[full_df["SEASON"] == season].copy()
        # if exclude_future_seasons:
        #     training_df = full_df[full_df["SEASON"] < season].copy()
        # else:
        #     training_df = full_df[~full_df["SEASON"].isin(seasons_to_test)].copy()


        model = train_or_load_model(
                full_df,
                exclude_seasons=[season],
                model_identifier=model_identifier,
                season_key=season,
                parallel=parallel
            )

        
        odds_path = os.path.join(DATA_ODDS_HISTORY_DIR, f"nba_{season.replace('-', '_')}.csv")
        odds_df = pd.read_csv(odds_path)
        merged = match_odds_with_dataset(odds_df, season_df, team_id_map)
        cleaned = clean_merged_matches(merged)

        bets_df = simulate_bets_optimized(
            merged_df=cleaned,
            model_pipeline=model,
            min_ev=min_ev,
            odds_max=odds_max,
            odds_min=odds_min,
            prob_diff=prob_diff,
            ev_diff_min=ev_diff_min,
            bankroll=current_bankroll,
            max_risk=max_risk,
            streak_limit=streak_limit,
            bankroll_stop=bankroll_stop,
            skip_first_n=skip_first_n
        )

        if not reset_bankroll_each_season and not bets_df.empty:
            current_bankroll = bets_df["bankroll"].iloc[-1]

        bets_all.append(bets_df)

    full_bets_df = pd.concat(bets_all, ignore_index=True)
    summary = evaluate_simulation(full_bets_df)

    return full_bets_df, summary
