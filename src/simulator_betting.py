import pandas as pd
import numpy as np
from datetime import timedelta
import json
from datetime import datetime

import joblib
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from src.feature_builder import *
from src.config import *
from src.utils import get_latest_file, json_serial

from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt


import warnings
from sklearn.exceptions import DataConversionWarning

# Désactive uniquement ce warning lié aux noms de features manquants
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")



def calculate_ev(prob, odds):
    return prob * (odds - 1) - (1 - prob)

def kelly_criterion(prob, odds):
    b = odds - 1
    q = 1 - prob
    kelly = (b * prob - q) / b if b != 0 else 0
    return max(kelly, 0)

def simulate_bets_optimized(
    merged_df, model_pipeline, 
    min_ev=0.15, bankroll=1000, max_risk=0.02, 
    prob_diff=0.10, odds_max=2.5, odds_min=1.2, 
    ev_diff_min=0.10, streak_limit=5, bankroll_stop=0.5, 
    skip_first_n=0, log_file_path="betting_simulation_log_optimized.json",
    stake_method="kelly", prob_diff_vs_book_range=None
):
    bets = []
    current_bankroll = bankroll
    initial_bankroll = bankroll
    consecutive_losses = 0
    feature_cols = model_pipeline.feature_names_in_

    merged_df = merged_df.sort_values("GAME_DATE")
    valid_games = [
        g for _, g in merged_df.groupby("GAME_ID") 
        if len(g) == 2
    ][skip_first_n:]

    for i, game in enumerate(valid_games, 1):
        if current_bankroll < initial_bankroll * bankroll_stop:
            print(f"Arrêt simulation : bankroll trop faible ({current_bankroll:.2f})")
            break
        if consecutive_losses >= streak_limit:
            print(f"Pause après {consecutive_losses} pertes consécutives (match {i})")
            consecutive_losses = 0
            continue

        try:
            game = game.sort_values("IS_HOME", ascending=False).reset_index(drop=True)
            display_info = game[["TEAM_NAME", "OPPONENT_NAME", "ODDS", "IS_HOME", "IS_WIN", "GAME_DATE"]]

            pred_rows = game.drop(columns=["TEAM_NAME", "OPPONENT_NAME", "date", "match_key"], errors="ignore")
            pred_input = pred_rows.loc[:, feature_cols]

            if pred_input.shape[1] != len(feature_cols):
                print(f"Match {i} ignoré : features incomplètes.")
                continue

            probs = model_pipeline.predict_proba(pred_input)
            team_probs = [probs[0][1], probs[1][1]]
            evs = [(p * game.loc[j, "ODDS"]) - 1 for j, p in enumerate(team_probs)]

            best_idx = int(evs[1] > evs[0])
            row = game.loc[best_idx]
            opp_row = game.loc[1 - best_idx]

            prob = team_probs[best_idx]
            ev = evs[best_idx]
            odds = row["ODDS"]
            won = int(row["IS_WIN"])
            comment = ""

            prob_book = 1 / odds
            opp_prob_book = 1 / opp_row["ODDS"]
            prob_book_norm = prob_book / (prob_book + opp_prob_book)
            prob_diff_vs_book = prob - prob_book_norm

            if prob_diff_vs_book_range is not None:
                min_diff, max_diff = prob_diff_vs_book_range
                if not (min_diff <= prob_diff_vs_book <= max_diff):
                    comment = f"prob_diff_vs_book hors range ({prob_diff_vs_book:.3f})"

            if not comment:
                if abs(team_probs[0] - team_probs[1]) < prob_diff:
                    comment = f"diff proba trop faible ({abs(team_probs[0] - team_probs[1]):.3f})"
                elif odds > odds_max or odds < odds_min:
                    comment = f"odds hors limites ({odds:.3f})"
                elif ev < min_ev:
                    comment = f"ev trop bas ({ev:.3f})"
                elif abs(evs[0] - evs[1]) < ev_diff_min:
                    comment = f"diff ev trop faible ({abs(evs[0] - evs[1]):.3f})"

            if comment:
                print(f"Match {i} ignoré : {comment}")
                bets.append({
                    "date": row["GAME_DATE"], "game_id": row["GAME_ID"],
                    "team": row["TEAM_NAME"], "odds": odds, "prob": prob,
                    "ev": ev, "stake": 0, "won": won, "gain": 0,
                    "bankroll": current_bankroll, "comment": comment,
                    "prob_book": prob_book, "opp_prob_book": opp_prob_book,
                    "prob_book_norm": prob_book_norm, "prob_diff_vs_book": prob_diff_vs_book,"season": row["SEASON"]
                })
                continue

            if stake_method == "kelly":
                f = kelly_criterion(prob, row["ODDS"])
                stake = min(f * current_bankroll, current_bankroll * max_risk)
            elif stake_method == "fixed":
                stake = current_bankroll * max_risk

            gain = stake * (odds - 1) if won else -stake
            current_bankroll += gain
            consecutive_losses = 0 if won else consecutive_losses + 1

            print(f"Match {i} placé : {row['TEAM_NAME']} - Mise: {stake:.2f} - {'GAGNÉ' if won else 'PERDU'} - Bankroll: {current_bankroll:.2f}")
            bets.append({
                "date": row["GAME_DATE"], "game_id": row["GAME_ID"],
                "team": row["TEAM_NAME"], "odds": odds, "prob": prob,
                "ev": ev, "stake": stake, "won": won, "gain": gain,
                "bankroll": current_bankroll, "comment": "bet placed",
                "prob_book": prob_book, "opp_prob_book": opp_prob_book,
                "prob_book_norm": prob_book_norm, "prob_diff_vs_book": prob_diff_vs_book, "season": row["SEASON"]
            })

        except Exception as e:
            print(f"!!! Erreur sur match {i} : {e} !!!")
            bets.append({
                "game_id": game.iloc[0]["GAME_ID"], "stake": 0, "gain": 0,
                "bankroll": current_bankroll, "comment": f"error: {str(e)}"
            })

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "min_ev": min_ev, "bankroll": bankroll, "max_risk": max_risk,
            "prob_diff": prob_diff, "odds_max": odds_max, "odds_min": odds_min,
            "ev_diff_min": ev_diff_min, "streak_limit": streak_limit,
            "bankroll_stop": bankroll_stop, "skip_first_n": skip_first_n
        },
        "results": evaluate_simulation(pd.DataFrame(bets)),
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
    wins = total_bet_placed["won"].sum()
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
        ('lgbm', LGBMClassifier(n_estimators=150, num_leaves=64, random_state=42, n_jobs=n_jobs)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.7, colsample_bytree=0.7, random_state=42, eval_metric='logloss', n_jobs=n_jobs, use_label_encoder=False)),
        ('cat', CatBoostClassifier(n_estimators=200, learning_rate=0.05, depth=6, rsm=0.8, verbose=0, random_state=42, thread_count=n_jobs)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, random_state=42)),
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', n_jobs=n_jobs))),
        ('mlp', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))),
        ('et', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=n_jobs)),
        ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=n_jobs)))
    ]
    
    # Meta-model (peut être LogisticRegression, simple et efficace)
    meta_model = LogisticRegression(solver='lbfgs', max_iter=5000)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', StackingClassifier(
            estimators=estimators,
            cv=5,
            final_estimator=meta_model,
            passthrough=False,
            n_jobs=n_jobs
        ))
    ])
    model.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC sur validation: {auc:.4f}")

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
                            stake_method="kelly",
                            streak_limit=5,
                            bankroll_stop=0.5,
                            skip_first_n=0,
                            model_identifier="default",
                            reset_bankroll_each_season=False,
                            parallel=True,
                            prob_diff_vs_book_range=None
                           ) -> Tuple[pd.DataFrame, dict]:

    bets_all = []
    current_bankroll = bankroll

    dataset_path = get_latest_file(DATA_FINAL_DATASET_DIR)
    full_df = pd.read_csv(dataset_path)
    
    # team_mapping_file = get_latest_file(DATA_TEAMS_DIR)
    # team_mapping = pd.read_csv(team_mapping_file)
    # team_id_map = dict(zip(team_mapping["id"].astype(str), team_mapping["full_name"]))

    for season in seasons_to_test:
        print(f"\n--- Saison: {season} ---")
        season_df = full_df[full_df["SEASON"] == season].copy()
        if exclude_future_seasons:
            training_df = full_df[full_df["SEASON"] < season].copy()
        else:
            training_df = full_df[~full_df["SEASON"].isin(seasons_to_test)].copy()


        model = train_or_load_model(
                training_df,
                exclude_seasons=[season],
                model_identifier=model_identifier,
                season_key=season,
                parallel=parallel
            )

        
        # odds_path = os.path.join(DATA_ODDS_HISTORY_DIR, f"nba_{season.replace('-', '_')}.csv")
        # odds_df = pd.read_csv(odds_path)
        # merged = match_odds_with_dataset(odds_df, season_dfdf 
        print("dataset cols before SIM :", season_df.columns.tolist())
               

        bets_df = simulate_bets_optimized(
            merged_df=season_df,
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
            skip_first_n=skip_first_n,
            stake_method=stake_method,
            prob_diff_vs_book_range=prob_diff_vs_book_range
        )
        
        

        if not reset_bankroll_each_season and not bets_df.empty:
            current_bankroll = bets_df["bankroll"].iloc[-1]

        bets_all.append(bets_df)

    full_bets_df = pd.concat(bets_all, ignore_index=True)
    summary = evaluate_simulation(full_bets_df)

    return full_bets_df, summary



def analyze_bets_diagnostics(bets_df: pd.DataFrame, bins=None):
    if bins is None:
        #default bins for prob_diff_vs_book eache 0.0125 from -1 to 1
        bins = np.arange(-1, 1, 0.0125).tolist()
        

    df = bets_df[bets_df["stake"] > 0].copy()  # Ne garder que les paris placés

    df["diff_bin"] = pd.cut(df["prob_diff_vs_book"], bins=bins)

    diagnostics = df.groupby("diff_bin").agg(
        count=("stake", "count"),
        win_rate=("won", "mean"),
        avg_gain=("gain", "mean"),
        total_gain=("gain", "sum"),
        avg_prob_diff_vs_book=("prob_diff_vs_book", "mean")
    ).reset_index()

    return diagnostics


def analyze_performance_by_odds(bets_df: pd.DataFrame, odds_bins=None):
    if odds_bins is None:
        # Par défaut, tranches de 0.1 entre 1.0 et 3.0
        odds_bins = np.arange(1.0, 3.05, 0.1)

    df = bets_df[bets_df["stake"] > 0].copy()  # uniquement les paris réellement placés
    df["odds_bin"] = pd.cut(df["odds"], bins=odds_bins)

    analysis = df.groupby("odds_bin").agg(
        count=("gain", "count"),
        win_rate=("won", "mean"),
        avg_gain=("gain", "mean"),
        total_gain=("gain", "sum"),
        avg_stake=("stake", "mean")
    ).reset_index()

    return analysis

def analyse_performance_by_model(bets_df: pd.DataFrame, bins=None):
    if bins is None:
        # Par défaut, tranches de 0.1 entre 0 et 1
        bins = np.arange(0, 1.05, 0.025)

    df = bets_df[bets_df["stake"] > 0].copy()  # uniquement les paris réellement placés
    df["prob_bin"] = pd.cut(df["prob"], bins=bins)

    analysis = df.groupby("prob_bin").agg(
        count=("gain", "count"),
        win_rate=("won", "mean"),
        avg_gain=("gain", "mean"),
        total_gain=("gain", "sum"),
        avg_odds=("odds", "mean")
    ).reset_index()

    return analysis
    


def plot_bankroll(bets_df: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    plt.plot(bets_df.index, bets_df["bankroll"])
    plt.title("Bankroll Over Time")
    plt.xlabel("Date")
    plt.ylabel("Bankroll")

    # Marqueurs de changement de saison
    last_season = None
    for i, season in enumerate(bets_df["season"]):
        if season != last_season:
            plt.axvline(x=i, color='red', linestyle='--', alpha=0.5)
            plt.text(i, bets_df["bankroll"].max(), season, rotation=90, verticalalignment='bottom', color='red')
            last_season = season

    plt.tight_layout()
    plt.show()

