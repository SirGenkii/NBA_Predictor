import os
import json
from datetime import datetime
from itertools import product
import multiprocessing
from src.config import *
from src.simulator_betting import run_season_simulation, train_or_load_model
from src.utils import get_latest_file
import pandas as pd

def generate_hyperparam_combinations():
    min_ev_vals = [0.12, 0.15]
    odds_max_vals = [2.8, 3.0, 3.5]
    odds_min_vals = [1.15, 1.20, 1.25]
    prob_diff_vals = [0.07]
    max_risk_vals = [0.02, 0.03]
    
    # min_ev_vals = [0.10, 0.12, 0.15]
    # odds_max_vals = [2.8, 3.0, 3.5]
    # odds_min_vals = [1.15, 1.20, 1.25]
    # prob_diff_vals = [0.025, 0.05, 0.075, 0.10]
    # max_risk_vals = [0.01, 0.02, 0.03, 0.05]

    combinations = list(product(min_ev_vals, odds_max_vals, odds_min_vals, prob_diff_vals, max_risk_vals))
    return combinations

def run_simulation_wrapper(args):
    seasons, min_ev, odds_max, odds_min, prob_diff, max_risk, model_identifier, reset_bankroll_each_season = args
    try:
        bets_df, summary = run_season_simulation(
            seasons_to_test=seasons,
            exclude_future_seasons=True,
            min_ev=min_ev,
            odds_max=odds_max,
            odds_min=odds_min,
            prob_diff=prob_diff,
            bankroll=1000,
            max_risk=max_risk,
            ev_diff_min=0.10,
            streak_limit=10,
            bankroll_stop=0.2,
            skip_first_n=0,
            model_identifier=model_identifier,
            reset_bankroll_each_season=reset_bankroll_each_season,
            parallel = False #set n_jobs to 1 to avoid nest parallelism issues (we already use multiprocessing in this script)
        )

        param_suffix = f"ev{min_ev}_odmax{odds_max}_odmin{odds_min}_pdiff{prob_diff}_risk{max_risk}".replace(".", "-")
        filename = f"bets_grid_{param_suffix}.csv"
        
        os.makedirs(DATA_GRID_SIMULATIONS_BETS_DIR, exist_ok=True)
        filepath = os.path.join(DATA_GRID_SIMULATIONS_BETS_DIR, filename)
        bets_df.to_csv(filepath, index=False)

        summary["param_suffix"] = param_suffix
        return summary

    except Exception as e:
        return {"error": str(e), "params": (min_ev, odds_max, odds_min, prob_diff, max_risk)}

def run_parallel_simulations(seasons, model_identifier="grid", reset_bankroll_each_season=False):
    combinations = generate_hyperparam_combinations()
    
    total_combinations = len(combinations)
    print("="*60)
    print(" DÉMARRAGE DE LA GRILLE DE SIMULATIONS ")
    print("="*60)
    print(f" - Nombre de combinaisons d'hyperparamètres : {total_combinations}")
    print(f" - Saisons testées : {seasons}")
    print(f" - Modèle : {model_identifier}")
    print(f" - Reset bankroll entre saisons : {reset_bankroll_each_season}")
    print(f" - Utilisation de {min(total_combinations, multiprocessing.cpu_count())} processus parallèles")
    print("="*60 + "\n")

    #train model with disabled parallelisation for each season before running simulations 
    #or load if exists 
    
    dataset_path = get_latest_file(DATA_FINAL_CLEANED_DATASET_DIR)
    full_df = pd.read_csv(dataset_path)
    
    for season in seasons:
        print(f"Entraînement ou chargement du modèle pour la saison {season} avec identifier {model_identifier}... \n")
        #train_or_load_model(season, model_identifier=model_identifier, parallel=False)
    
        model = train_or_load_model(
                    full_df,
                    exclude_seasons=[season],
                    model_identifier=model_identifier,
                    season_key=season,
                    parallel=False
                )

    
    args_list = [
        (seasons, min_ev, odds_max, odds_min, prob_diff, max_risk, model_identifier, reset_bankroll_each_season)
        for (min_ev, odds_max, odds_min, prob_diff, max_risk) in combinations
    ]

    with multiprocessing.Pool(processes=min(len(args_list), multiprocessing.cpu_count())) as pool:
        results = pool.map(run_simulation_wrapper, args_list)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(DATA_GRID_SIMULATIONS_BETS_DIR, f"summary_grid_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results
