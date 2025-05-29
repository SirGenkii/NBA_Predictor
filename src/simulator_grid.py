from itertools import product
import os
from src.config import *
from src.simulator_betting import run_season_simulation
from src.utils import *

def generate_hyperparam_combinations():
    min_ev_vals = [0.10, 0.12, 0.15]
    odds_max_vals = [2.8, 3.0, 3.5]
    odds_min_vals = [1.15, 1.20, 1.25]
    prob_diff_vals = [0.04, 0.05, 0.06]

    combinations = list(product(min_ev_vals, odds_max_vals, odds_min_vals, prob_diff_vals))
    return combinations

def run_grid_simulations(seasons, model_identifier="default", reset_bankroll_each_season=False):
    combinations = generate_hyperparam_combinations()
    results = []

    for i, (min_ev, odds_max, odds_min, prob_diff) in enumerate(combinations):
        print(f"\n=== Simulation {i+1}/{len(combinations)} ===")
        print(f"Params: min_ev={min_ev}, odds_max={odds_max}, odds_min={odds_min}, prob_diff={prob_diff}")
        
        try:
            bets_df, summary = run_season_simulation(
                seasons_to_test=seasons,
                exclude_future_seasons=True,
                min_ev=min_ev,
                odds_max=odds_max,
                odds_min=odds_min,
                prob_diff=prob_diff,
                bankroll=1000,
                max_risk=0.02,
                ev_diff_min=0.10,
                streak_limit=10,
                bankroll_stop=0.2,
                skip_first_n=0,
                model_identifier=model_identifier,
                reset_bankroll_each_season=reset_bankroll_each_season
            )

            param_suffix = f"ev{min_ev}_odmax{odds_max}_odmin{odds_min}_pdiff{prob_diff}".replace(".", "")
            filename = f"bets_grid_{param_suffix}.csv"
            filepath = os.path.join(DATA_RESULTS_DIR, filename)
            bets_df.to_csv(filepath, index=False)

            summary["param_suffix"] = param_suffix
            results.append(summary)

        except Exception as e:
            print(f"Erreur pendant la simulation {i+1}: {e}")
            continue

    return results

