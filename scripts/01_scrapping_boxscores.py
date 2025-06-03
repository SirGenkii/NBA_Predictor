import argparse
from datetime import datetime
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import get_latest_file
from src.config import *
from src.nba_scrapping import download_games_for_seasons, scrape_boxscores_v3_for_games




def main(start_year: int, end_year: int, run_timestamp: str = None):
    start_time = datetime.now()
    print("Start time: ", start_time)

    # Timestamp global pour cette session de scraping
    run_timestamp = run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Saisons ciblées
    seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

    # 1. Récupération des matchs
    path_games = download_games_for_seasons(seasons, DATA_GAMES_DIR, run_timestamp)
    games_df = pd.read_csv(path_games, dtype={'GAME_ID': str})

    # 2. Scraping des boxscores V3 saison par saison
    for season in seasons:
        print(f"--- Traitement de la saison {season} ---")
        season_df = games_df[games_df['SEASON'] == season]
        season_output_dir = os.path.join(DATA_BOXSCORES_BATCHES_DIR, season)
        scrape_boxscores_v3_for_games(season_df, season_output_dir, run_timestamp)

    end_time = datetime.now()
    print("End time: ", end_time)
    print("Total time: ", end_time - start_time)
    print("\n✅ Scraping historique V3 terminé")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NBA games and boxscores V3.")
    parser.add_argument('--start_year', type=int, required=True, help='Start season year (e.g., 2010)')
    parser.add_argument('--end_year', type=int, required=True, help='End season year (e.g., 2015)')
    parser.add_argument('--run_timestamp', type=str, default=None, help='Optional existing run timestamp folder')
    args = parser.parse_args()

    main(args.start_year, args.end_year, args.run_timestamp)
