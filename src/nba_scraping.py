import os
import pandas as pd
import time
import random
import glob
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3, boxscorefourfactorsv3, boxscoremiscv3, boxscorescoringv3, boxscoreusagev3
from src.config import *
from src.utils import save_dataframe_to_csv, get_latest_file

# -- 1. Download all games for a season (or several)
def download_games_for_seasons(seasons, output_dir, run_timestamp):
    all_games = []
    for season in seasons:
        print(f"Extraction saison {season}")
        gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00', season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        games['SEASON'] = season
        all_games.append(games)
        time.sleep(2)
    df_games = pd.concat(all_games, ignore_index=True)
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
    df_games = df_games[df_games['GAME_ID'].astype(str).str.startswith(('002','004', '005'))]
    df_games = df_games.sort_values(by='GAME_DATE')
    file_path = save_dataframe_to_csv(df_games, os.path.join(output_dir), prefix=f'nba_games', suffix=run_timestamp)
    return file_path

# -- 2. Trouver les nouveaux matchs à scraper
def get_new_games(hist_games_path, new_games_path):
    hist_games = pd.read_csv(hist_games_path, dtype={'GAME_ID': str})
    new_games = pd.read_csv(new_games_path, dtype={'GAME_ID': str})
    old_ids = set(hist_games['GAME_ID'])
    to_add = new_games[~new_games['GAME_ID'].isin(old_ids)]
    print(f"{len(to_add)} nouveaux matchs à traiter")
    return to_add

# -- 3. Scraper les boxscores enrichis V3 des nouveaux GAME_ID
def scrape_boxscores_v3_for_games(games_df, output_dir, run_timestamp, batch_size=25):
    all_data = {
        'traditional': [], 'advanced': [], 'fourfactors': [],
        'misc': [], 'scoring': [], 'usage': []
    }
    error_log = []
    game_ids = games_df['GAME_ID'].tolist()
    os.makedirs(os.path.join(output_dir, run_timestamp), exist_ok=True)
    batch_num = 0

    for idx, gid in enumerate(game_ids):
        print(f"[{idx+1}/{len(game_ids)}] GAME_ID: {gid} - {games_df.loc[games_df['GAME_ID'] == gid, 'GAME_DATE'].values[0]}")
        try:
            all_data['traditional'].append(boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['advanced'].append(boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['fourfactors'].append(boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['misc'].append(boxscoremiscv3.BoxScoreMiscV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['scoring'].append(boxscorescoringv3.BoxScoreScoringV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['usage'].append(boxscoreusagev3.BoxScoreUsageV3(game_id=gid, timeout=30).player_stats.get_data_frame())
        except Exception as e:
            print(f"Error for GAME_ID {gid}: {e}")
            error_log.append((gid, str(e)))
            time.sleep(30)
            continue

        time.sleep(random.uniform(3.5, 5.5))

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(game_ids):
            batch_num += 1
            for key, df_list in all_data.items():
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    filename = f"boxscores_{key}_v3_batch_{batch_num}"
                    save_dataframe_to_csv(df, os.path.join(output_dir, run_timestamp), prefix=filename)
                    print(f"✅ {key} V3 batch {batch_num} saved ({len(df)} rows)")
                    all_data[key] = []
            if error_log:
                with open(os.path.join(ERROR_LOG_FOLDER, f"errors_batch_{batch_num}_{run_timestamp}.txt"), "a") as f:
                    for err in error_log:
                        f.write(f"{err[0]}\t{err[1]}\n")
                error_log = []
    return True

# -- 4. Merge boxscores historiques et nouveaux
def load_all_csvs(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    dfs = [pd.read_csv(f, dtype={'GAME_ID': str}) for f in files]
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def merge_boxscores_batches(hist_dir, new_dir, out_dir, run_timestamp):
    hist_file = get_latest_file(hist_dir)
    hist = pd.read_csv(hist_file, dtype={'GAME_ID': str})
    new = load_all_csvs(new_dir)
    all_boxscores = pd.concat([hist, new])
    merged_path = save_dataframe_to_csv(all_boxscores, out_dir, prefix='merged_boxscores', suffix=run_timestamp)
    print(f"Merged boxscores from {hist_file} and {new_dir}  saved at {merged_path}")
    return merged_path
