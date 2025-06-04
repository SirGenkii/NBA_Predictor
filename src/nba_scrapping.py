import os
import pandas as pd
import time
import random
import glob
from datetime import datetime
import shutil

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3, boxscorefourfactorsv3, boxscoremiscv3, boxscorescoringv3, boxscoreusagev3
from src.config import *
from src.utils import save_dataframe_to_csv, get_latest_file, log_boxscores_scrapping


# -- 1. Download all games for a season (or several)
def download_games_for_seasons(seasons, output_dir, run_timestamp, max_retries=3):
    all_games = []
    for season in seasons:
        print(f"Extraction saison {season}")
        retries = 0
        while retries < max_retries:
            try:
                gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00', season_nullable=season)
                games = gamefinder.get_data_frames()[0]
                games['SEASON'] = season
                all_games.append(games)
                time.sleep(2)
                break
            except Exception as e:
                print(f"   - Error downloading season {season}: {e}")
                retries += 1
                time.sleep(5)
                if retries >= max_retries:
                    raise Exception(f"Max retries exceeded for season {season}")

    df_games = pd.concat(all_games, ignore_index=True)
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
    df_games = df_games[df_games['GAME_ID'].astype(str).str.startswith(('002', '004', '005'))]
    df_games = df_games.sort_values(by='GAME_DATE')

    seasons_folder = f"{seasons[0]}_{seasons[-1]}"
    file_path = save_dataframe_to_csv(df_games, os.path.join(output_dir, seasons_folder), prefix=f'nba_games', suffix=run_timestamp)
    return file_path


# -- 2. Trouver les nouveaux matchs à scraper
def get_new_games(hist_games_path, new_games_path):
    hist_games = pd.read_csv(hist_games_path, dtype={'GAME_ID': str})
    new_games = pd.read_csv(new_games_path, dtype={'GAME_ID': str})
    old_ids = set(hist_games['GAME_ID'])
    to_add = new_games[~new_games['GAME_ID'].isin(old_ids)]
    print(f"{len(to_add)} nouveaux matchs à traiter")
    return to_add

def scrape_boxscores_v3_for_games(games_df, output_dir, batch_size=25, max_retries=5):
    import glob
    all_data = {
        'traditional': [], 'advanced': [], 'fourfactors': [],
        'misc': [], 'scoring': [], 'usage': []
    }
    error_log = []

    game_ids = games_df['GAME_ID'].unique().tolist()
    seasons = games_df['SEASON'].unique()

    for season in seasons:
        season_df = games_df[games_df['SEASON'] == season]
        season_game_ids = season_df['GAME_ID'].unique().tolist()

        endpoint_dirs = {}
        start_game_id = None

        for endpoint in all_data.keys():
            endpoint_dir = os.path.join(output_dir, endpoint)
            os.makedirs(endpoint_dir, exist_ok=True)
            endpoint_dirs[endpoint] = endpoint_dir
            batch_files = sorted(glob.glob(os.path.join(endpoint_dir, '*.csv')))
            
            print(f"[DEBUG] Found {len(batch_files)} batch files for endpoint '{endpoint}' in season {season}")
            
            if endpoint == 'advanced' and batch_files:
                last_file = batch_files[-1]
                print(f"[DEBUG] Checking last batch file for endpoint '{endpoint}': {last_file}")
                try:
                    df = pd.read_csv(last_file, usecols=['gameId'], dtype={'gameId': str})
                    if not df.empty:
                        last_game_id = df.iloc[-1]['gameId']
                        if start_game_id is None or season_game_ids.index(last_game_id) > season_game_ids.index(start_game_id):
                            start_game_id = last_game_id
                            print(f"[INFO] Resuming from gameId {start_game_id} in season {season}")
                except Exception as e:
                    print(f"[ERROR] Failed to read last gameId from {last_file}: {e}")
                    continue

        if start_game_id:
            start_index = season_game_ids.index(start_game_id) + 1
        else:
            start_index = 0

        filtered_game_ids = season_game_ids[start_index:]

        log_msg_nb_games = f"--------- {len(filtered_game_ids)} GAME_ID to scrap for season {season} ---------"
        print(log_msg_nb_games)
        log_boxscores_scrapping(log_msg_nb_games)

        batch_num = len(glob.glob(os.path.join(list(endpoint_dirs.values())[0], 'boxscores_*_v3_batch_*.csv')))
        timeout_streak = 0

        for idx, gid in enumerate(filtered_game_ids):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg_treatment = f"[{idx+1}/{len(filtered_game_ids)}] GAME_ID: {gid} - {season_df.loc[season_df['GAME_ID'] == gid, 'GAME_DATE'].values[0]}"
            print(log_msg_treatment)
            log_boxscores_scrapping(log_msg_treatment)

            try:
                for endpoint, func in {
                    'traditional': boxscoretraditionalv3.BoxScoreTraditionalV3,
                    'advanced': boxscoreadvancedv3.BoxScoreAdvancedV3,
                    'fourfactors': boxscorefourfactorsv3.BoxScoreFourFactorsV3,
                    'misc': boxscoremiscv3.BoxScoreMiscV3,
                    'scoring': boxscorescoringv3.BoxScoreScoringV3,
                    'usage': boxscoreusagev3.BoxScoreUsageV3
                }.items():
                    for attempt in range(max_retries):
                        try:
                            df = func(game_id=gid, timeout=30).player_stats.get_data_frame()
                            all_data[endpoint].append(df)
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise Exception(f"{endpoint} endpoint failed after {max_retries} attempts: {e}")
                            time.sleep(3)
                timeout_streak = 0
            except Exception as e:
                log_msg_error = f"   - Error for GAME_ID {gid}: {e}"
                print(log_msg_error)
                log_boxscores_scrapping(log_msg_error)
                error_log.append((gid, str(e)))
                timeout_streak += 1
                if timeout_streak >= 10:
                    log_msg_timeout_streak = "         - 10 consecutive timeouts, stopping scraping for 10 minutes"
                    print(log_msg_timeout_streak)
                    log_boxscores_scrapping(log_msg_timeout_streak)
                    time.sleep(600)
                    timeout_streak = 0
                else:
                    time.sleep(30)
                continue

            time.sleep(random.uniform(1.5, 3.5))

            if (idx + 1) % int(batch_size) == 0 or (idx + 1) == len(filtered_game_ids):
                batch_num += 1
                for key, df_list in all_data.items():
                    if df_list:
                        df = pd.concat(df_list, ignore_index=True)
                        filename = f"boxscores_{key}_v3_batch_{batch_num}.csv"
                        filepath = os.path.join(endpoint_dirs[key], filename)
                        df.to_csv(filepath, index=False)
                        log_msg_save_batch = f" {key} V3 batch {batch_num} saved ({len(df)} rows)"
                        print(log_msg_save_batch)
                        log_boxscores_scrapping(log_msg_save_batch)
                        df_list.clear()

                if error_log:
                    for key in all_data.keys():
                        error_path = os.path.join(endpoint_dirs[key], f"errors_batch_{batch_num}.txt")
                        with open(error_path, "a") as f:
                            for err in error_log:
                                f.write(f"{err[0]}\t{err[1]}\n")
                    error_log.clear()
    return True




# -- 4. Retry scraping for GAME_IDs that failed previously
def retry_failed_boxscores_for_season(season_folder_path, batch_size=25, max_retries=5):

    endpoints = ['traditional', 'advanced', 'fourfactors', 'misc', 'scoring', 'usage']
    season_failed_game_ids = {}

    # 1. Collect GAME_IDs from error files for each endpoint
    for endpoint in endpoints:
        error_dir = os.path.join(season_folder_path, endpoint)
        error_txts = glob.glob(os.path.join(error_dir, "errors_batch_*.txt"))
        failed_game_ids = set()
        for error_file in error_txts:
            with open(error_file, 'r') as f:
                for line in f:
                    gid = line.strip().split('\t')[0]
                    if gid:
                        failed_game_ids.add(gid)
        season_failed_game_ids[endpoint] = failed_game_ids

    # 2. Check consistency
    all_game_id_sets = list(season_failed_game_ids.values())
    if not all(len(all_game_id_sets[0].intersection(s)) == len(all_game_id_sets[0]) for s in all_game_id_sets[1:]):
        raise ValueError("❌ Inconsistent GAME_IDs across endpoints. Retry aborted.")

    game_ids_to_retry = sorted(list(season_failed_game_ids[endpoints[0]]))
    print(f"[INFO] Retrying {len(game_ids_to_retry)} GAME_IDs for season folder {season_folder_path}")

    # 3. Prepare
    all_data = {k: [] for k in endpoints}
    error_log = []
    retry_batch_num = 1

    # 4. Retry logic
    for idx, gid in enumerate(game_ids_to_retry):
        print(f"[RETRY] GAME_ID {gid} ({idx+1}/{len(game_ids_to_retry)})")
        for endpoint, func in {
            'traditional': boxscoretraditionalv3.BoxScoreTraditionalV3,
            'advanced': boxscoreadvancedv3.BoxScoreAdvancedV3,
            'fourfactors': boxscorefourfactorsv3.BoxScoreFourFactorsV3,
            'misc': boxscoremiscv3.BoxScoreMiscV3,
            'scoring': boxscorescoringv3.BoxScoreScoringV3,
            'usage': boxscoreusagev3.BoxScoreUsageV3
        }.items():
            for attempt in range(max_retries):
                try:
                    df = func(game_id=gid, timeout=30).player_stats.get_data_frame()
                    all_data[endpoint].append(df)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"   - Error on {endpoint} for GAME_ID {gid}: {e}")
                        error_log.append((gid, f"{endpoint}: {e}"))
                    time.sleep(5)

        time.sleep(random.uniform(1.5, 3.5))

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(game_ids_to_retry):
            for key, df_list in all_data.items():
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    filename = f"boxscores_{key}_v3_retry_batch_{retry_batch_num}.csv"
                    path = os.path.join(season_folder_path, key, filename)
                    df.to_csv(path, index=False)
                    print(f"[SAVED] {key} retry batch {retry_batch_num} ({len(df)} rows)")
                    df_list.clear()

            if error_log:
                for key in endpoints:
                    error_dir = os.path.join(season_folder_path, key, 'errors_processed')
                    os.makedirs(error_dir, exist_ok=True)
                    for err_file in glob.glob(os.path.join(season_folder_path, key, 'errors_batch_*.txt')):
                        shutil.move(err_file, os.path.join(error_dir, os.path.basename(err_file)))
                new_errors = {}
                for gid, message in error_log:
                    ep = message.split(':')[0]
                    if ep not in new_errors:
                        new_errors[ep] = []
                    new_errors[ep].append((gid, message))
                for ep, lines in new_errors.items():
                    error_file = os.path.join(season_folder_path, ep, f"errors_retry_batch_{retry_batch_num}.txt")
                    with open(error_file, 'a') as f:
                        for gid, msg in lines:
                            f.write(f"{gid}\t{msg}\n")
                error_log.clear()
            retry_batch_num += 1

    print("✅ Retry process completed.")

# -- 5. Merge boxscores historiques et nouveaux
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
