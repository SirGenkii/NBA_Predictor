import os
import pandas as pd
import time
import random
import glob
from datetime import datetime
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

# -- 3. Scraper les boxscores enrichis V3 des nouveaux GAME_ID avec gestion de reprise
def scrape_boxscores_v3_for_games(games_df, output_dir, batch_size=25):
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
        filtered_game_ids = set(season_game_ids)

        endpoint_dirs = {}
        last_game_ids = {}

        for endpoint in all_data.keys():
            #endpoint_dir = os.path.join(output_dir, season, endpoint)
            endpoint_dir = os.path.join(output_dir, endpoint)
            os.makedirs(endpoint_dir, exist_ok=True)
            endpoint_dirs[endpoint] = endpoint_dir
            batch_files = sorted(glob.glob(os.path.join(endpoint_dir, '*.csv')))

            existing_game_ids = set()
            for file in batch_files:
                try:
                    df = pd.read_csv(file, usecols=['GAME_ID'], dtype={'GAME_ID': str})
                    existing_game_ids.update(df['GAME_ID'].unique().tolist())
                except Exception:
                    continue

            filtered_game_ids = filtered_game_ids.intersection(set(season_game_ids) - existing_game_ids)
            last_game_ids[endpoint] = max(existing_game_ids) if existing_game_ids else None

        filtered_game_ids = sorted(list(filtered_game_ids))

        log_msg_nb_games = f"--------- {len(filtered_game_ids)} GAME_ID to scrap for season {season} ---------"
        print(log_msg_nb_games)
        log_boxscores_scrapping(log_msg_nb_games)

        batch_num = 0
        timeout_streak = 0

        for idx, gid in enumerate(filtered_game_ids):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg_treatment = f"[{idx+1}/{len(filtered_game_ids)}] GAME_ID: {gid} - {season_df.loc[season_df['GAME_ID'] == gid, 'GAME_DATE'].values[0]}"
            print(log_msg_treatment)
            log_boxscores_scrapping(log_msg_treatment)

            try:
                all_data['traditional'].append(boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=30).player_stats.get_data_frame())
                all_data['advanced'].append(boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid, timeout=30).player_stats.get_data_frame())
                all_data['fourfactors'].append(boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id=gid, timeout=30).player_stats.get_data_frame())
                all_data['misc'].append(boxscoremiscv3.BoxScoreMiscV3(game_id=gid, timeout=30).player_stats.get_data_frame())
                all_data['scoring'].append(boxscorescoringv3.BoxScoreScoringV3(game_id=gid, timeout=30).player_stats.get_data_frame())
                all_data['usage'].append(boxscoreusagev3.BoxScoreUsageV3(game_id=gid, timeout=30).player_stats.get_data_frame())
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

            time.sleep(random.uniform(3.5, 5.5))

            
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
def retry_failed_boxscores(run_dir):
    error_folder = os.path.join(run_dir, "errors")
    failed_game_ids = set()
    for file in glob.glob(os.path.join(error_folder, "errors_batch_*.txt")):
        with open(file, 'r') as f:
            for line in f:
                gid = line.strip().split('\t')[0]
                failed_game_ids.add(gid)

    print(f"Found {len(failed_game_ids)} failed GAME_IDs to retry.")

    all_data = {k: [] for k in ['traditional', 'advanced', 'fourfactors', 'misc', 'scoring', 'usage']}
    error_log = []
    retry_batch_num = 1

    for idx, gid in enumerate(sorted(failed_game_ids)):
        try:
            print(f"Retrying GAME_ID: {gid} ({idx+1}/{len(failed_game_ids)})")
            all_data['traditional'].append(boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['advanced'].append(boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['fourfactors'].append(boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['misc'].append(boxscoremiscv3.BoxScoreMiscV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['scoring'].append(boxscorescoringv3.BoxScoreScoringV3(game_id=gid, timeout=30).player_stats.get_data_frame())
            all_data['usage'].append(boxscoreusagev3.BoxScoreUsageV3(game_id=gid, timeout=30).player_stats.get_data_frame())
        except Exception as e:
            print(f"Error retrying GAME_ID {gid}: {e}")
            error_log.append((gid, str(e)))
            time.sleep(8)
            continue
        time.sleep(random.uniform(3, 5))

        if (idx + 1) % 25 == 0 or (idx + 1) == len(failed_game_ids):
            for key, df_list in all_data.items():
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    filename = f"boxscores_{key}_v3_retry_batch_{retry_batch_num}"
                    save_dataframe_to_csv(df, run_dir, prefix=filename)
                    print(f" Retry {key} V3 batch {retry_batch_num} saved")
                    df_list.clear()
            if error_log:
                with open(os.path.join(error_folder, f"errors_retry_batch_{retry_batch_num}.txt"), "a") as f:
                    for err in error_log:
                        f.write(f"{err[0]}\t{err[1]}\n")
                error_log.clear()
            retry_batch_num += 1

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
