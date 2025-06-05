import os
import pandas as pd
import time
import random
import glob
from datetime import datetime
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

            time.sleep(random.uniform(1.0, 2.5))

            if (idx + 1) % int(batch_size) == 0 or (idx + 1) == len(filtered_game_ids):
                batch_num += 1
                for key, df_list in all_data.items():
                    if df_list:
                        df = pd.concat(df_list, ignore_index=True)
                        
                        batch_num_formatted = f"{batch_num:03d}"
                        
                        filename = f"boxscores_{key}_v3_batch_{batch_num_formatted}.csv"
                        filepath = os.path.join(endpoint_dirs[key], filename)
                        df.to_csv(filepath, index=False)
                        log_msg_save_batch = f" {key} V3 batch {batch_num_formatted} saved ({len(df)} rows)"
                        print(log_msg_save_batch)
                        log_boxscores_scrapping(log_msg_save_batch)
                        df_list.clear()

                if error_log:
                    for key in all_data.keys():
                        error_path = os.path.join(endpoint_dirs[key], f"errors_batch_{batch_num_formatted}.txt")
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

    # Move old error files before retrying
    for key in endpoints:
        error_dir = os.path.join(season_folder_path, key, 'errors_processed')
        os.makedirs(error_dir, exist_ok=True)
        for err_file in glob.glob(os.path.join(season_folder_path, key, 'errors_batch_*.txt')):
            print(f"[MOVE] {err_file} to {error_dir}")
            shutil.move(err_file, os.path.join(error_dir, os.path.basename(err_file)))

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




def merge_boxscore_batches_for_season(season_folder_path):
    endpoints = ['traditional', 'advanced', 'fourfactors', 'misc', 'scoring', 'usage']
    merged_dir = os.path.join(season_folder_path, 'merged_batches')
    os.makedirs(merged_dir, exist_ok=True)

    for endpoint in endpoints:
        batch_files = glob.glob(os.path.join(season_folder_path, endpoint, 'boxscores_*_batch_*.csv'))
        if not batch_files:
            print(f"[WARN] No batch files found for {endpoint} in {season_folder_path}")
            continue

        all_dfs = []
        for f in batch_files:
            df = pd.read_csv(f, dtype={'gameId': str})
            all_dfs.append(df)

        merged_df = pd.concat(all_dfs, ignore_index=True)

        if set(['gameId', 'teamId', 'playerSlug', 'minutes']) <= set(merged_df.columns):
            merged_df.drop_duplicates(subset=['gameId', 'teamId', 'playerSlug', 'minutes'], inplace=True)

        out_path = os.path.join(merged_dir, f'merged_{endpoint}.csv')
        merged_df.to_csv(out_path, index=False)
        print(f"[MERGED] {endpoint} saved to {out_path} ({len(merged_df)} rows)")


def merge_all_boxscore_stats(season_merged_dir, output_filename="final_merged_all_boxscores.csv"):
    # Chargement des fichiers
    endpoints = ['advanced', 'fourfactors', 'misc', 'scoring', 'traditional', 'usage']
    dfs = {}

    for endpoint in endpoints:
        path = os.path.join(season_merged_dir, f"merged_{endpoint}.csv")
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            return None

        df = pd.read_csv(path, dtype={'gameId': str})
        # Supprimer les colonnes dupliquées (hors clés)
        keep_cols = ['gameId', 'teamId', 'playerSlug', 'minutes']
        drop_cols = [c for c in df.columns if c not in keep_cols and df.columns.duplicated().sum() == 0]
        renamed = df.drop(columns=[col for col in df.columns if col not in keep_cols and col not in drop_cols])
        df = df.drop(columns=[col for col in df.columns if col not in keep_cols and col not in drop_cols])
        rename_map = {col: f"{col}_{endpoint}" for col in df.columns if col not in keep_cols}
        df = df.rename(columns=rename_map)
        dfs[endpoint] = df

    # Fusion progressive
    merged_df = dfs['advanced']
    for endpoint in endpoints[1:]:
        merged_df = merged_df.merge(dfs[endpoint], on=['gameId', 'teamId', 'playerSlug', 'minutes'], how='inner')


    # Nettoyage des colonnes
    print("Shape before cleaning:", merged_df.shape)
    
    merged_df = clean_merged_boxscore_dataframe(merged_df)
    
    print("Shape after cleaning:", merged_df.shape)

    # Export final
    output_path = os.path.join(season_merged_dir, output_filename)
    merged_df.to_csv(output_path, index=False)
    print(f"✅ All endpoints merged into: {output_path} ({len(merged_df)} rows)")
    return output_path


def clean_merged_boxscore_dataframe(df):
    columns_to_keep_from_advanced = [
        'teamCity', 'teamName', 'teamTricode', 'teamSlug',
        'personId', 'firstName', 'familyName', 'nameI',
        'position', 'comment', 'jerseyNum'
    ]

    rename_map = {f"{col}_advanced": col for col in columns_to_keep_from_advanced}
    df.rename(columns=rename_map, inplace=True)

    to_drop = []
    for col in df.columns:
        for base in columns_to_keep_from_advanced:
            if col.startswith(f"{base}_"):
                to_drop.append(col)

    df.drop(columns=to_drop, inplace=True)
    print(f"[CLEAN] Renamed and dropped {len(to_drop)} redundant columns.")


    # Remove too much correlated columns analysed from analyze_redundant_columns
    # [ANALYSIS] Found 15 highly correlated pairs (r > 0.95):
    # - estimatedOffensiveRating_advanced <--> offensiveRating_advanced => remove estimatedOffensiveRating_advanced
    # - effectiveFieldGoalPercentage_fourfactors <--> offensiveRating_advanced => keep both
    # - defensiveRating_advanced <--> estimatedDefensiveRating_advanced => remove estimatedDefensiveRating_advanced
    # - estimatedDefensiveRating_advanced <--> oppEffectiveFieldGoalPercentage_fourfactors => keep both
    # - offensiveReboundPercentage_advanced <--> offensiveReboundPercentage_fourfactors => remove offensiveReboundPercentage_fourfactors
    # - effectiveFieldGoalPercentage_advanced <--> trueShootingPercentage_advanced  <--> fieldGoalsPercentage_traditional => remove fieldGoalsPercentage_traditional
    # - estimatedUsagePercentage_advanced <--> usagePercentage_advanced <--> usagePercentage_usage => remove estimatedUsagePercentage_advanced and usagePercentage_usage
    # - pacePer40_advanced <--> pace_advanced => remove pacePer40_advanced
    # - blocks_misc <--> blocks_traditional => remove blocks_misc
    # - foulsPersonal_misc <--> foulsPersonal_traditional => remove foulsPersonal_misc
    # - fieldGoalsMade_traditional <--> points_traditional => keep both
    # - freeThrowsAttempted_traditional <--> freeThrowsMade_traditional => keep both

    cols_correlated_to_drop = [
        'estimatedOffensiveRating_advanced',
        'estimatedDefensiveRating_advanced',
        'offensiveReboundPercentage_fourfactors',
        'fieldGoalsPercentage_traditional',
        'estimatedUsagePercentage_advanced','usagePercentage_usage',
        'pacePer40_advanced',
        'blocks_misc',
        'foulsPersonal_misc',
    ]
    
    df.drop(columns=cols_correlated_to_drop, inplace=True)
    print(f"[CLEAN] Dropped {len(cols_correlated_to_drop)} highly correlated columns.")



    return df

def analyze_redundant_columns(df):
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.dropna(axis=1, how='all')

    # Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Filter high correlations
    high_corrs = []
    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if row != col and corr_matrix.loc[row, col] > 0.95:
                pair = tuple(sorted([row, col]))
                if pair not in high_corrs:
                    high_corrs.append(pair)

    print(f"[ANALYSIS] Found {len(high_corrs)} highly correlated pairs (r > 0.95):")
    for pair in high_corrs:
        print(f" - {pair[0]} <--> {pair[1]}")

    # Optional: heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, cbar_kws={'label': 'Absolute Correlation'})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


# # -- 5. Merge boxscores historiques et nouveaux
# def load_all_csvs(folder):
#     files = glob.glob(os.path.join(folder, "*.csv"))
#     dfs = [pd.read_csv(f, dtype={'GAME_ID': str}) for f in files]
#     if dfs:
#         return pd.concat(dfs, ignore_index=True)
#     else:
#         return pd.DataFrame()

# def merge_boxscores_batches(hist_dir, new_dir, out_dir, run_timestamp):
#     hist_file = get_latest_file(hist_dir)
#     hist = pd.read_csv(hist_file, dtype={'GAME_ID': str})
#     new = load_all_csvs(new_dir)
#     all_boxscores = pd.concat([hist, new])
#     merged_path = save_dataframe_to_csv(all_boxscores, out_dir, prefix='merged_boxscores', suffix=run_timestamp)
#     print(f"Merged boxscores from {hist_file} and {new_dir}  saved at {merged_path}")
#     return merged_path
