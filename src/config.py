import os
import sys

DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')

DATA_GAMES_DIR = os.path.join(DATA_RAW_DIR, 'games')
DATA_PLAYERS_DIR = os.path.join(DATA_RAW_DIR, 'players')
DATA_BOXSCORES_DIR = os.path.join(DATA_RAW_DIR, 'boxscores')
DATA_BOXSCORES_BATCHES_DIR = os.path.join(DATA_BOXSCORES_DIR, 'batches')
DATA_BOXSCORES_BATCHES_MERGED_DIR = os.path.join(DATA_BOXSCORES_DIR, 'batches_merged')


DATA_RAW_LAST_DIR = os.path.join(DATA_DIR, 'raw_last')
DATA_LAST_GAMES_DIR = os.path.join(DATA_RAW_LAST_DIR, 'games')
DATA_LAST_GAMES_MERGED_DIR = os.path.join(DATA_RAW_LAST_DIR, 'games_merged')
DATA_LAST_BOXSCORES_DIR = os.path.join(DATA_RAW_LAST_DIR, 'boxscores')
DATA_LAST_BOXSCORES_BATCHES_DIR = os.path.join(DATA_RAW_LAST_DIR, 'batches')
DATA_LAST_BOXSCORES_BATCHES_MERGED_DIR = os.path.join(DATA_RAW_LAST_DIR, 'batches_merged')

DATA_FINAL_DATASET_DIR = os.path.join(DATA_DIR, 'final_dataset')
DATA_FINAL_CLEANED_DATASET_DIR = os.path.join(DATA_DIR, 'final_cleaned_dataset')

DATA_TEAMS_DIR = os.path.join(DATA_RAW_DIR, 'teams')

DATA_PREDICTION_ROWS_DIR = os.path.join(DATA_DIR, 'predictions_rows')

BATCH_SIZE = 25

ERROR_LOG_FOLDER = 'logs'

FULL_CSV = 'nba_player_boxscores_full.csv'