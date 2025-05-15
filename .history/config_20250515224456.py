import os
import sys

DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')

DATA_MATCHES_DIR = os.path.join(DATA_RAW_DIR, 'matches')


BATCH_SIZE = 25
ERROR_LOG_FOLDER = 'nba_error_logs'
FULL_CSV = 'nba_player_boxscores_full.csv'