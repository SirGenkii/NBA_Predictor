import os
import sys

DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')

DATA_GAMES_DIR = os.path.join(DATA_RAW_DIR, 'games')
DATA_PLAYERS_DIR = os.path.join(DATA_RAW_DIR, 'players')
DATA_BOXSCORES_DIR = os.path.join(DATA_RAW_DIR, 'boxscores')
DATA_BOXSCORES_DIR = os.path.join(DATA_RAW_DIR, 'boxscores')

DATA_TEAMS_DIR = os.path.join(DATA_RAW_DIR, 'teams')

BATCH_SIZE = 25

ERROR_LOG_FOLDER = 'logs'

FULL_CSV = 'nba_player_boxscores_full.csv'