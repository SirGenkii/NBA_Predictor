import os
import sys


# Liste des colonnes à dropper pour clean dataset final. 
# Currently keeping ['GAME_DATE','GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'POINT_DIFF','SEASON'] to drop them before modeling.
COLS_MATCH_REAL = [
    # Identifiants et logs
    "OPP_GAME_DATE", "MATCHUP","IS_WIN_SHIFTED",
    
    #ajoutés par les odds 
    "TEAM_NAME","OPPONENT_NAME"
]

COLS_TO_DROP_TARGET_IS_WIN = ['GAME_DATE','GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'POINT_DIFF','SEASON']
COLS_TO_DROP_TARGET_POINT_DIFF = ['GAME_DATE','GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'IS_WIN','SEASON']

COLS_ODDS = ["ODDS","OPP_ODDS"]

# FEATURES_TO_ROLL = [
#     'PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'PLUS_MINUS',
#     'TS_PCT', 'EFG_PCT', 'AST_TO_RATIO', 'REB_RATE',
#     "POSSESSIONS","OPP_POSSESSIONS",
#     "OFF_RATING","DEF_RATING" 
# ]

N_LIST = [3, 5, 10, 25, 50, 100, 200]



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

DATA_MODELS_DIR = os.path.join(DATA_DIR, 'models')
DATA_MODELS_SIMULATIONS_DIR = os.path.join(DATA_MODELS_DIR, 'simulations')

DATA_ODDS_HISTORY_DIR = os.path.join(DATA_DIR, 'odds_history')
DATA_SIMULATIONS_DIR = os.path.join(DATA_DIR, 'simulations')
DATA_GRID_SIMULATIONS_BETS_DIR = os.path.join(DATA_DIR, 'grid_simulations')

BATCH_SIZE = 25

ERROR_LOG_FOLDER = 'logs'
BOXSCORES_SCRAPPING_LOG_FILE = os.path.join(ERROR_LOG_FOLDER, 'boxscores_scrapping.log')

FULL_CSV = 'nba_player_boxscores_full.csv'