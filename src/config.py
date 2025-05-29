import os
import sys


# Liste des colonnes à dropper (toutes les stats brutes et colonnes de match, identifiants inutiles, etc.)
COLS_MATCH_REAL = [
    # Identifiants et logs
    "OPP_GAME_DATE", "MATCHUP",

    "IS_WIN_SHIFTED", 

    # Stats brutes de match (pour les deux équipes)
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS", "MINUTES_PLAYED",
    "OPP_FGM", "OPP_FGA", "OPP_FG_PCT", "OPP_FG3M", "OPP_FG3A", "OPP_FG3_PCT", "OPP_FTM", "OPP_FTA", "OPP_FT_PCT",
    "OPP_OREB", "OPP_DREB", "OPP_REB", "OPP_AST", "OPP_STL", "OPP_BLK", "OPP_TO", "OPP_PF", "OPP_PTS",
    "OPP_PLUS_MINUS", "OPP_MINUTES_PLAYED",
    "POINT_DIFF",

    # Nouvelles features brutes ajoutées
    "TS_PCT", "EFG_PCT", "AST_TO_RATIO", "REB_RATE",
]



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

FULL_CSV = 'nba_player_boxscores_full.csv'