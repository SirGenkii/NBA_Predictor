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


cols_to_sum = [
    'fieldGoalsMade_traditional', 'fieldGoalsAttempted_traditional',
    'threePointersMade_traditional', 'threePointersAttempted_traditional',
    'freeThrowsMade_traditional', 'freeThrowsAttempted_traditional',
    'reboundsOffensive_traditional', 'reboundsDefensive_traditional',
    'reboundsTotal_traditional', 'assists_traditional', 'steals_traditional',
    'blocks_traditional', 'turnovers_traditional', 'foulsPersonal_traditional',
    'points_traditional', 'MINUTES_PLAYED',
    'pointsOffTurnovers_misc', 'pointsSecondChance_misc', 'pointsFastBreak_misc',
    'pointsPaint_misc', 'blocksAgainst_misc'
]

cols_to_weighted_avg = [
    'offensiveRating_advanced', 'defensiveRating_advanced', 'netRating_advanced',
    'assistPercentage_advanced', 'assistToTurnover_advanced', 'assistRatio_advanced',
    'offensiveReboundPercentage_advanced', 'defensiveReboundPercentage_advanced',
    'reboundPercentage_advanced', 'turnoverRatio_advanced', 'effectiveFieldGoalPercentage_advanced',
    'trueShootingPercentage_advanced', 'usagePercentage_advanced', 'estimatedPace_advanced',
    'pace_advanced', 'possessions_advanced', 'PIE_advanced',
    'effectiveFieldGoalPercentage_fourfactors', 'freeThrowAttemptRate_fourfactors',
    'teamTurnoverPercentage_fourfactors', 'oppEffectiveFieldGoalPercentage_fourfactors',
    'oppFreeThrowAttemptRate_fourfactors', 'oppTeamTurnoverPercentage_fourfactors',
    'oppOffensiveReboundPercentage_fourfactors', 'foulsDrawn_misc',
    'percentageFieldGoalsAttempted2pt_scoring', 'percentageFieldGoalsAttempted3pt_scoring',
    'percentagePoints2pt_scoring', 'percentagePointsMidrange2pt_scoring',
    'percentagePoints3pt_scoring', 'percentagePointsFastBreak_scoring',
    'percentagePointsFreeThrow_scoring', 'percentagePointsOffTurnovers_scoring',
    'percentagePointsPaint_scoring', 'percentageAssisted2pt_scoring',
    'percentageUnassisted2pt_scoring', 'percentageAssisted3pt_scoring',
    'percentageUnassisted3pt_scoring', 'percentageAssistedFGM_scoring',
    'percentageUnassistedFGM_scoring', 'threePointersPercentage_traditional',
    'freeThrowsPercentage_traditional', 'percentageFieldGoalsMade_usage',
    'percentageFieldGoalsAttempted_usage', 'percentageThreePointersMade_usage',
    'percentageThreePointersAttempted_usage', 'percentageFreeThrowsMade_usage',
    'percentageFreeThrowsAttempted_usage', 'percentageReboundsOffensive_usage',
    'percentageReboundsDefensive_usage', 'percentageReboundsTotal_usage',
    'percentageAssists_usage', 'percentageTurnovers_usage', 'percentageSteals_usage',
    'percentageBlocks_usage', 'percentageBlocksAllowed_usage', 'percentagePersonalFouls_usage',
    'percentagePersonalFoulsDrawn_usage', 'percentagePoints_usage','plusMinusPoints_traditional'
]

cols_player_stats = [
    'num_absent',
    'num_injured',
    'num_personal',
    'num_present',
    'num_resting',
    'num_suspended',
    'num_absent_other',
    'player_perf_score_mean',
    'player_perf_score_sum',
    'top_player_absent',
    'top_player_absent_rate',
    'top_player_count',
    'top_player_injured',
    'top_player_injury_rate',
    'top_player_personal',
    'top_player_personal_rate',
    'top_player_resting',
    'top_player_resting_rate',
    'top_player_suspended',
    'top_player_suspension_rate',
    'top_player_absent_other',
    'top_player_absent_other_rate',   
]
    
player_absent_input_cols = [
    'has_top_absent',
    'has_absent',
    'top_player_absent',
    'num_absent',
]

features_to_roll = cols_to_sum + cols_to_weighted_avg 
features_to_roll += [f"OPP_{col}" for col in cols_to_sum + cols_to_weighted_avg]
top_player_features_to_roll = cols_player_stats.copy()
top_player_features_to_roll += [f"OPP_{col}" for col in cols_player_stats] 

N_LIST = [3, 5, 10, 25, 50, 100, 200]

N_LIST_TOP = [1, 2, 3, 5, 10]

DATA_DIR = 'data'
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')


DATA_BRONZE_DIR = os.path.join(DATA_DIR, '01_bronze')
DATA_SILVER_DIR = os.path.join(DATA_DIR, '02_silver')
DATA_GOLD_DIR = os.path.join(DATA_DIR, '03_gold')
DATA_BRONZE_MATCHES_DIR = os.path.join(DATA_BRONZE_DIR, 'matches')

DATA_BRONZE_BOXSCORES_DIR =  os.path.join(DATA_BRONZE_DIR, 'boxscores')
DATA_BRONZE_TEAMS_DIR =  os.path.join(DATA_BRONZE_DIR, 'teams')
DATA_BRONZE_GAMES_DIR =  os.path.join(DATA_BRONZE_DIR, 'games')

DATA_GAMES_DIR = os.path.join(DATA_RAW_DIR, 'games')
DATA_PLAYERS_DIR = os.path.join(DATA_RAW_DIR, 'players')
DATA_BOXSCORES_DIR = os.path.join(DATA_RAW_DIR, 'boxscores')
DATA_BOXSCORES_BATCHES_DIR = os.path.join(DATA_BOXSCORES_DIR, 'batches')
DATA_BOXSCORES_BATCHES_MERGED_DIR = os.path.join(DATA_BOXSCORES_DIR, 'batches_merged')



DATA_RAW_LAST_DIR = os.path.join(DATA_DIR, 'raw_last')
DATA_LAST_GAMES_DIR = os.path.join(DATA_RAW_LAST_DIR, 'games')
DATA_LAST_GAMES_MERGED_DIR = DATA_BRONZE_GAMES_DIR #os.path.join(DATA_RAW_LAST_DIR, 'games_merged')
DATA_LAST_BOXSCORES_DIR = os.path.join(DATA_RAW_LAST_DIR, 'boxscores')
DATA_LAST_BOXSCORES_BATCHES_DIR = os.path.join(DATA_RAW_LAST_DIR, 'batches')
DATA_LAST_BOXSCORES_BATCHES_MERGED_DIR = DATA_BRONZE_BOXSCORES_DIR #os.path.join(DATA_RAW_LAST_DIR, 'batches_merged')
DATA_LAST_PLAYERS_STATS_DIR = os.path.join(DATA_RAW_LAST_DIR, 'player_stats')

DATA_FINAL_DATASET_DIR = DATA_SILVER_DIR #os.path.join(DATA_DIR, 'final_dataset')
DATA_FINAL_CLEANED_DATASET_DIR = DATA_GOLD_DIR #os.path.join(DATA_DIR, 'final_cleaned_dataset')

DATA_TEAMS_DIR = DATA_BRONZE_TEAMS_DIR #os.path.join(DATA_RAW_DIR, 'teams')

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
