from pathlib import Path


# Liste des colonnes à dropper pour clean dataset final.
# Currently keeping ['GAME_DATE','GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'POINT_DIFF','SEASON'] to drop them before modeling.
COLS_MATCH_REAL = [
    # Identifiants et logs
    "OPP_GAME_DATE",
    "MATCHUP",
    "IS_WIN_SHIFTED",
    # ajoutés par les odds
    "TEAM_NAME",
    "OPPONENT_NAME",
]

COLS_TO_DROP_TARGET_IS_WIN = [
    "GAME_DATE",
    "GAME_ID",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "POINT_DIFF",
    "POINT_TOTAL",
    "SEASON",
]
COLS_TO_DROP_TARGET_POINT_DIFF = [
    "GAME_DATE",
    "GAME_ID",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "IS_WIN",
    "SEASON",
]

COLS_ODDS = ["ODDS", "OPP_ODDS", "HOME_MONEYLINE", "AWAY_MONEYLINE"]


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

# Availability base columns reused by multiple modules
PLAYER_AVAILABILITY_BASE = [
    "has_absent",
    "has_top_absent",
    "num_absent",
    "top_player_absent",
    "top_player_absent_rate",
    "top_player_injury_rate",
    "top_player_resting_rate",
    "top_player_suspension_rate",
    "top_player_personal_rate",
    "top_player_absent_other_rate",
    "top_player_count",
    "num_injured",
    "num_resting",
    "num_suspended",
    "num_personal",
    "num_absent_other",
] + player_absent_input_cols
PLAYER_AVAILABILITY_BASE = list(dict.fromkeys(PLAYER_AVAILABILITY_BASE))
PLAYER_AVAILABILITY_WINDOWS = [3, 5, 10, 25]

MATCHUP_WINDOWS = [5, 10, 25]

# Raw stat columns that seed the match-level representation before derived features
BASE_TEAM_FEATURE_COLUMNS = sorted(
    set(cols_to_sum + cols_to_weighted_avg + cols_player_stats + player_absent_input_cols)
)

RESULT_BASE_COLUMNS = ["IS_WIN", "POINTS_FOR", "POINTS_AGAINST", "POINT_DIFF", "POINT_TOTAL"]

MATCH_IDENTIFIER_COLUMNS = ["GAME_ID", "GAME_DATE", "SEASON"]
MATCH_SIDE_PREFIXES = ("HOME", "AWAY")
MATCH_ALLOWED_PREFIXES = ("HOME_", "AWAY_", "DIFF_", "MATCH_", "TOTAL_", "ODDS_", "IMPLIED_")
MATCH_ALLOWED_BASE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "IS_WIN",
    "POINT_DIFF",
    "POINT_TOTAL",
]
    
features_to_roll = cols_to_sum + cols_to_weighted_avg 
features_to_roll += [f"OPP_{col}" for col in cols_to_sum + cols_to_weighted_avg]
top_player_features_to_roll = cols_player_stats.copy()
top_player_features_to_roll += [f"OPP_{col}" for col in cols_player_stats] 

N_LIST = [3, 5, 10, 25, 50, 100, 200]

DATA_ROOT = Path("data")
DATA_BRONZE_DIR = DATA_ROOT / "01_bronze"
DATA_SILVER_DIR = DATA_ROOT / "02_silver"
DATA_GOLD_DIR = DATA_ROOT / "03_gold"
DATA_BRONZE_MATCHES_DIR = DATA_BRONZE_DIR / "matches"
DATA_BRONZE_BOXSCORES_DIR = DATA_BRONZE_DIR / "boxscores"
DATA_BRONZE_GAMES_DIR = DATA_BRONZE_DIR / "games"
DATA_TEAMS_DIR = DATA_BRONZE_DIR / "teams"

DATA_LAST_ROOT = DATA_ROOT / "raw_last"
DATA_LAST_GAMES_DIR = DATA_LAST_ROOT / "games"
DATA_LAST_GAMES_MERGED_DIR = DATA_BRONZE_GAMES_DIR
DATA_LAST_BOXSCORES_BATCHES_DIR = DATA_LAST_ROOT / "boxscores"
DATA_LAST_BOXSCORES_BATCHES_MERGED_DIR = DATA_BRONZE_BOXSCORES_DIR
DATA_LAST_PLAYERS_STATS_DIR = DATA_LAST_ROOT / "player_stats"

DATA_FINAL_DATASET_DIR = DATA_SILVER_DIR
DATA_FINAL_CLEANED_DATASET_DIR = DATA_GOLD_DIR

DATA_ODDS_HISTORY_DIR = DATA_ROOT / "odds_history"
DATA_MODELS_SIMULATIONS_DIR = DATA_ROOT / "models" / "simulations"
DATA_GRID_SIMULATIONS_BETS_DIR = DATA_ROOT / "grid_simulations"

ERROR_LOG_FOLDER = Path("logs")
BOXSCORES_SCRAPPING_LOG_FILE = ERROR_LOG_FOLDER / "boxscores_scrapping.log"
