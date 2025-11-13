import os

from dotenv import load_dotenv


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Load .env before resolving config values
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or os.path.join(LOGS_DIR, "mlruns")
MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI", MLFLOW_TRACKING_URI)
MLFLOW_TOTALS_OU_EXPERIMENT = os.getenv("MLFLOW_TOTALS_OU_EXPERIMENT", "totals_ou")
MLFLOW_TOTALS_OU_MODEL_NAME = os.getenv("MLFLOW_TOTALS_OU_MODEL_NAME", "totals_ou_bo3_regressor")


PDF_TOURNAMENT_OPENAI_API_KEY = os.getenv("PDF_TOURNAMENT_OPENAI_API_KEY")
PDF_TOURNAMENT_OPENAI_MODEL = os.getenv("PDF_TOURNAMENT_OPENAI_MODEL", "gpt-5")
PDF_TOURNAMENT_CACHE_DIR = os.path.join(DATA_DIR, "tournament_identity_cache")
PDF_TOURNAMENT_MAX_OUTPUT_TOKENS = int(os.getenv("PDF_TOURNAMENT_MAX_OUTPUT_TOKENS", 15000))
PDF_TOURNAMENT_LOG_PATH = os.path.join(LOGS_DIR, "tournament_identity.log")

# Discord notifications
DISCORD_SUBSCRIBER_ROLE_ID = os.getenv("DISCORD_SUBSCRIBER_ROLE_ID")
DISCORD_ADMIN_ROLE_ID = os.getenv("DISCORD_ADMIN_ROLE_ID")

DISCORD_SUBSCRIBER_ROLE_MENTION = (
    f"<@&{DISCORD_SUBSCRIBER_ROLE_ID.strip()}>" if DISCORD_SUBSCRIBER_ROLE_ID else None
)
DISCORD_ADMIN_ROLE_MENTION = (
    f"<@&{DISCORD_ADMIN_ROLE_ID.strip()}>" if DISCORD_ADMIN_ROLE_ID else None
)



DISCORD_PYTHIA_EDGE_SERVER_ID = os.getenv("DISCORD_PYTHIA_EDGE_SERVER_ID")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_HEALTH_BOT_TOKEN = os.getenv("DISCORD_HEALTH_BOT_TOKEN")
DISCORD_SUBSCRIBER_PREDICTION_BOT_TOKEN = os.getenv("DISCORD_SUBSCRIBER_PREDICTION_BOT_TOKEN")
DISCORD_FREE_PREDICTION_BOT_TOKEN = os.getenv("DISCORD_FREE_PREDICTION_BOT_TOKEN")
DISCORD_WATCHER_DEBUG_BOT_TOKEN = os.getenv("DISCORD_WATCHER_DEBUG_BOT_TOKEN")
DISCORD_WELCOME_BOT_TOKEN = os.getenv("DISCORD_WELCOME_BOT_TOKEN")


DISCORD_HEALTH_CHANNEL_ID = os.getenv("DISCORD_HEALTH_CHANNEL_ID")
DISCORD_WATCHER_DEBUG_CHANNEL_ID = os.getenv("DISCORD_WATCHER_DEBUG_CHANNEL_ID")
DISCORD_SUBSCRIBER_PREDICTION_CHANNEL_ID = os.getenv("DISCORD_SUBSCRIBER_PREDICTION_CHANNEL_ID")
DISCORD_FREE_PREDICTION_CHANNEL_ID = os.getenv("DISCORD_FREE_PREDICTION_CHANNEL_ID")
DISCORD_WELCOME_CHANNEL_ID = os.getenv("DISCORD_WELCOME_CHANNEL_ID")
DISCORD_GUIDE_CHANNEL_ID = os.getenv("DISCORD_GUIDE_CHANNEL_ID")
DISCORD_RULES_CHANNEL_ID = os.getenv("DISCORD_RULES_CHANNEL_ID")

MACHINE_IDENTIFIER = os.getenv("MACHINE_IDENTIFIER", "unknown").strip() or "unknown"



# Sous-dossiers
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# --- TML repo settings ---

TML_REPO_URL  = "https://github.com/Tennismylife/TML-Database.git"
TML_REPO_DIR  = os.path.join(RAW_DATA_DIR, "git-data", "TML-Database")  # choose where to clone
TML_STATE_FILE = os.path.join(TML_REPO_DIR, ".last_ingested_commit")



RESULT_DATA_DIR = os.path.join(DATA_DIR, "results_prediction")

RESULT_BRONZE_DATA_DIR = os.path.join(RESULT_DATA_DIR, "01_bronze")
RESULT_SILVER_DATA_DIR = os.path.join(RESULT_DATA_DIR, "02_silver")
RESULT_GOLD_DATA_DIR = os.path.join(RESULT_DATA_DIR, "03_gold")
RESULT_GOLD_PARQUET_DATA_DIR = os.path.join(RESULT_DATA_DIR, "03_parquet")

RESULT_MODEL_DATA_DIR = os.path.join(MODELS_DIR, "results_prediction")

RESULT_BRONZE_FILENAME = "bronze_results_currentdate.csv"
RESULT_SILVER_FILENAME = "silver_results_currentdate.csv"
RESULT_GOLD_FILENAME = "gold_results_currentdate.csv"

RESULT_PREDICTIONS_LOG_DIR = os.path.join(RESULT_DATA_DIR, "predictions_logs")

MONITORING_DIR = os.path.join(DATA_DIR, "monitoring")
PREDICTION_MONITORING_DB = os.path.join(MONITORING_DIR, "predictions.db")
PREDICTION_MONITORING_LOG = os.path.join(LOGS_DIR, "prediction_linker.log")

# Mass prediction (screenshot workflow)
MASS_PREDICTION_DIR = os.path.join(DATA_DIR, "mass_result_prediction")
MASS_PREDICTION_SCREENSHOT_DIR = os.path.join(MASS_PREDICTION_DIR, "screenshots")
MASS_PREDICTION_SCREENSHOT_PROCESSED_DIR = os.path.join(MASS_PREDICTION_SCREENSHOT_DIR, "processed")
MASS_PREDICTION_SCREENSHOT_ERROR_DIR = os.path.join(MASS_PREDICTION_SCREENSHOT_DIR, "error")
MASS_PREDICTION_LOG_DIR = os.path.join(MASS_PREDICTION_DIR, "logs")
MASS_PREDICTION_EXPORT_PATTERN = os.path.join(MASS_PREDICTION_DIR, "mass_prediction_{date}.csv")
MASS_PREDICTION_EXPORT_DIR = os.path.dirname(MASS_PREDICTION_EXPORT_PATTERN)

MASS_PREDICTION_FEATURE_EXPORT_DIR =  os.path.join(MASS_PREDICTION_DIR, "features_predictions")
WEBSITE_MASS_PREDICTION_FEATURE_EXPORT_DIR =  os.path.join(MASS_PREDICTION_DIR, "website_features_predictions")
MASS_PREDICTION_WATCHER_STATE_FILE = os.path.join(DATA_DIR, "watcher_status", "watcher_status.json")

# env variable if set else default
DEFAULT_MAX_WORKERS = int(os.getenv("DEFAULT_MAX_WORKERS_ENV", 4))


# Nb-set prediction pipeline (BRONZE/SILVER/GOLD)
NBSET_DATA_DIR = os.path.join(DATA_DIR, "nbset_prediction")

NBSET_BRONZE_DATA_DIR = os.path.join(NBSET_DATA_DIR, "01_bronze")
NBSET_SILVER_DATA_DIR = os.path.join(NBSET_DATA_DIR, "02_silver")
NBSET_GOLD_DATA_DIR = os.path.join(NBSET_DATA_DIR, "03_gold")

NBSET_MODEL_DATA_DIR = os.path.join(MODELS_DIR, "nbset_prediction")

NBSET_BRONZE_FILENAME = "bronze_nbsets_currentdate.csv"
NBSET_SILVER_FILENAME = "silver_nbsets_currentdate.csv"
NBSET_GOLD_FILENAME = "gold_nbsets_currentdate.csv"  # legacy combined export
NBSET_GOLD_FILENAME_BO3 = "gold_nbsets_bo3_currentdate.csv"
NBSET_GOLD_FILENAME_BO5 = "gold_nbsets_bo5_currentdate.csv"

# Over/Under totals prediction pipeline
TOTALS_OU_DATA_DIR = os.path.join(DATA_DIR, "totals_ou_prediction")

TOTALS_OU_BRONZE_DATA_DIR = os.path.join(TOTALS_OU_DATA_DIR, "01_bronze")
TOTALS_OU_SILVER_DATA_DIR = os.path.join(TOTALS_OU_DATA_DIR, "02_silver")
TOTALS_OU_GOLD_DATA_DIR = os.path.join(TOTALS_OU_DATA_DIR, "03_gold")
TOTALS_OU_GOLD_PARQUET_DATA_DIR = os.path.join(TOTALS_OU_DATA_DIR, "03_parquet")

TOTALS_OU_MODEL_DATA_DIR = os.path.join(MODELS_DIR, "totals_ou_prediction")

TOTALS_OU_BRONZE_FILENAME = "bronze_totals_ou_currentdate.csv"
TOTALS_OU_SILVER_FILENAME = "silver_totals_ou_currentdate.csv"
TOTALS_OU_GOLD_FILENAME = "gold_totals_ou_currentdate.csv"

TOTALS_OU_METADATA_FILENAME = "metadata.json"

# Fichiers externes utilisés par l'application Streamlit
ATP_TOURNAMENTS_FILE = os.path.join(DATA_DIR, "atp_tournaments.csv")
PLAYERS_LIST_CSV = os.path.join(TML_REPO_DIR, "ATP_Database.csv")  # à ajuster selon l'emplacement réel

ATP_PLAYER_RANK_POINTS_DIR = os.path.join(DATA_DIR, "atp_rank_points") 

FLASHSCORE_DIR =  os.path.join(RAW_DATA_DIR, "flashscore")
FLASHSCORE_01_DIR =  os.path.join(FLASHSCORE_DIR, "01")
FLASHSCORE_02_DIR =  os.path.join(FLASHSCORE_DIR, "02")
FLASHSCORE_03_DIR =  os.path.join(FLASHSCORE_DIR, "03")
FLASHSCORE_04_DIR =  os.path.join(FLASHSCORE_DIR, "04")
FLASHSCORE_05_DIR =  os.path.join(FLASHSCORE_DIR, "05")

FLASHSCORE_MANUAL_ROUND_DATES_FILE = os.path.join(FLASHSCORE_DIR, "manual_round_dates.csv")

# Historical odds (tennis-data.co.uk)
FETCH_ODDS_USE_CURRENT_YEAR = os.getenv("FETCH_ODDS_USE_CURRENT_YEAR", 0)
ODDS_TENNISDATA_DIR = os.path.join(RAW_DATA_DIR, "odds_tennis-data")
ODDS_ODDSPORTAL_DIR = os.path.join(RAW_DATA_DIR, "odds_oddsportal")
ODDS_DEBUG_DIR = os.path.join(DATA_DIR, "odds_debug")

# Years you actually want to ingest
FIRST_DATASET_YEAR = 1970


PLAYERS_SNAPSHOT_DIR = os.path.join(DATA_DIR, "players_snapshot")
H2H_SNAPSHOT_DIR = os.path.join(DATA_DIR, "h2h_snapshot")
H2H_OVERALL_SNAPSHOT_DIR = os.path.join(H2H_SNAPSHOT_DIR, "overall")
H2H_SURFACE_SNAPSHOT_DIR = os.path.join(H2H_SNAPSHOT_DIR, "surface")
