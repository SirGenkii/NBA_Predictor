#function to save a dataframe to a csv file in dir parameter, with prefix parameter

import os
import pandas as pd
import numpy as np
from src.config import *

def save_dataframe_to_csv(df, output_dir, prefix, suffix=None):
    """
    Save a DataFrame to a CSV file with a specified prefix and suffix in the filename.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    output_dir (str): The directory where the CSV file will be saved.
    prefix (str): The prefix for the filename.
    suffix (str): The suffix for the filename.

    Returns:
    str: The path to the saved CSV file.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename
    if suffix:
        filename = f"{prefix}_{suffix}.csv"
    else:
        filename = f"{prefix}.csv"
    
    # Create the full path
    file_path = os.path.join(output_dir, filename)
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    
    return file_path

def get_latest_file(directory):
    """
    Get the latest file in a directory based on the filename.

    Parameters:
    directory (str): The directory to search for files.

    Returns:
    str: The path to the latest file.
    """
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Sort files and get the last one
    latest_file = sorted(files)[-1]
    
    return os.path.join(directory, latest_file)

def get_latest_dir_child(directory):
    """
    Get the latest directory in a parent directory.

    Parameters:
    directory (str): The parent directory to search for child directories.

    Returns:
    str: The path to the latest child directory.
    """
    
    # List all directories in the parent directory
    dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    # Sort directories and get the last one
    latest_dir = sorted(dirs)[-1]
    
    return os.path.join(directory, latest_dir)



def json_serial(obj):
    if isinstance(obj, (np.int64, np.float64)):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_team_mapping_id():
    team_mapping_file = get_latest_file(DATA_TEAMS_DIR)
    team_mapping = pd.read_csv(team_mapping_file)
    team_id_map = dict(zip(team_mapping["id"].astype(str), team_mapping["full_name"]))
    
    return team_id_map

def merge_odds_csv_files(odds_files_path):
    """
    Merge multiple odds CSV files into a single DataFrame.

    Parameters:
    odds_files (list): List of paths to the odds CSV files.

    Returns:
    pd.DataFrame: Merged DataFrame containing all odds data.
    """
    
    import glob
    import pandas as pd
    
    # Get all CSV files in the specified directory
    odds_files = glob.glob(os.path.join(odds_files_path, "*.csv"))
    if not odds_files:
        raise ValueError("No CSV files found in the specified directory.")
    
    dfs = []
    for file in odds_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df


def log_boxscores_scrapping(message):
    boxscores_log_file = BOXSCORES_SCRAPPING_LOG_FILE
    
    current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Ensure the error log folder exists
    os.makedirs(os.path.dirname(boxscores_log_file), exist_ok=True)
    with open(boxscores_log_file, 'a') as f:
        f.write(f"{current_time} - {message}\n")
        
        
def validate_model_inputs(df: pd.DataFrame, expected_cols: list, model_name: str = "model"):
    missing_cols = [col for col in expected_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_cols]

    if missing_cols:
        raise ValueError(f"[{model_name}] Données incomplètes : colonnes manquantes {missing_cols}")
    if len(extra_cols) > 50:  # arbitrairement 50 pour signaler du bruit
        print(f"[{model_name}] ⚠️ Attention : {len(extra_cols)} colonnes en trop dans les inputs")

    df_checked = df[expected_cols].copy()

    # Optionnel : vérif type numérique
    non_numeric_cols = df_checked.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        print(f"[{model_name}] ⚠️ Colonnes non-numériques dans les inputs : {non_numeric_cols}")

    return df_checked


def prepare_model_input(df: pd.DataFrame, target: str = "IS_WIN", model=None, drop_odds=True, verbose=True):
    """
    Prépare les données pour l'entraînement ou la prédiction.

    Args:
        df (pd.DataFrame): Données brutes contenant toutes les colonnes.
        target (str): Soit "IS_WIN" soit "POINT_DIFF".
        model: pipeline déjà entraîné (optionnel, pour vérifier les features exactes)
        drop_odds (bool): si True, on retire les colonnes de cotes.
        verbose (bool): Affiche les colonnes manquantes / en trop.

    Returns:
        pd.DataFrame: features prêtes à être utilisées.
    """

    if target == "IS_WIN":
        drop_cols = COLS_TO_DROP_TARGET_IS_WIN
    elif target == "POINT_DIFF":
        drop_cols = COLS_TO_DROP_TARGET_POINT_DIFF
    else:
        raise ValueError("Target non reconnue. Choisir 'IS_WIN' ou 'POINT_DIFF'")

    if drop_odds:
        drop_cols += COLS_ODDS

    drop_cols += COLS_MATCH_REAL + features_to_roll + top_player_features_to_roll
    features = [col for col in df.columns if col not in drop_cols + [target]]
    
    #drop nan
    df = df.dropna(subset=features)

    if model:
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in features]
        extra = [c for c in features if c not in expected]

        if verbose:
            if missing:
                print(f"❌ Colonnes manquantes : {missing}")
            if len(extra) > 1:
                print(f"⚠️ Trop de colonnes inutiles : {len(extra)}. \n colonnes : {extra}")
                

        features = expected

    return df[features].select_dtypes(include=["number"]).copy()


def get_optimal_model_params(model_name: str, use_gpu: bool = True, max_cpu_jobs: int = 3):
    """
    Retourne les meilleurs paramètres pour un modèle donné en fonction des ressources disponibles.

    Args:
        model_name (str): Nom du modèle ('xgb', 'lgbm', 'cat', 'rf', etc.)
        use_gpu (bool): Utilise le GPU si possible
        max_cpu_jobs (int): Limite du nombre de threads CPU à utiliser

    Returns:
        dict: Paramètres à passer au modèle
    """
    params = {}
    if model_name == "xgb":
        params.update({
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "eval_metric": "logloss",
            "random_state": 42,
            "use_label_encoder": False,
        })
        if use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["n_jobs"] = max_cpu_jobs

    elif model_name == "lgbm":
        params.update({
            "n_estimators": 150,
            "num_leaves": 64,
            "random_state": 42,
        })
        if use_gpu:
            params["device"] = "gpu"
        else:
            params["n_jobs"] = max_cpu_jobs

    elif model_name == "cat":
        params.update({
            "n_estimators": 200,
            "learning_rate": 0.05,
            "depth": 6,
            #"rsm": 0.8,
            "verbose": 0,
            "random_state": 42,
        })
        if use_gpu:
            params["task_type"] = "GPU"
        else:
            params["task_type"] = "CPU"
            params["thread_count"] = max_cpu_jobs

    elif model_name == "rf":
        params.update({
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": max_cpu_jobs
        })

    return params
