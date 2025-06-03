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