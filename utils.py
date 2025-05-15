#function to save a dataframe to a csv file in dir parameter, with prefix parameter

import os
import pandas as pd

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