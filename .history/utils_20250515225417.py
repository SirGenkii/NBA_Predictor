#get latest file from CLEANED_DATA_DIR. All files end with current date with following format: '%Y-%m-%d_%H-%M-%S'

# match_files = os.listdir(CLEANED_DATA_DIR)

# #get last generated file
# last_file = sorted(match_files)[-1]

# ALL_ATP_MATCHES = os.path.join(CLEANED_DATA_DIR, last_file)
# print("Latest match file found : ", os.path.join(CLEANED_DATA_DIR, last_file))


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
    filename = f"{prefix}_{suffix}.csv"
    
    # Create the full path
    file_path = os.path.join(output_dir, filename)
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    
    return file_path