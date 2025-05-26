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



def compute_ev_and_kelly(prob_win, odds, bankroll, kelly_fraction=1.0):
    """
    Calcule l'EV et la mise selon Kelly.
    
    :param prob_win: probabilité estimée de victoire (ex: 0.643)
    :param odds: cote du bookmaker (ex: 1.87)
    :param bankroll: bankroll actuelle en €
    :param kelly_fraction: fraction de Kelly (1.0 = plein Kelly, 0.5 = prudent, etc.)
    :return: dict avec EV, Kelly full, mise suggérée
    """
    b = odds - 1
    q = 1 - prob_win

    # EV
    ev = (prob_win * b) - q

    # Kelly
    kelly = ((b * prob_win) - q) / b if b > 0 else 0
    kelly = max(kelly, 0)  # Ne jamais parier avec Kelly négatif

    stake = bankroll * kelly * kelly_fraction

    return {
        'ev': round(ev, 4),
        'kelly_fraction': round(kelly, 4),
        'stake': round(stake, 2)
    }


# === EXEMPLE D’UTILISATION ===

# # Paramètres du match
# prob = 0.643379         # Probabilité donnée par ton modèle
# odds = 1.87             # Cote actuelle
# bankroll = 100          # Bankroll actuelle en euros
# kelly_frac = 1.0        # 1.0 = Kelly plein, 0.5 = Kelly modéré

# result = compute_ev_and_kelly(prob, odds, bankroll, kelly_fraction=kelly_frac)

# print("✅ Résultat du calcul :")
# print(f"EV : {result['ev']}")
# print(f"Kelly (fraction) : {result['kelly_fraction']}")
# print(f"Mise suggérée : {result['stake']} €")
