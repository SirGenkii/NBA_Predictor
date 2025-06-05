import pandas as pd


def compute_weighted_mean_features(df, group_keys, feature_cols, weight_col):
    """
    Calcule la moyenne pondérée de colonnes numériques par groupe (typiquement par match et équipe)
    :param df: DataFrame d'entrée (boxscores)
    :param group_keys: liste des colonnes sur lesquelles faire le groupby (ex: ['GAME_ID', 'TEAM_ID'])
    :param feature_cols: liste des colonnes numériques à moyenner
    :param weight_col: colonne des poids (ex: 'MINUTES_PLAYED')
    :return: DataFrame agrégé
    """
    df = df.copy()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce').fillna(0)

    weighted_sums = df[feature_cols].multiply(df[weight_col], axis=0)
    weighted_sums[group_keys] = df[group_keys]
    weights = df[[weight_col] + group_keys].copy()

    grouped_weighted_sum = weighted_sums.groupby(group_keys).sum()
    grouped_weights = weights.groupby(group_keys)[weight_col].sum()

    for col in feature_cols:
        grouped_weighted_sum[col] = grouped_weighted_sum[col] / grouped_weights

    return grouped_weighted_sum.reset_index()

def compute_simple_mean_features(df, group_keys, feature_cols):
    """
    Calcule la moyenne simple de colonnes numériques par groupe
    :param df: DataFrame d'entrée
    :param group_keys: colonnes de groupby
    :param feature_cols: colonnes à moyenner
    :return: DataFrame agrégé
    """
    df = df.copy()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df.groupby(group_keys)[feature_cols].mean().reset_index()
