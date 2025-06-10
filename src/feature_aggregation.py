import pandas as pd
from typing import List
import os
from src.config import DATA_LAST_PLAYERS_STATS_DIR

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


def aggregate_actual_team_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les stats par match/équipe : moyenne/somme des scores de perf et totaux de présence/absence.
    Sauvegarde dans un CSV.
    """
    os.makedirs(DATA_LAST_PLAYERS_STATS_DIR, exist_ok=True)
    agg = df_matches.groupby(['GAME_ID', 'TEAM_ID', 'GAME_DATE']).agg(
        player_perf_score_mean=('player_perf_score', 'mean'),
        player_perf_score_sum=('player_perf_score', 'sum'),
        num_present=('is_present', 'sum'),
        num_absent=('is_absent', 'sum'),
        num_injured=('is_injured', 'sum'),
        num_suspended=('is_suspended', 'sum'),
        num_resting=('is_resting', 'sum'),
        num_personal=('is_personal', 'sum'),
    ).reset_index()
    agg.to_csv(os.path.join(
        DATA_LAST_PLAYERS_STATS_DIR, 'aggregate_actual_team_features.csv'
    ), index=False)
    return agg


def identify_historical_top_players(df_boxscores: pd.DataFrame,
                                    n_games: int = 10,
                                    top_n: int = 5) -> pd.DataFrame:
    """
    Identifie les top_n joueurs d'une équipe pour chaque GAME_ID, en se basant uniquement
    sur leurs scores dans les n_games précédents.
    Renvoie un DataFrame mapping (GAME_ID, TEAM_ID, personId), avec flag is_historical_top.
    """
    df = df_boxscores[['GAME_ID', 'TEAM_ID', 'personId', 'GAME_DATE', 'player_perf_score']].copy()
    df.sort_values(['TEAM_ID', 'personId', 'GAME_DATE'], inplace=True)

    records = []
    for (team, pid), sub in df.groupby(['TEAM_ID', 'personId']):
        sub = sub.sort_values('GAME_DATE')
        # rolling mean pour les derniers n_games performances
        sub['rolling_avg'] = sub['player_perf_score'].shift().rolling(window=n_games, min_periods=1).mean()
        records.append(sub[['GAME_ID', 'TEAM_ID', 'personId', 'rolling_avg']])
    roll = pd.concat(records)

    # Pour chaque match et équipe, sélectionner les top_n par rolling_avg au moment du match
    def pick_top(group):
        return group.nlargest(top_n, 'rolling_avg')['personId']

    tops = roll.groupby(['GAME_ID', 'TEAM_ID']).apply(pick_top).reset_index()
    tops.rename(columns={0: 'personId'}, inplace=True)
    tops['is_historical_top'] = 1

    return tops[['GAME_ID', 'TEAM_ID', 'personId', 'is_historical_top']]


def flag_top_players_absences(df_boxscores: pd.DataFrame,
                              hist_tops: pd.DataFrame) -> pd.DataFrame:
    """
    Marque les absences des top joueurs historiques, en détaillant les causes (blessure, repos, etc.).
    """
    cols = ['GAME_ID', 'TEAM_ID', 'personId', 'is_absent', 'is_injured', 'is_resting', 'is_suspended', 'is_personal']
    df = df_boxscores[cols].copy()
    
    merged = hist_tops.merge(df, on=['GAME_ID', 'TEAM_ID', 'personId'], how='left')
    for col in ['is_absent', 'is_injured', 'is_resting', 'is_suspended', 'is_personal']:
        merged[col] = merged[col].fillna(0).astype(int)

    agg = merged.groupby(['GAME_ID', 'TEAM_ID']).agg(
        top_player_count=('personId', 'nunique'),
        top_player_absent_count=('is_absent', 'sum'),
        top_player_injured=('is_injured', 'sum'),
        top_player_resting=('is_resting', 'sum'),
        top_player_suspended=('is_suspended', 'sum'),
        top_player_personal=('is_personal', 'sum'),
    ).reset_index()

    agg['top_player_absence_rate'] = agg['top_player_absent_count'] / agg['top_player_count']
    agg['top_player_injury_rate'] = agg['top_player_injured'] / agg['top_player_count']
    agg['top_player_resting_rate'] = agg['top_player_resting'] / agg['top_player_count']
    agg['top_player_suspension_rate'] = agg['top_player_suspended'] / agg['top_player_count']
    agg['top_player_personal_rate'] = agg['top_player_personal'] / agg['top_player_count']
    
    return agg
