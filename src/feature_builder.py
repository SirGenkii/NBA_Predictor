import pandas as pd
from typing import Tuple
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
from src.utils import get_team_mapping_id
from datetime import timedelta

import numpy as np
import pandas as pd


def _rolling_shifted_mean(series, group_ids, window):
    """
    Helper: for each group, shift by 1 then compute rolling mean with given window.
    """
    return (
        series.groupby(group_ids)
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

def compute_rolling_features(df, group_col, sort_cols, value_cols, windows, method="mean", apply_log=False):
    """
    Calcule des features de rolling moyenne ou ewm pour une liste de colonnes.
    Si apply_log est True, applique log1p directement sur les colonnes générées, sans les dupliquer.
    
    Args:
        df (pd.DataFrame): Données d'entrée.
        group_col (str): Colonne de groupby (ex: TEAM_ID).
        sort_cols (list): Colonnes de tri (ex: ['TEAM_ID', 'GAME_DATE']).
        value_cols (list): Colonnes à transformer.
        windows (list): Fenêtres de rolling.
        method (str): "mean" ou "ewm".
        apply_log (bool): Si True, applique log1p directement à la colonne générée.
        
    Returns:
        pd.DataFrame enrichi.
    """
    df = df.sort_values(sort_cols).copy()

    for col in value_cols:
        for window in windows:
            roll_col = f"ROLL_{col}_{window}"
            if method == "ewm":
                df[roll_col] = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).ewm(span=window, min_periods=1).mean())
                )
            else:
                df[roll_col] = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

            if apply_log:
                # On remplace directement la colonne par sa version log1p
                df[roll_col] = np.log1p(df[roll_col].clip(lower=0))

    return df



def compute_win_streak(df: pd.DataFrame, group_col: str, win_col: str) -> pd.Series:
    streaks = []
    current_streak = 0
    for win in df[win_col]:
        if win:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)
    return pd.Series(streaks, index=df.index)

def compute_win_ratio(df: pd.DataFrame, group_col: str, win_col: str, windows: list) -> pd.DataFrame:
    df = df.sort_values([group_col, "GAME_DATE"]).copy()
    for n in windows:
        df[f"ROLL_WIN_RATIO_{n}"] = (
            df.groupby(group_col)[win_col]
              .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )
    return df



def streak_grouped_shifted(group, is_home):
    streak = 0
    streaks = []
    for _, row in group.iterrows():
        if row["IS_HOME"] == is_home and row["IS_WIN_SHIFTED"]:
            streak += 1
        elif row["IS_HOME"] == is_home:
            streak = 0
        streaks.append(streak)
    return pd.Series(streaks, index=group.index)


def compute_side_win_streak(df: pd.DataFrame, win_shifted_col: str = "IS_WIN_SHIFTED") -> pd.DataFrame:
    """
    Calcule les streaks de victoire à domicile et à l'extérieur en utilisant une colonne de victoire shiftée.
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    df["HOME_WIN_STREAK"] = df.groupby("TEAM_ID").apply(
        lambda g: streak_grouped_shifted(g, is_home=1)
    ).reset_index(level=0, drop=True)

    df["AWAY_WIN_STREAK"] = df.groupby("TEAM_ID").apply(
        lambda g: streak_grouped_shifted(g, is_home=0)
    ).reset_index(level=0, drop=True)

    return df



def rename_pts_against_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward compatibility helper: copy opponent rolling points columns into the old
    ROLL_[HOME|AWAY]_PTS_AGAINST_* slots, no-op if source columns are absent.
    """
    windows = [3, 5, 10, 25, 50, 100, 200]
    for n in windows:
        home_sources = [
            f"ROLL_HOME_OPP_PTS_AGAINST_{n}",
            f"ROLL_HOME_OPP_points_traditional_AGAINST_{n}",
        ]
        away_sources = [
            f"ROLL_AWAY_OPP_PTS_AGAINST_{n}",
            f"ROLL_AWAY_OPP_points_traditional_AGAINST_{n}",
        ]
        for src in home_sources:
            if src in df.columns:
                df[f"ROLL_HOME_PTS_AGAINST_{n}"] = df[src]
                break
        for src in away_sources:
            if src in df.columns:
                df[f"ROLL_AWAY_PTS_AGAINST_{n}"] = df[src]
                break
    return df


def compute_rest_days(df: pd.DataFrame, date_col: str, group_col: str) -> pd.Series:
    df[date_col] = pd.to_datetime(df[date_col])
    return df.groupby(group_col)[date_col].diff().dt.days.fillna(7)


def compute_elo(df: pd.DataFrame, k: int = 24, start: int = 1500) -> pd.DataFrame:

    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).copy()
    elo_history = defaultdict(lambda: start)
    elos = []

    for _, row in df.iterrows():
        team = row["TEAM_ID"]
        opp = row["OPP_TEAM_ID"]
        game_id = row["GAME_ID"]

        team_elo = elo_history[team]
        opp_elo = elo_history[opp]

        expected = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
        outcome = 1 if row["IS_WIN"] else 0

        new_elo = team_elo + k * (outcome - expected)
        elo_history[team] = new_elo

        elos.append({"GAME_ID": game_id, "TEAM_ID": team, "ELO_PRE": team_elo})

    elo_df = pd.DataFrame(elos)
    df = df.merge(elo_df, on=["GAME_ID", "TEAM_ID"], how="left")

    # Ajout propre de OPP_ELO_PRE
    opp_elo_df = elo_df.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "ELO_PRE": "OPP_ELO_PRE"})
    df = df.merge(opp_elo_df, on=["GAME_ID", "OPP_TEAM_ID"], how="left")

    return df



def compute_elo_season(df: pd.DataFrame, k: int = 24, start: int = 1500) -> pd.DataFrame:

    df = df.sort_values(["SEASON", "GAME_DATE", "GAME_ID", "TEAM_ID"]).copy()
    elo_history = defaultdict(lambda: start)
    elos = []

    for _, row in df.iterrows():
        season = row["SEASON"]
        team = row["TEAM_ID"]
        opp = row["OPP_TEAM_ID"]
        game_id = row["GAME_ID"]

        team_elo = elo_history[(season, team)]
        opp_elo = elo_history[(season, opp)]

        expected = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
        outcome = 1 if row["IS_WIN"] else 0

        new_elo = team_elo + k * (outcome - expected)
        elo_history[(season, team)] = new_elo

        elos.append({"GAME_ID": game_id, "TEAM_ID": team, "ELO_PRE_SEASON": team_elo})

    elo_df = pd.DataFrame(elos)
    df = df.merge(elo_df, on=["GAME_ID", "TEAM_ID"], how="left")

    # Ajout propre de OPP_ELO_PRE_SEASON
    opp_elo_df = elo_df.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "ELO_PRE_SEASON": "OPP_ELO_PRE_SEASON"})
    df = df.merge(opp_elo_df, on=["GAME_ID", "OPP_TEAM_ID"], how="left")

    return df


def convert_elos_to_elo_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes ELO_PRE et OPP_ELO_PRE en une seule colonne ELO_DIFF.
    """
    
    df = df.copy()
    
    df["ELO_DIFF"] = df["ELO_PRE"] - df["OPP_ELO_PRE"]
    df["ELO_DIFF_SEASON"] = df["ELO_PRE_SEASON"] - df["OPP_ELO_PRE_SEASON"]
    
    # Supprimer les colonnes originales
    df.drop(columns=["ELO_PRE", "OPP_ELO_PRE", "ELO_PRE_SEASON", "OPP_ELO_PRE_SEASON"], inplace=True, errors='ignore')
    
    return df


def compute_winrates(df: pd.DataFrame, group_col: str, win_col: str, home_col: str, windows: list) -> pd.DataFrame:
    for n in windows:
        df[f"ROLL_HOME_WINRATE_{n}"] = (
            df.sort_values([group_col, "GAME_DATE"])
              .groupby(group_col)
              .apply(lambda d: d[win_col].where(d[home_col] == 1).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
               .fillna(-1.0)
        )
        df[f"ROLL_AWAY_WINRATE_{n}"] = (
            df.sort_values([group_col, "GAME_DATE"])
              .groupby(group_col)
              .apply(lambda d: d[win_col].where(d[home_col] == 0).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
               .fillna(-1.0)
        )
    return df


def compute_h2h(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    from collections import defaultdict, deque

    df = df.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    for n in windows:
        results = defaultdict(lambda: deque(maxlen=n))
        diffs, winrates, counts = [], [], []

        for _, row in df.iterrows():
            team, opp = row["TEAM_ID"], row["OPP_TEAM_ID"]
            res = 1 if row["IS_WIN"] else 0

            hist = results[(team, opp)]
            count = len(hist)
            winrate = sum(hist)/count if count else 0.5
            diff = sum(hist) - (count - sum(hist)) if count else 0

            diffs.append(diff)
            winrates.append(winrate)
            counts.append(count)

            results[(team, opp)].append(res)

        df[f"H2H_LAST_{n}_DIFF"] = diffs
        df[f"H2H_LAST_{n}_WINRATE"] = winrates
        df[f"H2H_LAST_{n}_COUNT"] = counts

        #fill NaN values with 0
        df[f"H2H_LAST_{n}_DIFF"] = df[f"H2H_LAST_{n}_DIFF"].fillna(0)
        df[f"H2H_LAST_{n}_WINRATE"] = df[f"H2H_LAST_{n}_WINRATE"].fillna(0.5)
        df[f"H2H_LAST_{n}_COUNT"] = df[f"H2H_LAST_{n}_COUNT"].fillna(0)

    return df


def compute_home_away_pts(df: pd.DataFrame, group_col: str, is_home_col: str, pts_col: str, opp_pts_col: str, windows: list) -> pd.DataFrame:
    df = df.sort_values([group_col, "GAME_DATE"]).copy()
    for n in windows:
        home_pts = df[pts_col].where(df[is_home_col] == 1)
        home_opp_pts = df[opp_pts_col].where(df[is_home_col] == 1)
        away_pts = df[pts_col].where(df[is_home_col] == 0)
        away_opp_pts = df[opp_pts_col].where(df[is_home_col] == 0)

        df[f"ROLL_HOME_{pts_col}_FOR_{n}"] = _rolling_shifted_mean(home_pts, df[group_col], n)
        df[f"ROLL_HOME_{opp_pts_col}_AGAINST_{n}"] = _rolling_shifted_mean(home_opp_pts, df[group_col], n)
        df[f"ROLL_AWAY_{pts_col}_FOR_{n}"] = _rolling_shifted_mean(away_pts, df[group_col], n)
        df[f"ROLL_AWAY_{opp_pts_col}_AGAINST_{n}"] = _rolling_shifted_mean(away_opp_pts, df[group_col], n)
        
        #fill NaN values with -1
        df[f"ROLL_HOME_{pts_col}_FOR_{n}"] = df[f"ROLL_HOME_{pts_col}_FOR_{n}"].fillna(-1)
        df[f"ROLL_HOME_{opp_pts_col}_AGAINST_{n}"] = df[f"ROLL_HOME_{opp_pts_col}_AGAINST_{n}"].fillna(-1)
        df[f"ROLL_AWAY_{pts_col}_FOR_{n}"] = df[f"ROLL_AWAY_{pts_col}_FOR_{n}"].fillna(-1)
        df[f"ROLL_AWAY_{opp_pts_col}_AGAINST_{n}"] = df[f"ROLL_AWAY_{opp_pts_col}_AGAINST_{n}"].fillna(-1)
        
        
    return df


    

def compute_h2h_pts_margin(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    for n in windows:
        pts_for_hist = defaultdict(lambda: deque(maxlen=n))
        pts_against_hist = defaultdict(lambda: deque(maxlen=n))
        pts_for, pts_against, margins = [], [], []

        for _, row in df.iterrows():
            key = (row["TEAM_ID"], row["OPP_TEAM_ID"])
            history_for = pts_for_hist[key]
            history_against = pts_against_hist[key]
            count = len(history_for)
            avg_for = sum(history_for) / count if count else 0
            avg_against = sum(history_against) / count if count else 0
            avg_margin = (sum(history_for) - sum(history_against)) / count if count else 0
            pts_for.append(avg_for)
            pts_against.append(avg_against)
            margins.append(avg_margin)
            pts_for_hist[key].append(row["points_traditional"])
            pts_against_hist[key].append(row["OPP_points_traditional"])

        df[f"H2H_LAST_{n}_PTS_FOR"] = pts_for
        df[f"H2H_LAST_{n}_PTS_AGAINST"] = pts_against
        df[f"H2H_LAST_{n}_MARGIN"] = margins
        
        #fill NaN values with 0
        df[f"H2H_LAST_{n}_PTS_FOR"] = df[f"H2H_LAST_{n}_PTS_FOR"].fillna(0)
        df[f"H2H_LAST_{n}_PTS_AGAINST"] = df[f"H2H_LAST_{n}_PTS_AGAINST"].fillna(0)
        df[f"H2H_LAST_{n}_MARGIN"] = df[f"H2H_LAST_{n}_MARGIN"].fillna(0)
        

    return df

def compute_h2h_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    win_count = defaultdict(int)
    match_count = defaultdict(int)
    season_win_counts, season_match_counts = [], []

    for _, row in df.iterrows():
        key = (row["SEASON"], row["TEAM_ID"], row["OPP_TEAM_ID"])
        season_win_counts.append(win_count[key])
        season_match_counts.append(match_count[key])
        match_count[key] += 1
        if row["IS_WIN"]:
            win_count[key] += 1

    df["H2H_SEASON_WINS"] = season_win_counts
    df["H2H_SEASON_MATCHES"] = season_match_counts
    df["H2H_SEASON_WINRATE"] = [w / m if m > 0 else 0 for w, m in zip(season_win_counts, season_match_counts)]
    return df

def compute_h2h_streak(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    last_result = defaultdict(lambda: None)
    current_streak = defaultdict(int)
    streaks = []

    for _, row in df.iterrows():
        key = (row["TEAM_ID"], row["OPP_TEAM_ID"])
        streak = current_streak[key]
        last = last_result[key]
        streaks.append(streak)
        if row["IS_WIN"]:
            current_streak[key] = streak + 1 if last == "W" else 1
            last_result[key] = "W"
        else:
            current_streak[key] = 0
            last_result[key] = "L"

    df["H2H_WIN_STREAK"] = streaks
    return df

def compute_rolling_rest_advantage(df: pd.DataFrame, group_col: str, is_home_col: str, rest_col: str, windows: list) -> pd.DataFrame:
    df = df.sort_values(["GAME_DATE"]).copy()
    for n in windows:
        home_rest = df[rest_col].where(df[is_home_col] == 1)
        away_rest = df[rest_col].where(df[is_home_col] == 0)
        df[f"ROLL_HOME_REST_ADV_{n}"] = _rolling_shifted_mean(home_rest, df[group_col], n).fillna(0)
        df[f"ROLL_AWAY_REST_ADV_{n}"] = _rolling_shifted_mean(away_rest, df[group_col], n).fillna(0)
    return df


def add_advanced_boxscore_features(df):
    """
    Ajoute des features avancées dérivées des boxscores classiques.
    Nécessite les colonnes suivantes : FGA, FGM, FG3M, FTA, PTS, AST, TO, REB, OPP_REB
    """

    # True Shooting Percentage (TS%)
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']).replace(0, np.nan))

    # Effective Field Goal Percentage (eFG%)
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)

    # Assist to Turnover Ratio (AST/TO)
    df['AST_TO_RATIO'] = df['AST'] / df['TO'].replace(0, np.nan)

    # Rebound Rate (approx.)
    df['REB_RATE'] = df['REB'] / (df['REB'] + df['OPP_REB']).replace(0, np.nan)

    # Estimation des possessions (offensives) : FGA + 0.44*FTA + TO - OREB
    df["POSSESSIONS"] = df["FGA"] + 0.44 * df["FTA"] + df["TO"] - df["OREB"]
    df["OPP_POSSESSIONS"] = df["OPP_FGA"] + 0.44 * df["OPP_FTA"] + df["OPP_TO"] - df["OPP_OREB"]

    # Offensive et Defensive Rating (pts pour ou contre pour 100 possessions)
    df["OFF_RATING"] = 100 * df["PTS"] / df["POSSESSIONS"].replace(0, np.nan)
    df["DEF_RATING"] = 100 * df["OPP_PTS"] / df["OPP_POSSESSIONS"].replace(0, np.nan)


    # Sanitize NaNs and Infs
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(0, inplace=True)

    return df


def clean_team_name(name):
    return name.strip().lower() if isinstance(name, str) else name


def match_odds_with_dataset_test(odds_df, nba_df):
    
    team_id_map = get_team_mapping_id()
    
    nba_df = nba_df.copy()
    odds_df = odds_df.copy()

    # Conversion explicite des ID en str
    nba_df["TEAM_ID"] = nba_df["TEAM_ID"].astype(str)
    nba_df["OPP_TEAM_ID"] = nba_df["OPP_TEAM_ID"].astype(str)
    team_id_map = {str(k): v for k, v in team_id_map.items()}

    nba_df["TEAM_NAME"] = nba_df["TEAM_ID"].map(team_id_map).apply(clean_team_name)
    nba_df["OPPONENT_NAME"] = nba_df["OPP_TEAM_ID"].map(team_id_map).apply(clean_team_name)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"]).dt.date

    odds_df["home_team"] = odds_df["home_team"].apply(clean_team_name)
    odds_df["away_team"] = odds_df["away_team"].apply(clean_team_name)
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.date

    nba_df["match_key"] = nba_df.apply(
        lambda row: (row["GAME_DATE"], row["TEAM_NAME"], row["OPPONENT_NAME"])
        if row["IS_HOME"] == 1
        else (row["GAME_DATE"], row["OPPONENT_NAME"], row["TEAM_NAME"]),
        axis=1,
    )

    keys_full = []
    for shift in [-1, 0, 1]:
        shifted = odds_df.copy()
        shifted["match_key"] = shifted.apply(
            lambda row: (row["date"] + timedelta(days=shift), row["home_team"], row["away_team"]),
            axis=1
        )
        keys_full.append(shifted)

    odds_full = pd.concat(keys_full, ignore_index=True)
    odds_full = odds_full.drop_duplicates(subset=["match_key"])

    # print("\nExemples de clés de match dans odds_df (tolérance date):")
    # print(odds_full["match_key"].drop_duplicates().head())
    # print("\nExemples de clés de match dans nba_df:")
    # print(nba_df["match_key"].drop_duplicates().head())


    merged = pd.merge(odds_full, nba_df, on="match_key", how="right")
    
    # Attribution claire des cotes à chaque ligne équipe
    merged["ODDS"] = merged.apply(
        lambda row: row["home_odds"] if row["IS_HOME"] == 1 else row["away_odds"], axis=1
    )
    merged["OPP_ODDS"] = merged.apply(
        lambda row: row["away_odds"] if row["IS_HOME"] == 1 else row["home_odds"], axis=1
    )
    
    # Nettoyage des données fusionnées
    merged = clean_merged_matches(merged)

    return merged



def clean_merged_matches(df):
    df = df.copy()
    df = df[df['TEAM_ID'].notna() & df['OPP_TEAM_ID'].notna()]
    df = df.drop_duplicates(subset=["match_key", "TEAM_ID"])
    
    # Compter les lignes sans odds AVANT de supprimer les colonnes
    if 'home_odds' in df.columns and 'away_odds' in df.columns:
        odds_na_count = df[df['home_odds'].isna() | df['away_odds'].isna()].shape[0]
        print(f"------------------ Nombre de lignes sans cotes (home/away): {odds_na_count} ------------------")
    else:
        # Sinon, compter sur les colonnes ODDS/OPP_ODDS
        odds_na_count = df[df['ODDS'].isna() | df['OPP_ODDS'].isna()].shape[0]
        print(f"------------------ Nombre de lignes sans cotes (ODDS/OPP_ODDS): {odds_na_count} ------------------")

    # Suppression des colonnes originales de odds
    df.drop(columns=["date","match_key","home_team", "away_team", "home_odds", "away_odds", "home_score", "away_score"], inplace=True, errors='ignore')


    return df.reset_index(drop=True)

def match_odds_with_dataset(odds_df, nba_df):
    
    team_id_map = get_team_mapping_id()
    
    nba_df = nba_df.copy()
    odds_df = odds_df.copy()

    # Conversion explicite des ID en str
    nba_df["TEAM_ID"] = nba_df["TEAM_ID"].astype(str)
    nba_df["OPP_TEAM_ID"] = nba_df["OPP_TEAM_ID"].astype(str)
    team_id_map = {str(k): v for k, v in team_id_map.items()}

    nba_df["TEAM_NAME"] = nba_df["TEAM_ID"].map(team_id_map).apply(clean_team_name)
    nba_df["OPPONENT_NAME"] = nba_df["OPP_TEAM_ID"].map(team_id_map).apply(clean_team_name)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"]).dt.date

    odds_df["home_team"] = odds_df["home_team"].apply(clean_team_name)
    odds_df["away_team"] = odds_df["away_team"].apply(clean_team_name)
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.date

    nba_df["match_key"] = nba_df.apply(
        lambda row: (row["GAME_DATE"], row["TEAM_NAME"], row["OPPONENT_NAME"])
        if row["IS_HOME"] == 1
        else (row["GAME_DATE"], row["OPPONENT_NAME"], row["TEAM_NAME"]),
        axis=1,
    )

    keys_full = []
    for shift in [-1, 0, 1]:
        shifted = odds_df.copy()
        shifted["match_key"] = shifted.apply(
            lambda row: (row["date"] + timedelta(days=shift), row["home_team"], row["away_team"]),
            axis=1
        )
        keys_full.append(shifted)

    odds_full = pd.concat(keys_full, ignore_index=True)
    odds_full = odds_full.drop_duplicates(subset=["match_key"])

    # print("\nExemples de clés de match dans odds_df (tolérance date):")
    # print(odds_full["match_key"].drop_duplicates().head())
    # print("\nExemples de clés de match dans nba_df:")
    # print(nba_df["match_key"].drop_duplicates().head())


    merged = pd.merge(odds_full, nba_df, on="match_key", how="right")
    
    # Attribution claire des cotes à chaque ligne équipe
    merged["ODDS"] = merged.apply(
        lambda row: row["home_odds"] if row["IS_HOME"] == 1 else row["away_odds"], axis=1
    )
    merged["OPP_ODDS"] = merged.apply(
        lambda row: row["away_odds"] if row["IS_HOME"] == 1 else row["home_odds"], axis=1
    )
    
    # print(f"\nNombre de lignes fusionnées: {len(merged)}")
    # print(f"Nombre de correspondances réussies: {merged['TEAM_ID'].notna().sum()}")
    # print(f"Nombre de correspondances échouées: {merged['TEAM_ID'].isna().sum()}")

    # print("\nIDs manquants TEAM_ID:", nba_df[~nba_df["TEAM_ID"].isin(team_id_map.keys())]["TEAM_ID"].unique())
    # print("IDs manquants OPP_TEAM_ID:", nba_df[~nba_df["OPP_TEAM_ID"].isin(team_id_map.keys())]["OPP_TEAM_ID"].unique())

    # Nettoyage des données fusionnées
    merged = clean_merged_matches(merged)

    return merged



def build_player_status_features(df_boxscore: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des indicateurs de présence, absence, blessure et performance simple pour chaque joueur.
    Attend une colonne MINUTES_PLAYED en décimal.
    """
    df = df_boxscore.copy()

    df["is_present"] = df["MINUTES_PLAYED"] > 0

    comment = df["comment"].fillna("").str.lower()

    # Tags pour blessures
    injury_keywords = [
        "injury", "illness", "sore", "sprain", "fracture", "strain", "pain", "contusion", "concussion", 
        "discomfort", "tendon", "inflammation", "rehab", "recover", "migraine", "toe", "back", "knee", "ligament", "finger","stomach",
        "ankle", "hamstring", "groin", "shoulder", "wrist", "gastric", "conditioning", "reconditioning", "migraine","bruise","foot","leg",
        "foot", "turf", "flu", "health", "headache", "stomacjh", "virus", "covid", "gastro", "gastroenteritis", "poisoning", "fasciitis",
        "ankle","bruised","infection", "torn", "rupture", "tendinitis", "tendinopathy", "tendinosis", "tendonitis", "tendinopathy",
        "bronchitis","respiratory", "respiratory illness", "respiratory infection", "respiratory distress", "respiratory condition", "respiratory issue","strained",
        
    ]
    rest_keywords = ["rest", "load management", "reconditioning"]
    suspension_keywords = ["suspension", "suspended", "suspend","league suspension"]

    personal_keywords = ["personal", "paternity", "birth", "family", "not with team", "excused"]
    
    df["is_present"] = df["MINUTES_PLAYED"] > 0
    comment = df["comment"].fillna("").str.lower()
    df["comment"] = comment

    df["is_absent"] = ~df["is_present"]

    df["is_suspended"] = (~df["is_present"]) & comment.apply(lambda x: any(k in x for k in suspension_keywords))
    df["is_injured"] = (~df["is_present"]) & ~df["is_suspended"] & comment.apply(lambda x: any(k in x for k in injury_keywords))
    df["is_resting"] = (~df["is_present"]) & ~df["is_suspended"] & ~df["is_injured"] & comment.apply(lambda x: any(k in x for k in rest_keywords))
    df["is_personal"] = (~df["is_present"]) & ~df["is_suspended"] & ~df["is_injured"] & ~df["is_resting"] & comment.apply(lambda x: any(k in x for k in personal_keywords))
    df["is_absent_other"] = (~df["is_present"]) & ~(df["is_injured"] | df["is_resting"] | df["is_suspended"] | df["is_personal"])



    #convert all booleans in df to int 
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    

    # Score simple de performance avec pondération (à ajuster ou améliorer avec SHAP )
    df['player_perf_score'] = (
        1.0 * df['points_traditional'].fillna(0) +
        1.5 * df['assists_traditional'].fillna(0) +
        1.2 * df['reboundsTotal_traditional'].fillna(0) +
        1.0 * df['steals_traditional'].fillna(0) +
        1.0 * df['blocks_traditional'].fillna(0) -
        1.0 * df['turnovers_traditional'].fillna(0)
    )
    
    df['player_defense_score'] = (
        1.5 * df['steals_traditional'].fillna(0) +
        1.5 * df['blocks_traditional'].fillna(0) -
        1.0 * df['foulsPersonal_traditional'].fillna(0)
    )

    df['player_offense_score'] = (
        1.2 * df['points_traditional'].fillna(0) +
        1.5 * df['assists_traditional'].fillna(0) -
        1.0 * df['turnovers_traditional'].fillna(0)
    )
    
    df['player_impact_score'] = df['PIE_advanced'].fillna(0)



    return df[[
        'GAME_ID', 'TEAM_ID', 'GAME_DATE', 'personId', 
        'is_present', 'is_absent', 'is_injured', 'is_resting', 'is_suspended', 'is_personal', 'is_absent_other', 
        'player_perf_score', 'player_defense_score', 'player_offense_score', 
        'player_impact_score',
        'comment'
        ]].copy()
