import pandas as pd
from typing import Tuple
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
from src.utils import get_team_mapping_id
from datetime import timedelta

def compute_rolling_features(df, group_col, sort_cols, value_cols, windows, method="mean"):
    df = df.sort_values(sort_cols).copy()
    for col in value_cols:
        for window in windows:
            if method == "ewm":
                df[f"ROLL_{col}_{window}"] = (
                    df.groupby(group_col)[col]
                      .transform(lambda x: x.shift(1).ewm(span=window, min_periods=1).mean())
                )
            else:
                df[f"ROLL_{col}_{window}"] = (
                    df.groupby(group_col)[col]
                      .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
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


def compute_side_win_streak(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    df["IS_WIN_SHIFTED"] = df.groupby("TEAM_ID")["IS_WIN"].shift(1).fillna(0).astype(int)

    df["HOME_WIN_STREAK"] = df.groupby("TEAM_ID").apply(lambda g: streak_grouped_shifted(g, is_home=1)).reset_index(level=0, drop=True)
    df["AWAY_WIN_STREAK"] = df.groupby("TEAM_ID").apply(lambda g: streak_grouped_shifted(g, is_home=0)).reset_index(level=0, drop=True)

    return df.drop(columns=["IS_WIN_SHIFTED"])


def rename_pts_against_columns(df: pd.DataFrame) -> pd.DataFrame:
    for n in [3, 5, 10, 25, 50, 100, 200]:
        df[f"ROLL_HOME_PTS_AGAINST_{n}"] = df[f"ROLL_HOME_OPP_PTS_AGAINST_{n}"]
        df[f"ROLL_AWAY_PTS_AGAINST_{n}"] = df[f"ROLL_AWAY_OPP_PTS_AGAINST_{n}"]
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
        df[f"ROLL_HOME_{pts_col}_FOR_{n}"] = (
            df.groupby(group_col)
              .apply(lambda d: d[pts_col].where(d[is_home_col] == 1).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
        df[f"ROLL_HOME_{opp_pts_col}_AGAINST_{n}"] = (
            df.groupby(group_col)
              .apply(lambda d: d[opp_pts_col].where(d[is_home_col] == 1).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
        df[f"ROLL_AWAY_{pts_col}_FOR_{n}"] = (
            df.groupby(group_col)
              .apply(lambda d: d[pts_col].where(d[is_home_col] == 0).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
        df[f"ROLL_AWAY_{opp_pts_col}_AGAINST_{n}"] = (
            df.groupby(group_col)
              .apply(lambda d: d[opp_pts_col].where(d[is_home_col] == 0).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
        
        #fill NaN values with -1
        df[f"ROLL_HOME_{pts_col}_FOR_{n}"] = df[f"ROLL_HOME_{pts_col}_FOR_{n}"].fillna(-1)
        df[f"ROLL_HOME_{opp_pts_col}_AGAINST_{n}"] = df[f"ROLL_HOME_{opp_pts_col}_AGAINST_{n}"].fillna(-1)
        df[f"ROLL_AWAY_{pts_col}_FOR_{n}"] = df[f"ROLL_AWAY_{pts_col}_FOR_{n}"].fillna(-1)
        df[f"ROLL_AWAY_{opp_pts_col}_AGAINST_{n}"] = df[f"ROLL_AWAY_{opp_pts_col}_AGAINST_{n}"].fillna(-1)
        
        
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
            pts_for_hist[key].append(row["PTS"])
            pts_against_hist[key].append(row["OPP_PTS"])

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
        df[f"ROLL_HOME_REST_ADV_{n}"] = (
            df.groupby(group_col)
              .apply(lambda d: d[rest_col].where(d[is_home_col] == 1).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
              .fillna(0)
        )
        df[f"ROLL_AWAY_REST_ADV_{n}"] = (
            df.groupby(group_col)
              .apply(lambda d: d[rest_col].where(d[is_home_col] == 0).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
              .fillna(0)
        )
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

def clean_merged_matches(df):
    df = df.copy()
    df = df[df['TEAM_ID'].notna() & df['OPP_TEAM_ID'].notna()]
    df = df.drop_duplicates(subset=["match_key", "TEAM_ID"])
    
    # Suppression des colonnes originales de odds
    df.drop(columns=["date","match_key","home_team", "away_team", "home_odds", "away_odds", "home_score", "away_score"], inplace=True, errors='ignore')
    
    #df = df.dropna(subset=["home_odds", "away_odds"])
    
    #drop nan for "home_odds", "away_odds" and print the number of rows removed
    initial_rows = len(df)
    df = df.dropna(subset=["ODDS", "OPP_ODDS"])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"------------------ Nombre de lignes supprimées pour cotes manquantes: {removed_rows} ------------------")
    
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


    merged = pd.merge(odds_full, nba_df, on="match_key", how="left")
    
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
