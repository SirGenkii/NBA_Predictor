import pandas as pd
from typing import Tuple


def compute_rolling_features(df: pd.DataFrame, group_col: str, sort_cols: list, value_cols: list, windows: list) -> pd.DataFrame:
    df = df.sort_values(sort_cols).copy()
    for col in value_cols:
        for window in windows:
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


def compute_rest_days(df: pd.DataFrame, date_col: str, group_col: str) -> pd.Series:
    df[date_col] = pd.to_datetime(df[date_col])
    return df.groupby(group_col)[date_col].diff().dt.days.fillna(7)


def compute_elo(df: pd.DataFrame, k: int = 24, start: int = 1500) -> pd.DataFrame:
    from collections import defaultdict

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
    return df.merge(elo_df, on=["GAME_ID", "TEAM_ID"], how="left")


def compute_winrates(df: pd.DataFrame, group_col: str, win_col: str, home_col: str, windows: list) -> pd.DataFrame:
    for n in windows:
        df[f"ROLL_HOME_WINRATE_{n}"] = (
            df.sort_values([group_col, "GAME_DATE"])
              .groupby(group_col)
              .apply(lambda d: d[win_col].where(d[home_col] == 1).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
        df[f"ROLL_AWAY_WINRATE_{n}"] = (
            df.sort_values([group_col, "GAME_DATE"])
              .groupby(group_col)
              .apply(lambda d: d[win_col].where(d[home_col] == 0).shift(1).rolling(n, min_periods=1).mean())
              .reset_index(level=0, drop=True)
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
    return df
