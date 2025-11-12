from __future__ import annotations

from typing import Collection, Iterable, List, Optional, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import MATCH_IDENTIFIER_COLUMNS, MATCH_SIDE_PREFIXES


def team_rows_to_match_rows(
    team_df: pd.DataFrame,
    *,
    match_keys: Sequence[str] = MATCH_IDENTIFIER_COLUMNS,
) -> pd.DataFrame:
    """
    Convert the two-row-per-game team dataset (with IS_HOME) into a single row
    per match with HOME_/AWAY_/DIFF_ columns.
    """

    if "IS_HOME" not in team_df.columns:
        raise ValueError("Expected an 'IS_HOME' column in team-level dataset.")

    cleaned = _drop_opponent_columns(team_df)
    home = cleaned[cleaned["IS_HOME"] == 1].copy()
    away = cleaned[cleaned["IS_HOME"] == 0].copy()

    home = _deduplicate_side(home, "home", match_keys)
    away = _deduplicate_side(away, "away", match_keys)

    if home.empty or away.empty:
        raise ValueError("Dataset must contain both home and away rows per game.")

    for side_df, label in ((home, "home"), (away, "away")):
        if side_df.duplicated(subset=list(match_keys)).any():
            dup = side_df[side_df.duplicated(subset=list(match_keys), keep=False)]
            raise ValueError(f"Duplicate {label} rows detected for match keys:\n{dup.head()}")

    home = _prefix_side(home, "HOME", match_keys)
    away = _prefix_side(away, "AWAY", match_keys)

    match_df = home.merge(
        away,
        on=list(match_keys),
        how="inner",
        validate="one_to_one",
        suffixes=("", "_AWAYDROP"),
    )

    match_df["IS_WIN"] = match_df["HOME_IS_WIN"]
    match_df["POINT_DIFF"] = match_df["HOME_POINTS_FOR"] - match_df["AWAY_POINTS_FOR"]
    match_df["POINT_TOTAL"] = match_df["HOME_POINTS_FOR"] + match_df["AWAY_POINTS_FOR"]

    match_df = add_diff_columns(match_df)
    match_df = match_df.sort_values(list(match_keys)).reset_index(drop=True)
    return match_df


def match_rows_to_team_rows(
    match_df: pd.DataFrame,
    *,
    include_cols: Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    Explode the single-row-per-match dataframe back into a team-level view so we
    can reuse historical feature functions that expect TEAM_ID / OPP_TEAM_ID rows.
    """

    frames: List[pd.DataFrame] = []
    include = set(include_cols or [])

    for side in MATCH_SIDE_PREFIXES:
        opp_side = "AWAY" if side == "HOME" else "HOME"
        prefix = f"{side}_"
        opp_prefix = f"{opp_side}_"

        base = match_df[list(MATCH_IDENTIFIER_COLUMNS)].copy()
        base["TEAM_ID"] = match_df[f"{prefix}TEAM_ID"]
        base["OPP_TEAM_ID"] = match_df[f"{opp_prefix}TEAM_ID"]
        base["IS_HOME"] = 1 if side == "HOME" else 0
        base["IS_WIN"] = match_df.get(f"{prefix}IS_WIN", match_df["IS_WIN"])

        source_columns = include or _infer_feature_columns(match_df, prefix)
        for col in source_columns:
            src = f"{prefix}{col}"
            if src in match_df:
                base[col] = match_df[src]
            opp_src = f"{opp_prefix}{col}"
            if opp_src in match_df:
                base[f"OPP_{col}"] = match_df[opp_src]

        frames.append(base)

    team_df = pd.concat(frames, ignore_index=True)
    return team_df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def attach_team_features(
    match_df: pd.DataFrame,
    team_features: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Attach columns computed on the team-level view back onto the match dataframe
    using HOME_/AWAY_ prefixes.
    """

    cols = list(feature_cols)
    if not cols:
        return match_df

    enriched = match_df.copy()
    for side in MATCH_SIDE_PREFIXES:
        side_df = team_features[team_features["IS_HOME"] == (1 if side == "HOME" else 0)]
        side_df = side_df[["GAME_ID", "TEAM_ID"] + cols].copy()
        rename_map = {"TEAM_ID": f"{side}_TEAM_ID"}
        rename_map.update({col: f"{side}_{col}" for col in cols})
        side_df = side_df.rename(columns=rename_map)
        enriched = enriched.merge(
            side_df,
            on=["GAME_ID", f"{side}_TEAM_ID"],
            how="left",
        )
    enriched = add_diff_columns(enriched, limit_to=cols)
    return enriched


def add_diff_columns(
    match_df: pd.DataFrame,
    *,
    limit_to: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Ensure DIFF_* columns exist for every HOME_/AWAY_ numeric pair, optionally
    restricted to a subset of base column names.
    """

    df = match_df.copy()
    bases = limit_to if limit_to is not None else _derive_diff_bases(df.columns)

    for base in bases:
        home_col = f"HOME_{base}"
        away_col = f"AWAY_{base}"
        diff_col = f"DIFF_{base}"
        if home_col in df.columns and away_col in df.columns:
            if not (is_numeric_dtype(df[home_col]) and is_numeric_dtype(df[away_col])):
                continue
            df[diff_col] = df[home_col] - df[away_col]

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_opponent_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if col.startswith("OPP_")]
    return df.drop(columns=cols, errors="ignore")


def _deduplicate_side(df: pd.DataFrame, label: str, match_keys: Sequence[str]) -> pd.DataFrame:
    duplicated_mask = df.duplicated(subset=list(match_keys), keep=False)
    if duplicated_mask.any():
        dup_count = duplicated_mask.sum()
        print(f"[reshaping] dropping {dup_count} duplicate {label} rows based on match keys {match_keys}.")
        df = df.sort_values(list(match_keys)).drop_duplicates(subset=list(match_keys), keep="first")
    return df


def _prefix_side(df: pd.DataFrame, side: str, match_keys: Sequence[str]) -> pd.DataFrame:
    prefix = f"{side}_"
    base = df.copy()
    base = base.drop(columns=["IS_HOME"], errors="ignore")

    rename_map = {}
    for col in base.columns:
        if col in match_keys:
            continue
        rename_map[col] = f"{prefix}{col}"

    base = base.rename(columns=rename_map)
    return base


def _infer_feature_columns(match_df: pd.DataFrame, prefix: str) -> List[str]:
    cols = []
    for col in match_df.columns:
        if col.startswith(prefix):
            cols.append(col[len(prefix) :])
    return cols


def _derive_diff_bases(columns: Iterable[str]) -> List[str]:
    bases = []
    for col in columns:
        if col.startswith("HOME_"):
            bases.append(col[len("HOME_") :])
    return bases
