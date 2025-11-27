from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import (
    BASE_TEAM_FEATURE_COLUMNS,
    DATA_BRONZE_BOXSCORES_DIR,
    DATA_BRONZE_GAMES_DIR,
    DATA_BRONZE_MATCHES_DIR,
    DATA_ODDS_HISTORY_DIR,
    MATCH_CONTEXT_FLAGS,
    MATCH_IDENTIFIER_COLUMNS,
    cols_player_stats,
    cols_to_sum,
    cols_to_weighted_avg,
)
from src.datasets.base import BuildResult, save_dataset, timestamped_filename
from src.feature_aggregation import (
    aggregate_actual_team_features,
    compute_weighted_mean_features,
    flag_top_players_absences,
    identify_historical_top_players,
)
from src.feature_builder import build_player_status_features, match_odds_with_dataset
from src.features.reshaping import team_rows_to_match_rows
from src.utils import get_latest_file, merge_odds_csv_files


def _latest_csv(directory: str) -> Path:
    return Path(get_latest_file(directory))


def _resolve_path(path: Optional[Path], default_dir: str) -> Path:
    if path is not None:
        return Path(path)
    return _latest_csv(default_dir)


def _parse_minutes(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if ":" in text:
        mins, secs = text.split(":")
        try:
            return int(mins) + int(secs) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _select_existing(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [col for col in candidates if col in df.columns]


def load_games_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = _resolve_path(path, DATA_BRONZE_GAMES_DIR)
    df = pd.read_csv(csv_path)
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    # TEAM_ID can come in as float (e.g., 1610612738.0); normalize to integer-like strings.
    df["TEAM_ID"] = (
        pd.to_numeric(df["TEAM_ID"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = df["SEASON"].astype(str)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs").astype(int)
    df["IS_WIN"] = df["WL"].str.upper().eq("W").astype(int)
    df["POINTS_FOR"] = df["PTS"]
    df["TEAM_NAME"] = df["TEAM_NAME"].str.strip()
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].str.strip()
    return df


def load_boxscores_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = _resolve_path(path, DATA_BRONZE_BOXSCORES_DIR)
    df = pd.read_csv(csv_path)
    rename_map = {
        "gameId": "GAME_ID",
        "teamId": "TEAM_ID",
        "personId": "personId",
        "minutes": "MINUTES_RAW",
    }
    df = df.rename(columns=rename_map)
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    df["MINUTES_PLAYED"] = df["MINUTES_RAW"].apply(_parse_minutes)
    return df


def _prepare_games_meta(games_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "GAME_ID",
        "TEAM_ID",
        "GAME_DATE",
        "SEASON",
        "MATCHUP",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "IS_HOME",
        "IS_WIN",
        "POINTS_FOR",
    ]
    base = games_df[meta_cols].copy()

    opp_cols = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "POINTS_FOR"]
    opp = base[opp_cols].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "TEAM_NAME": "OPPONENT_NAME",
            "TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION",
            "POINTS_FOR": "OPP_PTS",
        }
    )
    merged = base.merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]]
    merged = merged.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    merged["POINTS_AGAINST"] = merged["OPP_PTS"]
    merged["POINT_DIFF"] = merged["POINTS_FOR"] - merged["POINTS_AGAINST"]
    merged["POINT_TOTAL"] = merged["POINTS_FOR"] + merged["POINTS_AGAINST"]
    merged["OPP_GAME_DATE"] = merged["GAME_DATE"]
    merged["IS_PLAYOFF"] = merged["GAME_ID"].astype(str).str.startswith("004").astype(int)
    merged["IS_IN_SEASON_TOURNAMENT"] = (
        ((merged["GAME_DATE"].dt.month == 11) | (merged["GAME_DATE"].dt.month == 12))
        & (merged["GAME_DATE"].dt.year >= 2023)
    ).astype(int)
    merged["IS_FINAL_WEEK"] = (
        ((merged["GAME_DATE"].dt.month == 4) & (merged["GAME_DATE"].dt.day >= 7))
        | (merged["GAME_DATE"].dt.month >= 5)
    ).astype(int)
    return merged


def _preprocess_boxscores(boxscores: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    game_dates = games_df[["GAME_ID", "GAME_DATE", "SEASON"]].drop_duplicates(subset=["GAME_ID"])
    merged = boxscores.merge(game_dates, on="GAME_ID", how="left")
    return merged


def _aggregate_team_boxscores(boxscores: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON"]
    sum_cols = _select_existing(boxscores, cols_to_sum)
    weighted_cols = _select_existing(boxscores, cols_to_weighted_avg)

    agg_sum = boxscores.groupby(group_keys)[sum_cols].sum().reset_index() if sum_cols else boxscores[group_keys].drop_duplicates()
    if weighted_cols:
        weighted = compute_weighted_mean_features(boxscores, group_keys, weighted_cols, weight_col="MINUTES_PLAYED")
        agg = agg_sum.merge(weighted, on=group_keys, how="left")
    else:
        agg = agg_sum
    return agg


def _build_player_team_features(boxscores: pd.DataFrame) -> pd.DataFrame:
    player_status = build_player_status_features(boxscores)
    team_presence = aggregate_actual_team_features(player_status)
    hist_tops = identify_historical_top_players(player_status)
    top_abs = flag_top_players_absences(player_status, hist_tops)
    team_features = team_presence.merge(top_abs, on=["GAME_ID", "TEAM_ID"], how="left")
    team_features["has_top_absent"] = (team_features["top_player_absent"].fillna(0) > 0).astype(int)
    team_features["has_absent"] = (team_features["num_absent"].fillna(0) > 0).astype(int)
    return team_features


def _attach_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = _select_existing(df, cols_to_sum + cols_to_weighted_avg + cols_player_stats + ["has_top_absent", "has_absent"])
    feature_cols = sorted(set(feature_cols))
    if not feature_cols:
        return df

    opp = df[["GAME_ID", "TEAM_ID"] + feature_cols].copy()
    rename_map = {col: f"OPP_{col}" for col in feature_cols}
    opp = opp.rename(columns=rename_map)
    merged = df.merge(
        opp,
        how="left",
        left_on=["GAME_ID", "OPP_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
        suffixes=("", "_DROP"),
    )
    merged = merged.drop(columns=["TEAM_ID_DROP"], errors="ignore")
    return merged


def assemble_match_dataset(
    games_df: pd.DataFrame,
    boxscores_df: pd.DataFrame,
    odds_dir: Optional[Path] = None,
) -> pd.DataFrame:
    games_meta = _prepare_games_meta(games_df)
    boxscores = _preprocess_boxscores(boxscores_df, games_meta)
    team_boxscores = _aggregate_team_boxscores(boxscores)
    team_players = _build_player_team_features(boxscores)

    team_boxscores = team_boxscores.drop(columns=["GAME_DATE", "SEASON"], errors="ignore")
    team_players = team_players.drop(columns=["GAME_DATE"], errors="ignore")

    team_stats = games_meta.merge(team_boxscores, on=["GAME_ID", "TEAM_ID"], how="left")
    team_stats = team_stats.merge(team_players, on=["GAME_ID", "TEAM_ID"], how="left")
    team_stats = _attach_opponent_features(team_stats)

    # Odds ingest disabled: the current bookmaker exports are incomplete so we keep
    # the helper code but stop injecting moneyline-based columns in silver datasets.
    # if odds_dir and Path(odds_dir).exists():
    #     odds_df = merge_odds_csv_files(str(odds_dir))
    #     team_stats = match_odds_with_dataset(odds_df, team_stats)

    team_stats = team_stats.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

    keep_cols = set(MATCH_IDENTIFIER_COLUMNS + ["TEAM_ID", "IS_HOME", "IS_WIN", "POINTS_FOR", "POINTS_AGAINST", "POINT_TOTAL", "POINT_DIFF", "ODDS"])
    keep_cols.update(MATCH_CONTEXT_FLAGS)
    keep_cols.update(BASE_TEAM_FEATURE_COLUMNS)
    filtered_cols = [col for col in team_stats.columns if col in keep_cols]
    filtered = team_stats[filtered_cols].copy()

    match_df = team_rows_to_match_rows(filtered)
    match_df = match_df.rename(
        columns={
            "HOME_ODDS": "HOME_MONEYLINE",
            "AWAY_ODDS": "AWAY_MONEYLINE",
        }
    )
    return match_df


def build_match_dataset_from_bronze(
    *,
    games_path: Optional[Path] = None,
    boxscores_path: Optional[Path] = None,
    odds_dir: Optional[Path] = Path(DATA_ODDS_HISTORY_DIR),
    persist_artifact: bool = True,
    output_prefix: str = "bronze_matches",
    output_format: str = "parquet",
) -> BuildResult:

    games_df = load_games_dataframe(games_path)
    boxscores_df = load_boxscores_dataframe(boxscores_path)
    match_df = assemble_match_dataset(games_df, boxscores_df, odds_dir=odds_dir)

    artifact_path: Optional[Path] = None
    if persist_artifact:
        filename = timestamped_filename(output_prefix, extension=output_format)
        artifact_path = Path(DATA_BRONZE_MATCHES_DIR) / filename
        save_dataset(match_df, artifact_path, fmt=output_format)

    metadata = {
        "rows": len(match_df),
        "columns": len(match_df.columns),
        "games_source": _resolve_path(games_path, DATA_BRONZE_GAMES_DIR).name,
        "boxscores_source": _resolve_path(boxscores_path, DATA_BRONZE_BOXSCORES_DIR).name,
    }

    return BuildResult(dataset=match_df, path=artifact_path, metadata=metadata)
