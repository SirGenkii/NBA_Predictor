from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
import structlog

from nba_predictor import settings

logger = structlog.get_logger("nba_predictor.datasets.training")

TEAM_FACTS_ROOT = settings.data_paths.silver_team_game_facts
TEAM_FORM_ROOT = settings.data_paths.silver_team_form_windowed
H2H_FEATURES_ROOT = settings.data_paths.silver_matchups_h2h_features
OUTPUT_ROOT = settings.data_paths.gold_training_sets

DIFF_SUFFIX = "_diff"


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _scan_parquet(root: Path) -> pl.LazyFrame:
    pattern = root / "**/*.parquet"
    matches = glob(str(pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return pl.concat([pl.scan_parquet(path) for path in matches])


def _suffix_features(df: pl.LazyFrame, suffix: str, protected: Iterable[str]) -> pl.LazyFrame:
    protected_set = set(protected)
    rename_map = {col: f"{col}_{suffix}" for col in df.columns if col not in protected_set}
    if rename_map:
        df = df.rename(rename_map)
    return df


def _compute_differences(df: pl.LazyFrame) -> pl.LazyFrame:
    diff_exprs = []
    columns = df.columns
    for col in columns:
        if col.endswith("_home"):
            base = col[:-5]
            away_col = f"{base}_away"
            if away_col in columns:
                diff_exprs.append((pl.col(col) - pl.col(away_col)).alias(f"{base}{DIFF_SUFFIX}"))
    if diff_exprs:
        df = df.with_columns(diff_exprs)
    return df


def prepare_match_feature_frame() -> pl.LazyFrame:
    """
    Assemble a feature table combining team-form and H2H statistics for each match.
    """

    team_facts = _scan_parquet(TEAM_FACTS_ROOT).select(
        [
            "season",
            "game_date",
            "game_id",
            "team_id",
            "opponent_team_id",
            "is_home",
            "is_win",
            "team_points",
            "opponent_points",
            "team_plus_minus",
            "team_ast",
            "team_reb",
            "team_tov",
            "team_pf",
        ]
    )

    team_form = _scan_parquet(TEAM_FORM_ROOT)
    h2h_features = _scan_parquet(H2H_FEATURES_ROOT)

    joined = (
        team_facts.join(
            team_form,
            on=["season", "game_id", "team_id"],
            how="left",
        )
        .join(
            h2h_features,
            on=["season", "game_id", "team_id", "opponent_team_id"],
            how="left",
        )
        .with_columns(
            [
                pl.col("game_date").cast(pl.Date),
                pl.col("is_win").cast(pl.Int8),
            ]
        )
    )

    home = (
        joined.filter(pl.col("is_home"))
        .rename({"team_id": "home_team_id", "opponent_team_id": "away_team_id", "is_win": "home_is_win"})
        .pipe(
            _suffix_features,
            suffix="home",
            protected=["season", "game_date", "game_id", "home_team_id", "away_team_id", "home_is_win"],
        )
    )

    away = (
        joined.filter(~pl.col("is_home"))
        .rename({"team_id": "away_team_id", "opponent_team_id": "home_team_id", "is_win": "away_is_win"})
        .pipe(
            _suffix_features,
            suffix="away",
            protected=["season", "game_date", "game_id", "home_team_id", "away_team_id", "away_is_win"],
        )
    )

    dataset = (
        home.join(
            away,
            on=["season", "game_date", "game_id", "home_team_id", "away_team_id"],
            how="inner",
        )
        .pipe(_compute_differences)
        .with_columns(
            [
                pl.col("home_is_win").cast(pl.Int8).alias("target_home_win"),
                (pl.col("team_points_home") - pl.col("team_points_away")).alias("target_point_margin"),
            ]
        )
        .sort("game_date")
    )

    return dataset


def build_training_dataset(
    *,
    ingest_ts: str | None = None,
    game_dates: Sequence[str] | None = None,
    compression: str = "zstd",
) -> Path:
    """
    Build the gold training dataset with home/away features and diff columns.
    """

    ingest_ts = ingest_ts or _now_ts()
    logger.info("build_training_dataset_start", ingest_ts=ingest_ts)

    dataset = prepare_match_feature_frame()

    if game_dates:
        parsed_dates = [pl.lit(date).str.strptime(pl.Date, strict=False) for date in game_dates]
        dataset = dataset.filter(pl.col("game_date").is_in([d.evaluate() for d in parsed_dates]))

    materialized = dataset.collect()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"nba_training_{ingest_ts}.parquet"
    latest_path = OUTPUT_ROOT / "latest.parquet"

    materialized.write_parquet(output_path, compression=compression)
    materialized.write_parquet(latest_path, compression=compression)

    logger.info(
        "build_training_dataset_complete",
        rows=materialized.height,
        columns=len(materialized.columns),
        output=str(output_path),
    )
    return output_path


def build_scoring_payload(
    *,
    match_date: str,
    ingest_ts: str | None = None,
    compression: str = "zstd",
) -> Path:
    """
    Produce a scoring payload for a specific match date and persist under gold/scoring_payloads.
    """

    ingest_ts = ingest_ts or _now_ts()
    logger.info("build_scoring_payload_start", ingest_ts=ingest_ts, match_date=match_date)

    dataset = prepare_match_feature_frame()
    try:
        target_date = datetime.fromisoformat(match_date).date()
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"match_date must be ISO format YYYY-MM-DD, received {match_date}") from exc

    payload = dataset.filter(pl.col("game_date") == pl.lit(target_date)).collect()

    scoring_root = settings.data_paths.gold_scoring_payloads
    scoring_root.mkdir(parents=True, exist_ok=True)

    output_path = scoring_root / f"nba_scoring_{match_date}_{ingest_ts}.parquet"
    payload.write_parquet(output_path, compression=compression)

    logger.info(
        "build_scoring_payload_complete",
        rows=payload.height,
        output=str(output_path),
    )
    return output_path


__all__ = ["build_training_dataset", "build_scoring_payload", "prepare_match_feature_frame"]
