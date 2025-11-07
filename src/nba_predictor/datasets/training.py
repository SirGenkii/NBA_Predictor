from __future__ import annotations

import gc
import shutil
import tempfile
from datetime import date, datetime, timezone
from glob import glob
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import polars as pl
import structlog
import typer

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None

from nba_predictor import settings

logger = structlog.get_logger("nba_predictor.datasets.training")

TEAM_FACTS_ROOT = settings.data_paths.silver_team_game_facts
TEAM_FORM_ROOT = settings.data_paths.silver_team_form_windowed
H2H_FEATURES_ROOT = settings.data_paths.silver_matchups_h2h_features
OUTPUT_ROOT = settings.data_paths.gold_training_sets

DIFF_SUFFIX = "_diff"
DEFAULT_CHUNK_SIZE = 1  # unique game dates per chunk to control memory usage
MAX_JOINED_COLUMNS = 2_500
COMPUTE_FEATURE_DIFFERENCES = False
METADATA_COLUMNS = {
    "team_name",
    "team_tricode",
    "team_abbreviation",
    "bronze_ingest_ts",
    "bronze_source_snapshot",
    "bronze_source_file",
}
TEAM_FORM_DROP_SUBSTRINGS = ("_std_last", "last25")
H2H_DROP_SUBSTRINGS = ("_std_last", "last25")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_ingest_directories(root: Path, seasons: Sequence[str] | None = None, *, label: str | None = None) -> list[Path]:
    season_filter = set(seasons) if seasons else None
    ingest_dirs: list[Path] = []
    for season_dir in sorted(root.glob("season=*")):
        if not season_dir.is_dir():
            continue
        _, _, season_value = season_dir.name.partition("=")
        if season_filter and season_value not in season_filter:
            continue
        candidates = sorted(p for p in season_dir.glob("ingest_ts=*") if p.is_dir())
        if not candidates:
            continue
        latest = candidates[-1]
        ingest_dirs.append(latest)
        if label:
            logger.debug(
                "selected_ingest_partition",
                dataset=label,
                season=season_value,
                ingest_ts=latest.name.partition("=")[-1],
            )
    return ingest_dirs


def _scan_parquet(root: Path, *, seasons: Sequence[str] | None = None, label: str | None = None) -> pl.LazyFrame:
    ingest_dirs = _latest_ingest_directories(root, seasons=seasons, label=label)
    matches: list[str] = []
    for directory in ingest_dirs:
        matches.extend(glob(str(directory / "**" / "*.parquet"), recursive=True))
    if not matches:
        target = ", ".join(seasons) if seasons else "all seasons"
        raise FileNotFoundError(f"No parquet files found under {root} for {target}")
    return pl.concat([pl.scan_parquet(path) for path in matches], how="diagonal_relaxed")


def _drop_metadata_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    if not METADATA_COLUMNS:
        return df
    return df.select(pl.all().exclude(list(METADATA_COLUMNS)))


def _drop_columns_containing(df: pl.LazyFrame, substrings: Sequence[str]) -> pl.LazyFrame:
    if not substrings:
        return df
    schema = df.collect_schema()
    drop_cols = [col for col in schema.names() if any(token in col for token in substrings)]
    if drop_cols:
        df = df.drop(drop_cols)
    return df


def _suffix_features(df: pl.LazyFrame, suffix: str, protected: Iterable[str]) -> pl.LazyFrame:
    protected_set = set(protected)
    schema = df.collect_schema()
    rename_map = {col: f"{col}_{suffix}" for col in schema.names() if col not in protected_set}
    if rename_map:
        df = df.rename(rename_map)
    return df


def _guard_join_width(df: pl.LazyFrame, *, label: str) -> None:
    schema = df.collect_schema()
    width = len(schema)
    if width > MAX_JOINED_COLUMNS:
        raise RuntimeError(
            f"{label} has {width} columns which exceeds MAX_JOINED_COLUMNS={MAX_JOINED_COLUMNS}. "
            "Drop unused features upstream before building gold."
        )


def _with_join_guard(df: pl.LazyFrame, *, label: str) -> pl.LazyFrame:
    _guard_join_width(df, label=label)
    return df


def _compute_differences(df: pl.LazyFrame) -> pl.LazyFrame:
    schema = df.collect_schema()
    away_cols = {col for col in schema.names() if col.endswith("_away")}
    diff_exprs = []
    for col in schema.names():
        if col.endswith("_home"):
            base = col[:-5]
            if base in {"is_home"}:
                continue
            away_col = f"{base}_away"
            if away_col in away_cols:
                left_type = schema[col]
                right_type = schema[away_col]
                if _is_numeric(left_type) and _is_numeric(right_type):
                    diff_exprs.append((pl.col(col) - pl.col(away_col)).alias(f"{base}{DIFF_SUFFIX}"))
    if diff_exprs:
        df = df.with_columns(diff_exprs)
    return df


def prepare_match_feature_frame_streaming(
    *,
    seasons: Sequence[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Return home and away LazyFrames ready for a streaming join."""

    team_facts = _scan_parquet(TEAM_FACTS_ROOT, seasons=seasons, label="team_game_facts").select(
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
    team_form = (
        _scan_parquet(TEAM_FORM_ROOT, seasons=seasons, label="team_form_windowed")
        .pipe(_drop_metadata_columns)
        .pipe(_drop_columns_containing, substrings=TEAM_FORM_DROP_SUBSTRINGS)
    )
    h2h_features = (
        _scan_parquet(H2H_FEATURES_ROOT, seasons=seasons, label="matchups_h2h_features")
        .pipe(_drop_metadata_columns)
        .pipe(_drop_columns_containing, substrings=H2H_DROP_SUBSTRINGS)
    )

    if date_start and date_end:
        date_filter = (pl.col("game_date") >= pl.lit(date_start)) & (pl.col("game_date") <= pl.lit(date_end))
        team_facts = team_facts.filter(date_filter)
        team_form = team_form.filter(date_filter)
        h2h_features = h2h_features.filter(date_filter)

    joined = (
        team_facts.join(
            team_form,
            on=["season", "game_id", "team_id"],
            how="left",
            suffix="_team_form",
        )
        .join(
            h2h_features,
            on=["season", "game_id", "team_id", "opponent_team_id"],
            how="left",
            suffix="_h2h",
        )
        .with_columns([
            pl.col("game_date").cast(pl.Date),
            pl.col("is_win").cast(pl.Int8),
        ])
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

    return home, away


def prepare_match_feature_frame(
    *,
    seasons: Sequence[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> pl.LazyFrame:
    """Wrapper that keeps the previous API while leveraging streaming joins."""

    home, away = prepare_match_feature_frame_streaming(
        seasons=seasons,
        date_start=date_start,
        date_end=date_end,
    )

    merged = home.join(
        away,
        on=["season", "game_date", "game_id", "home_team_id", "away_team_id"],
        how="inner",
    ).pipe(_with_join_guard, label="gold_join")

    if COMPUTE_FEATURE_DIFFERENCES:
        merged = merged.pipe(_compute_differences)

    return (
        merged.with_columns([
            pl.col("home_is_win").cast(pl.Int8).alias("target_home_win"),
            (pl.col("team_points_home") - pl.col("team_points_away")).alias("target_point_margin"),
        ])
        .sort("game_date")
    )


def _season_partitions(root: Path) -> list[str]:
    if not root.exists():
        return []
    seasons: list[str] = []
    for path in sorted(root.glob("season=*")):
        if path.is_dir():
            _, _, value = path.name.partition("=")
            if value:
                seasons.append(value)
    return seasons


def _season_label_from_date(value: date) -> str:
    start_year = value.year if value.month >= 7 else value.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _chunk_date_ranges(dates: list[date], chunk_size: int) -> Iterator[tuple[date, date]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for idx in range(0, len(dates), chunk_size):
        start = dates[idx]
        end = dates[min(idx + chunk_size - 1, len(dates) - 1)]
        yield start, end


def _season_game_dates(season: str) -> list[date]:
    dates_lazy = (
        _scan_parquet(TEAM_FACTS_ROOT, seasons=[season], label="team_game_facts")
        .select(pl.col("game_date").cast(pl.Date))
        .unique()
    )
    dates_df = dates_lazy.collect(streaming=True).sort("game_date")
    return dates_df["game_date"].to_list()


def build_training_dataset(
    *,
    ingest_ts: str | None = None,
    game_dates: Sequence[str] | None = None,
    compression: str = "zstd",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """
    Build the gold training dataset with home/away features and diff columns.

    Data is processed per season and further chunked by `chunk_size` unique game dates
    to avoid large in-memory materialisations.
    """

    if pq is None:  # pragma: no cover
        raise ImportError("pyarrow is required to build the training dataset. Install `pyarrow` and retry.")

    ingest_ts = ingest_ts or _now_ts()
    logger.info("build_training_dataset_start", ingest_ts=ingest_ts)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"nba_training_{ingest_ts}.parquet"
    latest_path = OUTPUT_ROOT / "latest.parquet"

    target_dates_by_season: dict[str, set[date]] = {}
    if game_dates:
        for raw_value in game_dates:
            parsed_date = datetime.fromisoformat(raw_value).date()
            season_label = _season_label_from_date(parsed_date)
            target_dates_by_season.setdefault(season_label, set()).add(parsed_date)

    seasons = _season_partitions(TEAM_FACTS_ROOT)
    if not seasons:
        raise FileNotFoundError(f"No season partitions found under {TEAM_FACTS_ROOT}")

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    column_count = 0

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for season in seasons:
                if target_dates_by_season and season not in target_dates_by_season:
                    continue

                logger.info("build_training_dataset_season_start", season=season)

                allowed_dates = target_dates_by_season.get(season)
                if allowed_dates:
                    date_list = sorted(allowed_dates)
                else:
                    date_list = _season_game_dates(season)

                if not date_list:
                    logger.warning("build_training_dataset_no_dates", season=season)
                    continue

                season_rows = 0
                chunk_counter = 0
                for start_date, end_date in _chunk_date_ranges(date_list, chunk_size):
                    chunk_counter += 1
                    logger.debug(
                        "build_training_dataset_chunk_start",
                        season=season,
                        chunk=chunk_counter,
                        start=str(start_date),
                        end=str(end_date),
                    )

                    home_lazy, away_lazy = prepare_match_feature_frame_streaming(
                        seasons=[season],
                        date_start=start_date,
                        date_end=end_date,
                    )

                    home_temp = tmpdir_path / f"{season}_{chunk_counter}_home.parquet"
                    away_temp = tmpdir_path / f"{season}_{chunk_counter}_away.parquet"

                    home_lazy.sort("game_date").sink_parquet(
                        home_temp.as_posix(),
                        compression=compression,
                        maintain_order=True,
                        statistics=False,
                    )

                    away_lazy.sort("game_date").sink_parquet(
                        away_temp.as_posix(),
                        compression=compression,
                        maintain_order=True,
                        statistics=False,
                    )

                    home_df = pl.scan_parquet(home_temp.as_posix())
                    away_df = pl.scan_parquet(away_temp.as_posix())

                    merged = (
                        home_df.join(
                            away_df,
                            on=["season", "game_date", "game_id", "home_team_id", "away_team_id"],
                            how="inner",
                        )
                        .pipe(_compute_differences)
                        .with_columns([
                            pl.col("home_is_win").cast(pl.Int8).alias("target_home_win"),
                            (pl.col("team_points_home") - pl.col("team_points_away")).alias("target_point_margin"),
                        ])
                        .sort("game_date")
                    )

                    chunk_output = tmpdir_path / f"{season}_{chunk_counter}.parquet"
                    merged.sink_parquet(
                        chunk_output.as_posix(),
                        compression=compression,
                        maintain_order=True,
                        statistics=False,
                    )

                    home_temp.unlink(missing_ok=True)
                    away_temp.unlink(missing_ok=True)

                    parquet_file = pq.ParquetFile(chunk_output.as_posix())
                    if parquet_file.metadata.num_rows == 0:
                        chunk_output.unlink(missing_ok=True)
                        continue

                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path.as_posix(),
                            parquet_file.schema_arrow,
                            compression=compression,
                        )
                        column_count = parquet_file.schema_arrow.num_fields

                    for row_group_index in range(parquet_file.num_row_groups):
                        writer.write_table(parquet_file.read_row_group(row_group_index))

                    chunk_rows = parquet_file.metadata.num_rows
                    season_rows += chunk_rows
                    total_rows += chunk_rows
                    chunk_output.unlink(missing_ok=True)

                    del parquet_file, home_df, away_df, merged
                    gc.collect()

                    logger.debug(
                        "build_training_dataset_chunk_complete",
                        season=season,
                        chunk=chunk_counter,
                        start=str(start_date),
                        end=str(end_date),
                        rows=chunk_rows,
                    )

                logger.info(
                    "build_training_dataset_season_complete",
                    season=season,
                    rows=season_rows,
                    chunks=chunk_counter,
                )
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise RuntimeError("No rows produced while building the training dataset.")

    shutil.copy2(output_path, latest_path)

    logger.info(
        "build_training_dataset_complete",
        rows=total_rows,
        columns=column_count,
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

    try:
        target_date = datetime.fromisoformat(match_date).date()
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"match_date must be ISO format YYYY-MM-DD, received {match_date}") from exc

    season_label = _season_label_from_date(target_date)
    dataset = prepare_match_feature_frame(seasons=[season_label], date_start=target_date, date_end=target_date)
    payload = dataset.collect(streaming=True)

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


app = typer.Typer(help="Gold dataset utilities.")


@app.command("build")
def cli_build_training_dataset(
    ingest_ts: str | None = typer.Option(None, help="Custom ingest timestamp."),
    chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, min=1, help="Unique game dates per chunk."),
    compression: str = typer.Option("zstd", help="Parquet compression codec."),
) -> None:
    path = build_training_dataset(ingest_ts=ingest_ts, chunk_size=chunk_size, compression=compression)
    typer.echo(f"Training dataset written to {path}")


@app.command("scoring")
def cli_build_scoring_payload(
    match_date: str = typer.Argument(..., help="Match date in YYYY-MM-DD format."),
    ingest_ts: str | None = typer.Option(None, help="Custom ingest timestamp."),
    compression: str = typer.Option("zstd", help="Parquet compression codec."),
) -> None:
    path = build_scoring_payload(match_date=match_date, ingest_ts=ingest_ts, compression=compression)
    typer.echo(f"Scoring payload written to {path}")


NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Decimal,
)


def _is_numeric(dtype: pl.DataType) -> bool:
    return isinstance(dtype, NUMERIC_DTYPES)


__all__ = ["build_training_dataset", "build_scoring_payload", "prepare_match_feature_frame"]


if __name__ == "__main__":
    app()
