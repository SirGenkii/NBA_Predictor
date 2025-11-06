from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import polars as pl
import structlog
import typer

from nba_predictor import settings
from nba_predictor.logging import configure_logging

app = typer.Typer(help="Rebuild bronze Parquet snapshots from legacy CSV exports.")

logger = structlog.get_logger("nba_predictor.scripts.rebuild_bronze")


def _default_ingest_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iter_csv_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def _relative_parent(path: Path, root: Path) -> Path:
    try:
        return path.parent.relative_to(root)
    except ValueError:
        return path.parent


def _make_output_dir(dest_root: Path, snapshot: str, relative_parent: Path, ingest_ts: str) -> Path:
    output_dir = dest_root / snapshot / relative_parent / f"ingest_ts={ingest_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _hash_source(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return digest[:12]


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        try_parse_dates=True,
        ignore_errors=True,
        null_values=["", "None", "null", "NULL"],
        schema_overrides=SCHEMA_OVERRIDES,
    )


def _append_metadata(df: pl.DataFrame, ingest_ts: str, snapshot: str, source_path: Path) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.lit(ingest_ts).alias("ingest_ts"),
            pl.lit(snapshot).alias("source_snapshot"),
            pl.lit(str(source_path)).alias("source_file"),
        ]
    )


def _write_dataframe(df: pl.DataFrame, output_dir: Path, source_path: Path) -> Path:
    filename = f"{source_path.stem}_{_hash_source(source_path)}.parquet"
    destination = output_dir / filename
    df.write_parquet(destination, compression="zstd")
    return destination


def _normalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {name: _snake_case(name) for name in df.columns}
    return df.rename(rename_map)

IDENTIFIER_COLUMNS = {
    "game_id",
    "team_id",
    "opponent_team_id",
    "home_team_id",
    "away_team_id",
    "player_id",
    "person_id",
    "opp_team_id",
    "opp_player_id",
    "opp_person_id",
    "teamid",
    "gameid",
    "playerid",
}

_IDENTIFIER_SCHEMA_NAMES = {
    "gameId",
    "GAME_ID",
    "GameID",
    "game_id",
    "teamId",
    "TEAM_ID",
    "TeamID",
    "team_id",
    "opponentTeamId",
    "OPPONENT_TEAM_ID",
    "opponent_team_id",
    "homeTeamId",
    "HOME_TEAM_ID",
    "awayTeamId",
    "AWAY_TEAM_ID",
    "playerId",
    "PLAYER_ID",
    "player_id",
    "personId",
    "PERSON_ID",
    "person_id",
    "oppTeamId",
    "opp_team_id",
    "oppPlayerId",
    "oppPersonId",
    "TEAMID",
    "GAMEID",
}

SCHEMA_OVERRIDES = {name: pl.Utf8 for name in _IDENTIFIER_SCHEMA_NAMES}


def _cast_identifier_columns(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in IDENTIFIER_COLUMNS:
            expr = pl.col(col)
            expr = expr.cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).cast(pl.Utf8, strict=False)
            if "game_id" in col_lower or col_lower.endswith("gameid"):
                expr = expr.str.zfill(10)
            out = out.with_columns(expr.alias(col))
    return out


def _snake_case(name: str) -> str:
    name = name.strip()
    name = name.replace("%", "pct")
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.replace("__", "_")
    return name.lower()


def _process_directory(source_root: Path, dest_root: Path, ingest_ts: str, snapshot_label: str, *, normalise: bool) -> Tuple[int, int]:
    files = list(_iter_csv_files(source_root))
    if not files:
        logger.warning("no_csv_found", root=str(source_root))
        return 0, 0

    written = 0
    total_rows = 0
    for csv_path in files:
        relative_parent = _relative_parent(csv_path, source_root)
        output_dir = _make_output_dir(dest_root, snapshot_label, relative_parent, ingest_ts)
        try:
            df = _read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("csv_read_failed", path=str(csv_path), error=str(exc))
            continue

        if df.is_empty():
            logger.warning("csv_empty", path=str(csv_path))
            continue

        if normalise:
            df = _normalise_columns(df)

        df = _cast_identifier_columns(df)

        df = _append_metadata(df, ingest_ts, snapshot_label, csv_path)
        total_rows += df.height

        destination = _write_dataframe(df, output_dir, csv_path)
        written += 1
        logger.info(
            "bronze_file_written",
            source=str(csv_path),
            destination=str(destination),
            rows=df.height,
        )

    return written, total_rows


@app.command("all")
def rebuild_all(
    ingest_ts: str = typer.Option(None, help="Timestamp identifier (UTC, format YYYYmmddTHHMMSSZ)."),
    include_raw: bool = typer.Option(True, help="Include legacy raw snapshots."),
    include_raw_last: bool = typer.Option(True, help="Include latest raw_last snapshots."),
    normalise_columns: bool = typer.Option(
        True,
        help="Convert column names to snake_case for downstream processing.",
    ),
) -> None:
    """
    Convert every CSV under `data/legacy` into bronze Parquet datasets.
    """

    configure_logging(settings.data_paths.logs_root, pipeline="rebuild_bronze")

    if ingest_ts is None:
        ingest_ts = _default_ingest_ts()

    dest_root = settings.data_paths.bronze_root / "nba_api"
    dest_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_rows = 0

    if include_raw:
        written, rows = _process_directory(
            settings.data_paths.legacy_raw,
            dest_root,
            ingest_ts,
            snapshot_label="raw",
            normalise=normalise_columns,
        )
        total_files += written
        total_rows += rows

    if include_raw_last:
        written, rows = _process_directory(
            settings.data_paths.legacy_raw_last,
            dest_root,
            ingest_ts,
            snapshot_label="raw_last",
            normalise=normalise_columns,
        )
        total_files += written
        total_rows += rows

    logger.info(
        "bronze_rebuild_complete",
        files_written=total_files,
        total_rows=total_rows,
        ingest_ts=ingest_ts,
    )


if __name__ == "__main__":
    app()
