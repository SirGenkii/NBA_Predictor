from __future__ import annotations

from pathlib import Path

from feast import FileSource

from src.config import DATA_BRONZE_MATCHES_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_SILVER_EXPORT = (PROJECT_ROOT / "data/feast_sources/silver_latest.parquet").resolve()


def _pattern_path(directory: Path, pattern: str) -> str:
    base = (PROJECT_ROOT / directory).resolve()
    return str(base / pattern)


matches_bronze_source = FileSource(
    name="matches_bronze",
    path=_pattern_path(DATA_BRONZE_MATCHES_DIR, "bronze_matches_*.parquet"),
    timestamp_field="GAME_DATE",
    created_timestamp_column=None,
)


silver_feature_source = FileSource(
    name="silver_features",
    path=str(FEAST_SILVER_EXPORT),
    timestamp_field="GAME_DATE",
    created_timestamp_column=None,
)


matchup_features_source = FileSource(
    name="matchup_features",
    path=str(FEAST_SILVER_EXPORT),
    timestamp_field="GAME_DATE",
    created_timestamp_column=None,
)
