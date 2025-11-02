from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from feast import Entity, FeatureView, FileSource
from feast.data_source import FileFormat
from feast.field import Field

from nba_predictor import settings
from nba_predictor.feature_store.utils import infer_fields_from_parquet

TEAM_ENTITY = Entity(name="team", join_keys=["team_id"], description="NBA team identifier.")
OPPONENT_ENTITY = Entity(name="opponent", join_keys=["opponent_team_id"], description="Opponent team identifier.")

DEFAULT_SOURCE_PATH = settings.data_paths.silver_matchups_h2h_features
DEFAULT_EXCLUDE_COLUMNS = {
    "team_id",
    "opponent_team_id",
    "game_id",
    "game_date",
    "season",
    "ingest_ts",
    "created_at",
}


def build_h2h_source(path: Path | None = None) -> FileSource:
    source_dir = Path(path or DEFAULT_SOURCE_PATH)
    pattern = source_dir / "**/*.parquet"
    return FileSource(
        path=str(pattern),
        event_timestamp_column="game_date",
        created_timestamp_column=None,
        file_format=FileFormat.PARQUET,
    )


def build_team_vs_opp_h2h_view(
    *,
    source: Optional[FileSource] = None,
    fields: Optional[Iterable[Field]] = None,
    description: str | None = None,
) -> FeatureView:
    file_source = source or build_h2h_source()
    feature_fields = list(fields) if fields is not None else infer_fields_from_parquet(
        Path(file_source.path),
        exclude=DEFAULT_EXCLUDE_COLUMNS,
    )

    return FeatureView(
        name="team_vs_opp_h2h",
        entities=[TEAM_ENTITY, OPPONENT_ENTITY],
        ttl=None,
        schema=feature_fields,
        source=file_source,
        online=True,
        description=description or "Head-to-head aggregates between teams.",
    )
