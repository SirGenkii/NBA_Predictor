from __future__ import annotations

from datetime import timedelta

from feast import FeatureView

from data_sources import matchup_features_source
from entities import match_entity
from schema_loader import load_silver_feature_fields


def _load_fields(exclude_extra=None):
    try:
        return load_silver_feature_fields(exclude_columns=exclude_extra)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Feast silver export missing. Run `python -m src.feast.pipeline` before applying Feast."
        ) from exc


point_total_fields = _load_fields()
is_win_fields = _load_fields()


point_total_features_view = FeatureView(
    name="point_total_features",
    entities=[match_entity],
    ttl=timedelta(days=7),
    schema=point_total_fields,
    online=True,
    source=matchup_features_source,
    tags={"stage": "silver", "target": "POINT_TOTAL"},
)


is_win_features_view = FeatureView(
    name="is_win_features",
    entities=[match_entity],
    ttl=timedelta(days=7),
    schema=is_win_fields,
    online=True,
    source=matchup_features_source,
    tags={"stage": "silver", "target": "IS_WIN"},
)
