from __future__ import annotations

from datetime import timedelta

from feast import FeatureView

from data_sources import matchup_features_source
from entities import match_entity
from schema_loader import load_silver_feature_fields


try:
    matchup_fields = load_silver_feature_fields()
except FileNotFoundError as exc:
    raise RuntimeError(
        "Feast silver export missing. Run `python -m src.feast.pipeline` before applying Feast."
    ) from exc


matchup_features_view = FeatureView(
    name="matchup_features",
    entities=[match_entity],
    ttl=timedelta(days=7),
    schema=matchup_fields,
    online=True,
    source=matchup_features_source,
    tags={"stage": "gold"},
)
