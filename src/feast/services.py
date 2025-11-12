from __future__ import annotations

from feast import FeatureService

from feature_views import matchup_features_view


point_total_feature_service = FeatureService(
    name="point_total_service",
    features=[matchup_features_view],
    tags={"target": "POINT_TOTAL"},
)


is_win_feature_service = FeatureService(
    name="is_win_service",
    features=[matchup_features_view],
    tags={"target": "IS_WIN"},
)
