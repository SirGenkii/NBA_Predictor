from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from feast import FeatureService, FeatureStore

FEAST_REPO_PATH = Path("src/feast")


def get_feature_store(repo_path: Optional[Path] = None) -> FeatureStore:
    repo = Path(repo_path) if repo_path else FEAST_REPO_PATH
    return FeatureStore(repo_path=str(repo))


def get_feature_service(name: str, *, repo_path: Optional[Path] = None) -> FeatureService:
    store = get_feature_store(repo_path)
    return store.get_feature_service(name)


def fetch_historical_features(
    *,
    entity_df: pd.DataFrame,
    feature_service: str,
    repo_path: Optional[Path] = None,
) -> pd.DataFrame:
    store = get_feature_store(repo_path)
    service = store.get_feature_service(feature_service)
    return store.get_historical_features(entity_df=entity_df, features=service).to_df()


def fetch_online_features(
    *,
    request_df: pd.DataFrame,
    feature_service: str,
    repo_path: Optional[Path] = None,
) -> pd.DataFrame:
    store = get_feature_store(repo_path)
    service = store.get_feature_service(feature_service)
    return store.get_online_features(
        features=service,
        entity_rows=request_df.to_dict("records"),
    ).to_df()
