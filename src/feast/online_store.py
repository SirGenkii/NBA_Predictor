from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import pandas as pd

from src.feast.loader import get_feature_store


POINT_TOTAL_FEATURE_SERVICE = "point_total_service"
POINT_TOTAL_FEATURE_VIEW = "point_total_features"


def build_point_total_online_payload(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    payload = df.copy()
    if "match_id" not in payload.columns:
        if "GAME_ID" not in payload.columns:
            raise KeyError("Le dataframe ne contient ni 'match_id' ni 'GAME_ID'.")
        payload["match_id"] = payload["GAME_ID"].astype(str)
    payload["match_id"] = payload["match_id"].astype(str)
    if "GAME_DATE" not in payload.columns:
        raise KeyError("Le dataframe doit contenir une colonne 'GAME_DATE'.")
    payload["GAME_DATE"] = pd.to_datetime(payload["GAME_DATE"])
    payload["event_timestamp"] = payload["GAME_DATE"]
    columns = ["match_id", "GAME_DATE", "event_timestamp", *point_total_feature_columns()]
    columns = _unique_columns(columns)
    return payload.reindex(columns=columns, fill_value=pd.NA)


def write_point_total_online_store(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    payload = build_point_total_online_payload(df)
    if payload.empty:
        return 0
    store = get_feature_store()
    store.write_to_online_store(POINT_TOTAL_FEATURE_VIEW, payload)
    return len(payload)


@lru_cache(maxsize=1)
def point_total_feature_columns() -> List[str]:
    store = get_feature_store()
    view = store.get_feature_view(POINT_TOTAL_FEATURE_VIEW)
    return _unique_columns([field.name for field in view.schema])


def _unique_columns(columns: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for col in columns:
        if col in seen:
            continue
        ordered.append(col)
        seen.add(col)
    return ordered


__all__ = [
    "POINT_TOTAL_FEATURE_SERVICE",
    "POINT_TOTAL_FEATURE_VIEW",
    "build_point_total_online_payload",
    "write_point_total_online_store",
    "point_total_feature_columns",
]
