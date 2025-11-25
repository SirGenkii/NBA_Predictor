from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.modeling.builders import point_total_bundle
from src.modeling.data import load_dataset, prepare_features
from src.prediction.point_total import (
    PredictionRequest,
    _collect_prediction_features,
    _training_reference,
)


@dataclass(frozen=True)
class MatchKey:
    home_team_id: int
    away_team_id: int
    match_date: str  # ISO formatted date

    def tuple_key(self) -> Tuple[int, int, str]:
        return (int(self.home_team_id), int(self.away_team_id), self.match_date)


def matches_from_artifact(artifact: dict) -> List[MatchKey]:
    keys: List[MatchKey] = []
    for match in artifact.get("matches", []):
        meta = match.get("match") or {}
        home = meta.get("home") or {}
        away = meta.get("away") or {}
        match_date = meta.get("match_date")
        if not match_date:
            continue
        try:
            home_id = int(home.get("team_id"))
            away_id = int(away.get("team_id"))
        except (TypeError, ValueError):
            continue
        keys.append(MatchKey(home_team_id=home_id, away_team_id=away_id, match_date=match_date))
    return keys


def _normalize_date_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.date
    return pd.to_datetime(series).dt.date


def collect_offline_features(matches: Sequence[MatchKey]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame()
    bundle = point_total_bundle(enable_registry=False)
    df = load_dataset(bundle.dataset)
    df = df.copy()
    home_ids = df["HOME_TEAM_ID"].astype(int)
    away_ids = df["AWAY_TEAM_ID"].astype(int)
    match_dates = _normalize_date_column(df["GAME_DATE"])

    features, _, feature_names = prepare_features(df, bundle.dataset)
    features = features.copy()
    multi_index = pd.MultiIndex.from_arrays(
        [home_ids, away_ids, match_dates],
        names=["home_team_id", "away_team_id", "match_date"],
    )
    features.index = multi_index

    selected_rows = []
    missing = []
    for mk in matches:
        tuple_key = (mk.home_team_id, mk.away_team_id, pd.to_datetime(mk.match_date).date())
        if tuple_key not in features.index:
            missing.append(tuple_key)
            continue
        selected_rows.append(features.loc[tuple_key])
    if missing:
        raise KeyError(f"Matches introuvables dans le dataset offline: {missing}")
    offline_df = pd.DataFrame(selected_rows, columns=feature_names)
    offline_index: List[Tuple[int, int, object]] = []
    for mk in matches:
        tuple_key = (mk.home_team_id, mk.away_team_id, pd.to_datetime(mk.match_date).date())
        if tuple_key in features.index:
            offline_index.append(tuple_key)
    offline_df.index = pd.MultiIndex.from_tuples(
        offline_index,
        names=["home_team_id", "away_team_id", "match_date"],
    )
    return offline_df


def collect_online_features(matches: Sequence[MatchKey]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame()
    requests = [
        PredictionRequest(
            home_team_id=mk.home_team_id,
            away_team_id=mk.away_team_id,
            match_date=mk.match_date,
        )
        for mk in matches
    ]
    training_ref = _training_reference()
    features, entities = _collect_prediction_features(requests, training_ref.feature_names)
    match_index: List[Tuple[int, int, object]] = []
    for entity in entities:
        req = entity.request
        match_index.append((req.home_team_id, req.away_team_id, pd.to_datetime(req.match_date).date()))
    features.index = pd.MultiIndex.from_tuples(
        match_index,
        names=["home_team_id", "away_team_id", "match_date"],
    )
    return features[training_ref.feature_names]


def compare_feature_frames(
    offline: pd.DataFrame,
    online: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    if offline.empty or online.empty:
        return pd.DataFrame()
    common_index = offline.index.intersection(online.index)
    rows: List[Dict[str, object]] = []
    for key in common_index:
        off_row = offline.loc[key]
        on_row = online.loc[key]
        diff = (off_row - on_row).abs()
        rows.append(
            {
                "match_key": key,
                "max_abs_diff": float(diff.max()) if not diff.empty else 0.0,
                "num_columns_over_tol": int((diff > tolerance).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)
