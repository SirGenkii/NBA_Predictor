from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feast.loader import fetch_historical_features

from .config import DatasetConfig, TrainingConfig


def load_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    path = cfg.resolve_path()
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Format non supporté pour {path}")
    df.attrs["source_path"] = str(path)
    return df


def prepare_features(
    df: pd.DataFrame, cfg: DatasetConfig
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if cfg.target not in df.columns:
        raise KeyError(f"Cible {cfg.target} introuvable dans le dataset.")

    target = df[cfg.target]
    if cfg.use_feast:
        features = _feast_features(df, cfg)
    else:
        features = _dataframe_features(df, cfg)

    if cfg.dropna:
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

    feature_names = features.columns.tolist()
    return features, target, feature_names


def _dataframe_features(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    features = df.drop(columns=[cfg.target], errors="ignore")
    for col in cfg.drop_columns:
        if col in features.columns:
            features = features.drop(columns=col)

    if cfg.keep_numeric_only:
        numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
        features = features[numeric_cols]

    bool_cols = features.select_dtypes(include=["bool"]).columns
    features[bool_cols] = features[bool_cols].astype(int)
    features = features.dropna(axis=1, how="all")
    return features


def _feast_features(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    if not cfg.feast_feature_service:
        raise ValueError("DatasetConfig.use_feast=True mais aucun feature service n'est défini.")
    timestamp_col = cfg.feast_timestamp_column
    if timestamp_col not in df.columns:
        raise KeyError(f"Colonne timestamp '{timestamp_col}' introuvable pour la récupération Feast.")

    entity_df = pd.DataFrame()
    entity_df["event_timestamp"] = pd.to_datetime(df[timestamp_col])
    for feast_key, source_col in cfg.feast_entity_mapping.items():
        if source_col not in df.columns:
            raise KeyError(f"Colonne '{source_col}' requise pour l'entité Feast '{feast_key}'.")
        entity_df[feast_key] = df[source_col].astype(str)
    entity_df["__row_id"] = range(len(df))

    try:
        feature_df = fetch_historical_features(
            entity_df=entity_df,
            feature_service=cfg.feast_feature_service,
            repo_path=cfg.feast_repo_path,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de récupérer les features Feast ({cfg.feast_feature_service}): {exc}"
        ) from exc
    if "__row_id" not in feature_df.columns:
        raise RuntimeError("La récupération Feast n'a pas renvoyé '__row_id'; impossible d'aligner les colonnes.")

    drop_cols = ["event_timestamp"] + list(cfg.feast_entity_mapping.keys())
    feature_df = (
        feature_df.sort_values("__row_id")
        .drop(columns=drop_cols, errors="ignore")
        .set_index("__row_id")
    )
    if cfg.keep_numeric_only:
        numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns
        feature_df = feature_df[numeric_cols]
    bool_cols = feature_df.select_dtypes(include=["bool"]).columns
    feature_df[bool_cols] = feature_df[bool_cols].astype(int)
    feature_df = feature_df.dropna(axis=1, how="all")
    feature_df.index = df.index[: len(feature_df)]
    return feature_df


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainingConfig,
):
    stratify = y if (cfg.stratify and cfg.task_type == "classification") else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
