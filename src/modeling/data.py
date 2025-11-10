from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

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

    features = df.drop(columns=[cfg.target], errors="ignore")
    for col in cfg.drop_columns:
        if col in features.columns:
            features = features.drop(columns=col)

    if cfg.keep_numeric_only:
        numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
        features = features[numeric_cols]

    # Convertir bool -> int pour éviter les surprises avec LightGBM
    bool_cols = features.select_dtypes(include=["bool"]).columns
    features[bool_cols] = features[bool_cols].astype(int)

    # Drop columns that are entirely NaN (would break imputers/models)
    features = features.dropna(axis=1, how="all")

    if cfg.dropna:
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = df.loc[valid_idx, cfg.target]
    else:
        target = df[cfg.target]

    feature_names = features.columns.tolist()
    return features, target, feature_names


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
