from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feast.loader import fetch_historical_features

from .config import DatasetConfig, TrainingConfig


@dataclass
class TrainCalibTestSplit:
    X_train: pd.DataFrame
    X_calib: Optional[pd.DataFrame]
    X_test: pd.DataFrame
    y_train: pd.Series
    y_calib: Optional[pd.Series]
    y_test: pd.Series


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
    if cfg.target in feature_df.columns:
        feature_df = feature_df.drop(columns=[cfg.target], errors="ignore")
    if cfg.drop_columns:
        feature_df = feature_df.drop(columns=cfg.drop_columns, errors="ignore")
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
    df_meta: Optional[pd.DataFrame] = None,
) -> TrainCalibTestSplit:
    if cfg.split_strategy == "walk_forward":
        return _walk_forward_split(X, y, cfg, df_meta)

    stratify = y if (cfg.stratify and cfg.task_type == "classification") else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    return TrainCalibTestSplit(
        X_train=X_train,
        X_calib=None,
        X_test=X_test,
        y_train=y_train,
        y_calib=None,
        y_test=y_test,
    )


def generate_walk_forward_folds(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainingConfig,
    df_meta: Optional[pd.DataFrame],
) -> List[TrainCalibTestSplit]:
    """Create multiple chronological folds (train/calib/test) by sliding season windows."""
    if df_meta is None:
        raise ValueError("Walk-forward split requiert le dataframe source pour accéder aux saisons.")
    season_col = cfg.season_column
    if season_col not in df_meta.columns:
        raise KeyError(f"Colonne '{season_col}' introuvable pour le split walk-forward.")

    seasons = pd.Series(df_meta[season_col].astype(str))
    unique_seasons = seasons.dropna().unique().tolist()
    unique_seasons.sort()
    total_needed = cfg.train_seasons + cfg.calibration_seasons + cfg.test_seasons
    if len(unique_seasons) < total_needed:
        raise ValueError(
            f"Pas assez de saisons pour un split walk-forward ({len(unique_seasons)} < {total_needed})."
        )

    all_folds: List[TrainCalibTestSplit] = []
    for end_idx in range(total_needed, len(unique_seasons) + 1):
        window = unique_seasons[end_idx - total_needed : end_idx]
        train_seasons = window[: cfg.train_seasons]
        calib_seasons = window[cfg.train_seasons : cfg.train_seasons + cfg.calibration_seasons]
        test_seasons = window[-cfg.test_seasons :] if cfg.test_seasons else []

        def _mask(seasons_list):
            return seasons.isin(seasons_list)

        train_idx = _mask(train_seasons)
        calib_idx = _mask(calib_seasons) if calib_seasons else pd.Series([False] * len(seasons))
        test_idx = _mask(test_seasons)

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_calib = X.loc[calib_idx] if calib_seasons else None
        y_calib = y.loc[calib_idx] if calib_seasons else None
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        if X_train.empty or X_test.empty:
            continue

        all_folds.append(
            TrainCalibTestSplit(
                X_train=X_train,
                X_calib=X_calib,
                X_test=X_test,
                y_train=y_train,
                y_calib=y_calib,
                y_test=y_test,
            )
        )

    if not all_folds:
        raise ValueError("Aucun fold valide généré pour le split walk-forward.")

    return all_folds[-cfg.walk_forward_folds :] if cfg.walk_forward_folds > 0 else all_folds


def _walk_forward_split(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainingConfig,
    df_meta: Optional[pd.DataFrame],
) -> TrainCalibTestSplit:
    if df_meta is None:
        raise ValueError("Walk-forward split requiert le dataframe source pour accéder aux saisons.")
    season_col = cfg.season_column
    if season_col not in df_meta.columns:
        raise KeyError(f"Colonne '{season_col}' introuvable pour le split walk-forward.")

    seasons = pd.Series(df_meta[season_col].astype(str))
    unique_seasons = seasons.dropna().unique().tolist()
    unique_seasons.sort()
    total_needed = cfg.train_seasons + cfg.calibration_seasons + cfg.test_seasons
    if len(unique_seasons) < total_needed:
        raise ValueError(
            f"Pas assez de saisons pour un split walk-forward ({len(unique_seasons)} < {total_needed})."
        )

    train_start = len(unique_seasons) - total_needed
    train_seasons = unique_seasons[train_start : train_start + cfg.train_seasons]
    calib_start = train_start + cfg.train_seasons
    calib_seasons = unique_seasons[calib_start : calib_start + cfg.calibration_seasons]
    test_start = calib_start + cfg.calibration_seasons
    test_seasons = unique_seasons[test_start : test_start + cfg.test_seasons]

    def _mask(seasons_list):
        return seasons.isin(seasons_list)

    train_idx = _mask(train_seasons)
    calib_idx = _mask(calib_seasons) if calib_seasons else pd.Series([False] * len(seasons))
    test_idx = _mask(test_seasons)

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_calib = X.loc[calib_idx] if calib_seasons else None
    y_calib = y.loc[calib_idx] if calib_seasons else None
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    if X_train.empty or X_test.empty:
        raise ValueError("Split walk-forward invalide : train ou test vide.")

    return TrainCalibTestSplit(
        X_train=X_train,
        X_calib=X_calib,
        X_test=X_test,
        y_train=y_train,
        y_calib=y_calib,
        y_test=y_test,
    )
