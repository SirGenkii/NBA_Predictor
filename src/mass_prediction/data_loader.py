from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ATP_TOURNAMENTS_FILE,
    PLAYERS_LIST_CSV,
    RESULT_BRONZE_DATA_DIR,
    RESULT_MODEL_DATA_DIR,
)
from src.prediction_pipeline_result import load_artifacts


@dataclass(frozen=True)
class ArtifactsBundle:
    artifacts: object
    run_path: Path


class DataLoaderError(RuntimeError):
    """Raised when required datasets cannot be found."""


def find_latest_model_dir(base_dir: Optional[Path | str] = None) -> Path:
    base = Path(base_dir or RESULT_MODEL_DATA_DIR)
    if not base.exists():
        raise DataLoaderError(f"Model directory inconnue : {base}")
    candidates = sorted([p for p in base.iterdir() if p.is_dir()])
    if not candidates:
        raise DataLoaderError(f"Aucun modèle trouvé dans {base}")
    return candidates[-1]


def load_latest_artifacts(run_dir: Optional[Path | str] = None) -> ArtifactsBundle:
    run_path = Path(run_dir) if run_dir else find_latest_model_dir()
    artifacts = load_artifacts(run_path)
    return ArtifactsBundle(artifacts=artifacts, run_path=run_path)


def load_latest_history() -> pd.DataFrame:
    bronze_dir = Path(RESULT_BRONZE_DATA_DIR)
    csv_files = sorted(bronze_dir.glob("bronze_results_*.csv"))
    if not csv_files:
        raise DataLoaderError("Aucun fichier bronze_results_*.csv trouvé pour le pipeline résultat")
    history = pd.read_csv(csv_files[-1], low_memory=False)
    if "tourney_date" in history.columns:
        history["tourney_date"] = pd.to_datetime(history["tourney_date"], errors="coerce")
    return history


def load_players_catalog() -> pd.DataFrame:
    candidates = ("utf-8", "latin-1", "cp1252")
    last_exc: Optional[Exception] = None
    for encoding in candidates:
        try:
            df = pd.read_csv(PLAYERS_LIST_CSV, encoding=encoding)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    else:  # pragma: no cover - depends on local files
        raise DataLoaderError(f"Impossible de lire {PLAYERS_LIST_CSV}: {last_exc}")

    df = df.rename(columns={"id": "player_id"})
    df["player_id"] = df["player_id"].astype(str)
    if "birthdate" in df.columns:
        raw = df["birthdate"].astype(str).str.strip().replace({"": np.nan})
        df["birthdate"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    if "height" in df.columns:
        df["height"] = pd.to_numeric(df["height"], errors="coerce")
    return df


def compute_latest_player_stats(history: pd.DataFrame) -> pd.DataFrame:
    prefixes: set[str] = set()
    for col in history.columns:
        if "_" not in col:
            continue
        prefix, suffix = col.split("_", 1)
        if suffix in {"hand", "ioc", "name", "player_id", "rank", "rank_points", "age", "height"}:
            prefixes.add(f"{prefix}_")

    base_cols = [col for col in history.columns if all(not col.startswith(pref) for pref in prefixes)]

    neutral_frames = []
    for prefix in prefixes:
        prefixed_cols = [col for col in history.columns if col.startswith(prefix)]
        if not prefixed_cols:
            continue
        rename_map: Dict[str, str] = {}
        for col in prefixed_cols:
            suffix = col[len(prefix):]
            match suffix:
                case "id":
                    rename_map[col] = "player_id"
                case "name":
                    rename_map[col] = "name"
                case "ioc":
                    rename_map[col] = "ioc"
                case "hand":
                    rename_map[col] = "hand"
                case "rank":
                    rename_map[col] = "rank"
                case "rank_points":
                    rename_map[col] = "rank_points"
                case "age":
                    rename_map[col] = "age"
                case "height" | "ht":
                    rename_map[col] = "height"
        if not rename_map:
            continue
        subset = history[base_cols + list(rename_map.keys())].copy()
        subset = subset.rename(columns=rename_map)
        neutral_frames.append(subset)

    if not neutral_frames:
        return pd.DataFrame()

    combined = pd.concat(neutral_frames, ignore_index=True, sort=False)
    combined = combined.dropna(subset=["player_id"])
    if "tourney_date" in combined.columns:
        combined["tourney_date"] = pd.to_datetime(combined["tourney_date"], errors="coerce")
        combined = combined.sort_values("tourney_date")
    else:
        combined = combined.sort_index()

    latest = combined.drop_duplicates(subset="player_id", keep="last").set_index("player_id")
    return latest
 

def load_tournaments_catalog() -> pd.DataFrame:
    df = pd.read_csv(ATP_TOURNAMENTS_FILE)
    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    if "tourney_level" in df.columns:
        df["tourney_level"] = (
            df["tourney_level"].astype(str).str.strip().str.upper().replace({"": np.nan})
        )
    if "indoor_flag" in df.columns:
        def _normalize_indoor(value):
            if pd.isna(value):
                return np.nan
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                text = str(value).strip().lower()
                if text in {"1", "true", "t", "y", "yes", "indoor"}:
                    return 1.0
                if text in {"0", "false", "f", "n", "no", "outdoor", "out"}:
                    return 0.0
                return np.nan
            else:
                return 1.0 if numeric >= 1 else 0.0

        df["indoor_flag"] = df["indoor_flag"].apply(_normalize_indoor)
    return df


__all__ = [
    "ArtifactsBundle",
    "DataLoaderError",
    "compute_latest_player_stats",
    "find_latest_model_dir",
    "load_latest_artifacts",
    "load_latest_history",
    "load_players_catalog",
    "load_tournaments_catalog",
]
