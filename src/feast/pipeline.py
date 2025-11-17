from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import shutil
from src.datasets.bronze import build_match_dataset_from_bronze
from src.datasets.gold import GoldBuildConfig, build_gold_dataset
from src.datasets.silver import build_silver_dataset
from src.prediction.data_refresh import refresh_recent_boxscores


DEFAULT_TARGETS = ("POINT_TOTAL", "IS_WIN")
FEATURE_STORE_PATH = Path("src/feast/feature_store.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
FEAST_EXPORT_DIR = (REPO_ROOT / "data/feast_sources")
FEAST_SILVER_EXPORT = FEAST_EXPORT_DIR / "silver_latest.parquet"


@dataclass
class FeastBuildArtifacts:
    seasons: List[str]
    match_path: Optional[Path]
    silver_path: Optional[Path]
    gold_paths: Dict[str, Optional[Path]] = field(default_factory=dict)


def refresh_and_build(
    *,
    seasons: Sequence[str],
    targets: Sequence[str] = DEFAULT_TARGETS,
    build_silver: bool = True,
    build_gold: bool = True,
) -> FeastBuildArtifacts:
    season_list = sorted(set(seasons))
    if not season_list:
        raise ValueError("At least one season must be provided.")

    refresh_recent_boxscores(season_list)

    match_result = build_match_dataset_from_bronze()
    silver_result = None
    if build_silver:
        silver_result = build_silver_dataset(match_result.dataset)
        _sync_export(silver_result.path, FEAST_SILVER_EXPORT)

    gold_paths: Dict[str, Optional[Path]] = {}
    if build_gold:
        base_df = silver_result.dataset if silver_result else build_silver_dataset(match_result.dataset).dataset
        for target in targets:
            cfg = GoldBuildConfig(target=target, output_prefix="gold_dataset")
            gold_result = build_gold_dataset(base_df, config=cfg)
            gold_paths[target] = gold_result.path

    return FeastBuildArtifacts(
        seasons=season_list,
        match_path=match_result.path,
        silver_path=silver_result.path if silver_result else None,
        gold_paths=gold_paths,
    )


def run_feast_apply(feature_store: Path = FEATURE_STORE_PATH) -> None:
    repo_dir = feature_store.parent
    _run_feast_cmd(["-c", str(repo_dir), "apply"])


def run_feast_materialize(
    *,
    since: Optional[datetime] = None,
    feature_store: Path = FEATURE_STORE_PATH,
    incremental: bool = True,
) -> None:
    if incremental:
        ref = since or (datetime.now(timezone.utc) - timedelta(days=30))
        start = ref.isoformat(timespec="seconds")
        repo_dir = feature_store.parent
        _run_feast_cmd(["-c", str(repo_dir), "materialize-incremental", start])
    else:
        if since is None:
            raise ValueError("Full materialization requires an explicit start datetime.")
        end = datetime.now(timezone.utc).isoformat(timespec="seconds")
        repo_dir = feature_store.parent
        _run_feast_cmd(
            [
                "-c",
                str(repo_dir),
                "materialize",
                since.isoformat(timespec="seconds"),
                end,
            ]
        )


def _run_feast_cmd(args: Sequence[str]) -> None:
    cmd = ["feast", *args]
    env = os.environ.copy()
    # Inject the venv bin directory into PATH so `feast` is found even without activation.
    python_bin = Path(sys.executable)
    env["PATH"] = f"{python_bin.parent}:{env.get('PATH', '')}"
    existing = env.get("PYTHONPATH", "")
    repo_path = str(REPO_ROOT)
    if existing:
        env["PYTHONPATH"] = f"{repo_path}:{existing}"
    else:
        env["PYTHONPATH"] = repo_path
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("Feast CLI not found on PATH. Install `feast` to continue.") from exc


def _infer_default_season(today: Optional[datetime] = None) -> str:
    ref = today or datetime.now(timezone.utc)
    year = ref.year
    start_year = year if ref.month >= 10 else year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh NBA datasets and (optionally) materialize into Feast.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Season identifiers to refresh (e.g., 2024-25). Defaults to inferred current season.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Targets for which to export gold datasets.",
    )
    parser.add_argument(
        "--no-gold",
        action="store_true",
        help="Skip gold dataset creation (only build matches/silver).",
    )
    parser.add_argument(
        "--skip-feast-apply",
        action="store_true",
        help="Do not run `feast apply` after building datasets.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Run `feast materialize-incremental` after apply.",
    )
    parser.add_argument(
        "--materialize-days",
        type=int,
        default=30,
        help="Lookback window (in days) for incremental materialization.",
    )
    parser.add_argument(
        "--feature-store",
        default=str(FEATURE_STORE_PATH),
        help="Path to Feast feature_store.yaml.",
    )
    args = parser.parse_args(argv)

    seasons = args.seasons or [_infer_default_season()]
    artifacts = refresh_and_build(
        seasons=seasons,
        targets=args.targets,
        build_gold=not args.no_gold,
    )

    print(f"Refreshed seasons: {', '.join(artifacts.seasons)}")
    if artifacts.match_path:
        print(f"  Bronze match dataset -> {artifacts.match_path}")
    if artifacts.silver_path:
        print(f"  Silver dataset -> {artifacts.silver_path}")
    if artifacts.gold_paths:
        for target, path in artifacts.gold_paths.items():
            print(f"  Gold ({target}) -> {path}")

    feature_store = Path(args.feature_store)
    if not args.skip_feast_apply:
        run_feast_apply(feature_store=feature_store)
        if args.materialize:
            since = datetime.now(timezone.utc) - timedelta(days=args.materialize_days)
            run_feast_materialize(
                since=since,
                feature_store=feature_store,
                incremental=True,
            )


def _sync_export(source: Optional[Path], destination: Path) -> None:
    if source is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = destination if destination.suffix else destination.with_suffix(".parquet")
    src = Path(source)
    if src.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(src)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        if "GAME_ID" in df.columns:
            df["match_id"] = df["GAME_ID"].astype(str)
        df.to_parquet(target, index=False)
    elif src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        if "GAME_ID" in df.columns:
            df["match_id"] = df["GAME_ID"].astype(str)
        df.to_parquet(target, index=False)
    else:
        shutil.copy2(src, target)


if __name__ == "__main__":
    main()
