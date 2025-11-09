"""
CLI utility to build silver/gold datasets from a bronze match-level file.

Usage:
    python -m src.datasets.build --bronze data/01_bronze/matches/latest.parquet --stage all --target IS_WIN
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from .gold import GoldBuildConfig, build_gold_dataset
from .silver import SilverBuildConfig, build_silver_dataset
from .bronze import build_match_dataset_from_bronze

Stage = Literal["silver", "gold", "all"]


def _load_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bronze dataset not found: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format for {path.suffix}. Use CSV or Parquet.")


def _print_result(stage: str, result) -> None:
    path_display = str(result.path) if result.path else "<not saved>"
    print(f"[{stage.upper()}] rows={result.metadata.get('rows')} cols={result.metadata.get('columns')} -> {path_display}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build silver/gold datasets from a bronze match file.")
    parser.add_argument("--matches", type=Path, help="Path to the pre-merged bronze match dataset (CSV or Parquet).")
    parser.add_argument(
        "--build-from-bronze",
        action="store_true",
        help="If set, build the match dataset from the latest bronze CSVs before running silver/gold.",
    )
    parser.add_argument("--bronze-games", type=Path, help="Optional override for the bronze games CSV.")
    parser.add_argument("--bronze-boxscores", type=Path, help="Optional override for the bronze boxscores CSV.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["silver", "gold", "all"],
        help="Which stage(s) to materialize.",
    )
    parser.add_argument(
        "--target",
        default="IS_WIN",
        help="Target name for the gold stage (IS_WIN, POINT_DIFF, POINT_TOTAL, ...).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not persist artifacts to disk (useful for dry runs).",
    )
    args = parser.parse_args()

    if args.build_from_bronze:
        match_result = build_match_dataset_from_bronze(
            games_path=args.bronze_games,
            boxscores_path=args.bronze_boxscores,
            persist_artifact=not args.no_save,
        )
        bronze_df = match_result.dataset
        _print_result("matches", match_result)
    else:
        if not args.matches:
            parser.error("Provide --matches or use --build-from-bronze.")
        bronze_df = _load_matches(args.matches)

    silver_cfg = SilverBuildConfig()
    gold_cfg = GoldBuildConfig(target=args.target)

    if args.no_save:
        silver_cfg.persist_artifact = False
        gold_cfg.persist_artifact = False

    silver_result = build_silver_dataset(bronze_df, config=silver_cfg)
    _print_result("silver", silver_result)

    if args.stage in {"gold", "all"}:
        gold_result = build_gold_dataset(silver_result.dataset, config=gold_cfg)
        _print_result("gold", gold_result)


if __name__ == "__main__":
    main()
