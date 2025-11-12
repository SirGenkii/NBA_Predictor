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
from .schema import build_schema, save_schema
from .validation import (
    validate_gold_dataset,
    validate_match_dataset,
    validate_silver_dataset,
    ValidationResult,
)

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


def _print_validation(result: ValidationResult) -> None:
    status = "OK" if result.ok else "FAILED"
    stats = ", ".join(f"{k}={v}" for k, v in result.stats.items() if v is not None)
    print(f"[VALIDATE:{result.stage.upper()}] {status} {stats}".strip())
    for issue in result.issues:
        print(f"  - {issue}")
    for warning in result.warnings:
        print(f"  * warning: {warning}")


def _write_schema(df: pd.DataFrame, directory: Path, stage: str, fmt: str) -> None:
    report = build_schema(df, stage=stage)
    filename = directory / f"{stage}_schema.{fmt}"
    save_schema(report, filename, fmt=fmt)
    print(f"[SCHEMA:{stage.upper()}] saved to {filename}")


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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run dataset validation checks after each stage.",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        help="Optional directory where schema CSVs will be written (matches/silver/gold).",
    )
    parser.add_argument(
        "--schema-format",
        default="csv",
        choices=["csv", "md"],
        help="Output format for schema snapshots.",
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
        if args.schema_dir:
            _write_schema(match_result.dataset, args.schema_dir, "matches", args.schema_format)
        if args.validate:
            _print_validation(validate_match_dataset(bronze_df, stage="matches"))
    else:
        if not args.matches:
            parser.error("Provide --matches or use --build-from-bronze.")
        bronze_df = _load_matches(args.matches)
        if args.schema_dir:
            _write_schema(bronze_df, args.schema_dir, "matches", args.schema_format)
        if args.validate:
            _print_validation(validate_match_dataset(bronze_df, stage="matches"))

    silver_cfg = SilverBuildConfig()
    gold_cfg = GoldBuildConfig(target=args.target)

    if args.no_save:
        silver_cfg.persist_artifact = False
        gold_cfg.persist_artifact = False

    silver_result = build_silver_dataset(bronze_df, config=silver_cfg)
    _print_result("silver", silver_result)
    if args.schema_dir:
        _write_schema(silver_result.dataset, args.schema_dir, "silver", args.schema_format)
    if args.validate:
        _print_validation(validate_silver_dataset(silver_result.dataset))

    if args.stage in {"gold", "all"}:
        gold_result = build_gold_dataset(silver_result.dataset, config=gold_cfg)
        _print_result("gold", gold_result)
        if args.schema_dir:
            _write_schema(gold_result.dataset, args.schema_dir, "gold", args.schema_format)
        if args.validate:
            _print_validation(validate_gold_dataset(gold_result.dataset))


if __name__ == "__main__":
    main()
