from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pandas as pd

from src.config import COLS_ODDS, MATCH_ALLOWED_BASE_COLUMNS, MATCH_ALLOWED_PREFIXES


@dataclass
class ValidationResult:
    stage: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


def validate_match_dataset(df: pd.DataFrame, *, stage: str = "matches") -> ValidationResult:
    result = ValidationResult(stage=stage)
    required_cols = [
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "HOME_IS_WIN",
        "AWAY_IS_WIN",
        "HOME_POINTS_FOR",
        "AWAY_POINTS_FOR",
        "POINT_DIFF",
        "POINT_TOTAL",
        "IS_WIN",
    ]
    _ensure_columns(df, required_cols, result)
    if result.issues:
        return result

    result.stats.update(
        {
            "rows": len(df),
            "games": df["GAME_ID"].nunique(),
            "seasons": df["SEASON"].nunique() if "SEASON" in df.columns else None,
        }
    )

    duplicate_games = df.duplicated(["GAME_ID"]).sum()
    if duplicate_games:
        result.issues.append(f"{duplicate_games} duplicate GAME_ID rows detected.")

    same_teams = (df["HOME_TEAM_ID"] == df["AWAY_TEAM_ID"]).sum()
    if same_teams:
        result.issues.append(f"{same_teams} rows have identical HOME/AWAY team IDs.")

    _check_relation(df, "HOME_POINTS_FOR", "AWAY_POINTS_FOR", "POINT_TOTAL", result, "POINT_TOTAL mismatch")
    _check_relation(df, "HOME_POINTS_FOR", "AWAY_POINTS_FOR", "POINT_DIFF", result, "POINT_DIFF mismatch", diff=True)

    if not ((df["HOME_IS_WIN"] == df["IS_WIN"]).all() and (df["AWAY_IS_WIN"] == 1 - df["IS_WIN"]).all()):
        result.issues.append("HOME_IS_WIN / AWAY_IS_WIN inconsistent with IS_WIN.")

    return result


def validate_silver_dataset(df: pd.DataFrame) -> ValidationResult:
    result = validate_match_dataset(df, stage="silver")
    _check_unexpected_columns(df, result, treat_as_issue=False)
    return result


def validate_gold_dataset(df: pd.DataFrame) -> ValidationResult:
    result = validate_match_dataset(df, stage="gold")
    _check_unexpected_columns(df, result, treat_as_issue=True)
    odds_present = [col for col in COLS_ODDS if col in df.columns]
    if odds_present:
        result.issues.append(f"Gold dataset still contains odds columns: {odds_present}")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame, required: Sequence[str], result: ValidationResult) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        result.issues.append(f"Missing required columns: {missing}")


def _check_relation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    target_col: str,
    result: ValidationResult,
    message: str,
    *,
    diff: bool = False,
) -> None:
    if any(col not in df.columns for col in (col_a, col_b, target_col)):
        return
    computed = df[col_a] - df[col_b] if diff else df[col_a] + df[col_b]
    mismatch = (computed - df[target_col]).abs().gt(1e-6).sum()
    if mismatch:
        result.issues.append(f"{message}: {mismatch} rows.")


def _check_unexpected_columns(df: pd.DataFrame, result: ValidationResult, *, treat_as_issue: bool) -> None:
    allowed = set(MATCH_ALLOWED_BASE_COLUMNS)
    unexpected = [
        col
        for col in df.columns
        if col not in allowed and not col.startswith(MATCH_ALLOWED_PREFIXES)
    ]
    if unexpected:
        msg = f"{len(unexpected)} columns fall outside the allowed match prefixes."
        if treat_as_issue:
            result.issues.append(msg + f" Examples: {unexpected[:5]}")
        else:
            result.warnings.append(msg + f" Examples: {unexpected[:5]}")
