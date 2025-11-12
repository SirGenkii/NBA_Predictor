from __future__ import annotations

import pandas as pd

from src.config import COLS_ODDS


def drop_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bookmaker odds columns when exporting gold datasets."""

    return df.drop(columns=COLS_ODDS, errors="ignore")
