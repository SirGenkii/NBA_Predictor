from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.feast.data_sources import FEAST_SILVER_EXPORT
from src.feast.online_store import write_point_total_online_store


def _parse_dates(values: Iterable[str]) -> set[pd.Timestamp]:
    return {pd.to_datetime(value).normalize() for value in values}


def load_silver_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "match_id" not in df.columns:
        df["match_id"] = df.get("GAME_ID", df.index).astype(str)
    df["match_id"] = df["match_id"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def filter_matches(
    df: pd.DataFrame,
    *,
    match_ids: Iterable[str] | None,
    match_dates: Iterable[str] | None,
    since: str | None,
    until: str | None,
    limit: int | None,
) -> pd.DataFrame:
    result = df
    if match_ids:
        ids = {str(mid) for mid in match_ids}
        result = result[result["match_id"].isin(ids)]
    if match_dates:
        targets = _parse_dates(match_dates)
        result = result[result["GAME_DATE"].isin(targets)]
    if since:
        result = result[result["GAME_DATE"] >= pd.to_datetime(since)]
    if until:
        result = result[result["GAME_DATE"] <= pd.to_datetime(until)]
    if limit:
        result = result.sort_values("GAME_DATE").head(limit)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Push silver features to Feast online store.")
    parser.add_argument(
        "--silver-path",
        type=Path,
        default=Path(FEAST_SILVER_EXPORT),
        help="Path to the silver parquet used for Feast.",
    )
    parser.add_argument(
        "--match-id",
        action="append",
        help="Match ID to push. Can be provided multiple times.",
    )
    parser.add_argument(
        "--match-date",
        action="append",
        help="Match date (YYYY-MM-DD) to push. Can be repeated.",
    )
    parser.add_argument("--since", help="Only push matches with GAME_DATE >= this date.")
    parser.add_argument("--until", help="Only push matches with GAME_DATE <= this date.")
    parser.add_argument("--limit", type=int, help="Limit the number of rows pushed.")
    args = parser.parse_args(argv)

    df = load_silver_dataframe(args.silver_path)
    filtered = filter_matches(
        df,
        match_ids=args.match_id,
        match_dates=args.match_date,
        since=args.since,
        until=args.until,
        limit=args.limit,
    )
    if filtered.empty:
        print("No matches to push.")
        return

    count = write_point_total_online_store(filtered)
    print(f"Pushed {count} match(es) to Feast online store from {args.silver_path}.")


if __name__ == "__main__":
    main()
