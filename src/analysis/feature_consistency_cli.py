from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .feature_consistency import (
    MatchKey,
    collect_offline_features,
    collect_online_features,
    compare_feature_frames,
    matches_from_artifact,
)


def _parse_manual_match(text: str) -> MatchKey:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Match format attendu: HOME_ID,AWAY_ID,YYYY-MM-DD")
    home, away, date = parts
    return MatchKey(home_team_id=int(home), away_team_id=int(away), match_date=date.strip())


def _load_matches_from_json(paths: Iterable[Path]) -> List[MatchKey]:
    matches: List[MatchKey] = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        matches.extend(matches_from_artifact(data))
    return matches


def run_consistency_check(matches: List[MatchKey], tolerance: float) -> pd.DataFrame:
    offline = collect_offline_features(matches)
    online = collect_online_features(matches)
    return compare_feature_frames(offline, online, tolerance=tolerance)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare offline vs online (Feast) features for matches.")
    parser.add_argument(
        "--run-json",
        action="append",
        type=Path,
        help="Mass prediction run JSON file to extract matches from (can be repeated).",
    )
    parser.add_argument(
        "--match",
        action="append",
        type=_parse_manual_match,
        help="Manual match spec HOME_ID,AWAY_ID,YYYY-MM-DD (can be repeated).",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of matches to compare.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance for column diffs.")
    args = parser.parse_args(argv)

    matches: List[MatchKey] = []
    if args.run_json:
        matches.extend(_load_matches_from_json(args.run_json))
    if args.match:
        matches.extend(args.match)
    if not matches:
        parser.error("Aucun match fourni (via --run-json ou --match).")
    seen = set()
    deduped: List[MatchKey] = []
    for mk in matches:
        key = mk.tuple_key()
        if key in seen:
            continue
        deduped.append(mk)
        seen.add(key)
    if args.limit:
        deduped = deduped[: args.limit]

    diff_df = run_consistency_check(deduped, args.tolerance)
    if diff_df.empty:
        print("Aucun écart détecté entre offline et online pour les matches fournis.")
        return
    pd.set_option("display.max_rows", None)
    print(diff_df)


if __name__ == "__main__":
    main()
