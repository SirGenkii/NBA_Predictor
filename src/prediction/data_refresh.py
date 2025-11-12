from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.config import (
    DATA_LAST_BOXSCORES_BATCHES_DIR,
    DATA_LAST_BOXSCORES_BATCHES_MERGED_DIR,
    DATA_LAST_GAMES_DIR,
    DATA_LAST_GAMES_MERGED_DIR,
)
from src.nba_scrapping import (
    download_games_for_seasons,
    get_new_games,
    merge_all_boxscore_stats,
    merge_boxscore_batches_for_season,
    retry_failed_boxscores_for_season,
    scrape_boxscores_v3_for_games,
)
from src.utils import get_latest_file, save_dataframe_to_csv


def refresh_recent_boxscores(seasons: Iterable[str]) -> Optional[str]:
    """
    Mirror notebooks 03/04: download latest games, scrape boxscores, merge with history.

    Args:
        seasons: Iterable of NBA season strings ("2024-25").

    Returns:
        Path to the updated merged boxscore CSV, or None if nothing changed.
    """

    season_list = sorted(set(seasons))
    if not season_list:
        return None

    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    hist_games_path = _safe_latest_file(DATA_LAST_GAMES_MERGED_DIR)
    if hist_games_path is None:
        raise FileNotFoundError(
            f"Aucun historique de matchs trouvé dans {DATA_LAST_GAMES_MERGED_DIR}"
        )
    games_download = download_games_for_seasons(
        season_list, DATA_LAST_GAMES_DIR, run_ts, max_retries=10
    )
    new_games = get_new_games(hist_games_path, games_download)
    if new_games.empty:
        return None

    historical_games = pd.read_csv(hist_games_path, dtype={"GAME_ID": str})
    all_games = pd.concat([historical_games, new_games], ignore_index=True)
    save_dataframe_to_csv(
        all_games,
        DATA_LAST_GAMES_MERGED_DIR,
        prefix="games_merged_all_seasons",
        suffix=run_ts,
    )

    latest_boxscore_hist = _safe_latest_file(DATA_LAST_BOXSCORES_BATCHES_MERGED_DIR)
    hist_box_df = (
        pd.read_csv(latest_boxscore_hist, dtype={"gameId": str})
        if latest_boxscore_hist
        else pd.DataFrame()
    )

    merged_paths = []
    for season in new_games["SEASON"].unique():
        season_df = new_games[new_games["SEASON"] == season]
        season_folder = Path(DATA_LAST_BOXSCORES_BATCHES_DIR) / run_ts / season
        season_folder.mkdir(parents=True, exist_ok=True)
        scrape_boxscores_v3_for_games(season_df, str(season_folder), max_retries=10)
        retry_failed_boxscores_for_season(str(season_folder), max_retries=10)
        merge_boxscore_batches_for_season(str(season_folder))
        merged_dir = season_folder / "merged_batches"
        merged_path = merge_all_boxscore_stats(
            str(merged_dir),
            output_filename=f"final_merged_all_boxscores_{run_ts}.csv",
        )
        if merged_path:
            merged_paths.append(merged_path)

    if not merged_paths:
        return None

    latest_frames = [pd.read_csv(p, dtype={"gameId": str}) for p in merged_paths]
    combined = pd.concat([hist_box_df, *latest_frames], ignore_index=True)
    output_path = (
        Path(DATA_LAST_BOXSCORES_BATCHES_MERGED_DIR)
        / f"all_seasons_boxscores_merged_{run_ts}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return str(output_path)


def _safe_latest_file(directory: str) -> Optional[str]:
    path = Path(directory)
    if not path.exists():
        return None
    files = sorted(path.iterdir())
    if not files:
        return None
    return str(files[-1])
