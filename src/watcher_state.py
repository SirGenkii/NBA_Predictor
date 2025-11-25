from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.config import NBA_MASS_PREDICTION_DIR

WATCHER_STATE_PATH = Path(NBA_MASS_PREDICTION_DIR) / "watcher_state.json"


def write_state(state: str, **kwargs: Any) -> None:
    """
    Persist the current watcher state into a JSON file so external monitors can inspect it.
    """
    WATCHER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state": state,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    for key, value in kwargs.items():
        if isinstance(value, Mapping):
            payload[key] = {k: v for k, v in value.items() if v is not None}
        elif value is not None:
            payload[key] = value
    WATCHER_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


__all__ = ["write_state", "WATCHER_STATE_PATH"]
