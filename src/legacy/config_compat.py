"""
Compatibility layer exposing the legacy constants required by existing notebooks.

New modules should prefer importing from :mod:`nba_predictor.config.settings`.
"""

from src.config import *  # noqa: F401,F403
