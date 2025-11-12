"""
Feature engineering helpers organized by category (team history, matchup, cleanup, targets, plans).

The modules inside this package operate on the prematch single-row-per-game
representation introduced in the feature roadmap.
"""

from . import availability, cleanup, matchup, odds, plan, reshaping, targets, team  # noqa: F401
