"""Utilities for resolving OCR names to NBA team identifiers."""

from .team_resolver import MatchResolver, TeamResolver, TeamResolverError

__all__ = ["MatchResolver", "TeamResolver", "TeamResolverError"]
