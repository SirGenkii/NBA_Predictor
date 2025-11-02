"""
Structured package for the modernized NBA Predictor codebase.

This module exposes the global ``settings`` object so that downstream code can
import ``from nba_predictor import settings`` without having to know the
internal layout of the configuration package.
"""

from .config.settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
