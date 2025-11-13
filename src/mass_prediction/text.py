from __future__ import annotations

import unicodedata


def normalize_text(value: str) -> str:
    """Return a lowercase, accent-free identifier for fuzzy name matching."""
    normalized = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    collapsed = " ".join(stripped.replace("-", " ").split())
    return collapsed.lower()


__all__ = ["normalize_text"]
