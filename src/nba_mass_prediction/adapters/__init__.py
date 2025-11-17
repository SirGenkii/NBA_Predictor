"""Input adapters (OCR, JSON API, etc.) for the NBA mass prediction engine."""

from .json_api import payloads_from_json
from .ocr_adapter import extract_matches_from_image

__all__ = ["payloads_from_json", "extract_matches_from_image"]
