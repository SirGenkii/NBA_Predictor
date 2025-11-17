from __future__ import annotations

import base64
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dotenv import load_dotenv

from src.config import NBA_MASS_PAYLOAD_DIR, NBA_OCR_OPENAI_API_KEY, NBA_OCR_OPENAI_MODEL
from src.nba_mass_prediction.schemas import MatchPayload, PivotMarket

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

load_dotenv()

PROMPT_TEMPLATE = (
    "Tu es un assistant qui extrait les marchés NBA Over/Under (total points) à partir d'une capture "
    "d'écran Unibet (ou source similaire). La date actuelle est {today}.\n"
    "Réponds exclusivement avec du JSON strictement valide de la forme suivante :\n"
    "{{\n"
    '  "success": bool,\n'
    '  "matches": [\n'
    "    {{\n"
    '      "home_team": string,\n'
    '      "away_team": string,\n'
    '      "match_date": string | null,  // format YYYY-MM-DD, utiliser la date du jour si absente\n'
    '      "markets": [\n'
    "        {{\n"
    '          "label": string | null,\n'
    '          "pivot": float,          // total points (ex: 224.5)\n'
    '          "over_odds": float,      // cotes décimales\n'
    '          "under_odds": float,\n'
    "        }}\n"
    "      ]\n"
    "    }}\n"
    "  ],\n"
    '  "notes": string | null,\n'
    '  "error": string | null\n'
    "}}\n"
    "Contraintes importantes :\n"
    "- Si l'image n'est pas un marché NBA, renvoie success=false et décris le problème dans \"error\".\n"
    "- Convertis les virgules françaises en décimales (224,5 -> 224.5).\n"
    "- Ne rajoute aucun texte hors JSON.\n"
)


class OCRExtractionError(RuntimeError):
    """Raised when OCR extraction fails."""


def _ensure_client(api_key: Optional[str]) -> Any:
    if openai is None:  # pragma: no cover - depends on optional dependency
        raise ImportError("Le package 'openai' est requis pour l'OCR.")
    key = (
        api_key
        or os.getenv("NBA_OCR_OPENAI_API_KEY")
        or NBA_OCR_OPENAI_API_KEY
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise OCRExtractionError("Variable d'environnement NBA_OCR_OPENAI_API_KEY manquante.")
    openai.api_key = key
    return openai


def _decode_json_from_response(text: str) -> Dict[str, Any]:
    json_match = re.search(r"(\{[\s\S]*\})", text)
    if not json_match:
        raise OCRExtractionError("Le modèle n'a pas renvoyé de JSON exploitable.")
    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError as exc:
        raise OCRExtractionError(f"Réponse JSON invalide : {exc}") from exc


def _parse_date(value: Optional[str], fallback: date) -> date:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return fallback


def _parse_float(value: Any) -> float:
    if value is None:
        raise ValueError("Valeur absente")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.replace(",", ".").strip()
        return float(clean)
    raise ValueError(f"Valeur non numérique : {value!r}")


def _build_market(entry: Mapping[str, Any]) -> PivotMarket:
    pivot = _parse_float(entry.get("pivot"))
    over_odds = _parse_float(entry.get("over_odds"))
    under_odds = _parse_float(entry.get("under_odds"))
    label = entry.get("label")
    bookmaker = entry.get("bookmaker")
    metadata = {}
    if "notes" in entry and entry["notes"]:
        metadata["notes"] = entry["notes"]
    if bookmaker:
        metadata["bookmaker"] = bookmaker
    return PivotMarket(
        pivot=pivot,
        over_odds=over_odds,
        under_odds=under_odds,
        label=label or None,
        bookmaker=bookmaker or None,
        metadata=metadata,
    )


def _store_raw_response(
    *,
    screenshot_path: Path,
    raw_text: str,
    reason: str,
) -> Path:
    dump_dir = Path(NBA_MASS_PAYLOAD_DIR) / "ocr_raw" / datetime.now().date().isoformat()
    dump_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    filename = dump_dir / f"{screenshot_path.stem}_{timestamp}.txt"
    payload = {
        "screenshot": str(screenshot_path),
        "reason": reason,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "raw_text": raw_text,
    }
    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return filename


def extract_matches_from_image(
    image_path: Path | str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    today: Optional[date] = None,
) -> Tuple[List[MatchPayload], Dict[str, Any]]:
    """Run OCR via GPT on the provided screenshot and return parsed matches."""
    client = _ensure_client(api_key)
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    today_value = today or datetime.now().date()

    with path.open("rb") as fh:
        payload = base64.b64encode(fh.read()).decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(today=today_value.isoformat())
    chosen_model = model or NBA_OCR_OPENAI_MODEL

    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=chosen_model,
        messages=[
            {"role": "system", "content": "Assistant OCR spécialisé NBA"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
                ],
            },
        ],
        temperature=0,
        max_tokens=4092,
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        parsed = _decode_json_from_response(raw_text)
    except OCRExtractionError as exc:
        dump_path = _store_raw_response(screenshot_path=path, raw_text=raw_text, reason=str(exc))
        raise OCRExtractionError(f"{exc} (dump: {dump_path})") from exc

    if not parsed.get("success"):
        error_message = parsed.get("error") or "OCR non concluant"
        raise OCRExtractionError(error_message)

    matches: List[MatchPayload] = []
    for entry in parsed.get("matches", []):
        if not isinstance(entry, dict):
            continue
        home = str(entry.get("home_team") or "").strip()
        away = str(entry.get("away_team") or "").strip()
        if not home or not away:
            continue

        match_date = _parse_date(entry.get("match_date"), today_value)
        match_time = entry.get("match_time")
        bookmaker = entry.get("bookmaker")
        markets_raw = entry.get("markets") or []
        markets: List[PivotMarket] = []
        for market_entry in markets_raw:
            if not isinstance(market_entry, dict):
                continue
            try:
                market = _build_market({**market_entry, "bookmaker": bookmaker})
            except Exception:
                continue
            markets.append(market)
        if not markets:
            continue

        payload_obj = MatchPayload(
            home_team=home,
            away_team=away,
            match_date=match_date,
            match_time=match_time,
            markets=tuple(markets),
            screenshot_path=path,
            source="ocr",
            raw_payload=entry,
        )
        matches.append(payload_obj)

    if not matches:
        raise OCRExtractionError("Aucun match exploitable détecté dans le screenshot.")

    return matches, parsed


__all__ = ["extract_matches_from_image", "OCRExtractionError"]
