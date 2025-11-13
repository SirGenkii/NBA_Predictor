from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

try:  # Lazy import so tests without OpenAI SDK can still import the module
    import openai  # type: ignore
except ImportError:  # pragma: no cover - will raise later when used
    openai = None  # type: ignore


load_dotenv()

PROMPT_TEMPLATE = (
    "Tu es un assistant chargé d'extraire les matchs de tennis à partir d'une capture d'écran du site Unibet.\n"
    "La date actuelle est {today}.\n"
    "Réponds exclusivement avec du JSON valide de la forme suivante :\n"
    "{{\n"
    "  \"success\": bool,\n"
    "  \"matches\": [\n"
    "    {{\n"
    "      \"tournament\": string | null,\n"
    "      \"match_date\": string | null,  // format YYYY-MM-DD\n"
    "      \"match_time\": string | null,  // format HH:MM, heure locale extraite si connue\n"
    "      \"player1\": string,\n"
    "      \"player2\": string,\n"
    "      \"odds_player1\": float,\n"
    "      \"odds_player2\": float\n"
    "    }}\n"
    "  ],\n"
    "  \"notes\": string | null,\n"
    "  \"error\": string | null\n"
    "}}\n"
    "Consignes supplémentaires :\n"
    "- Si l'image ne correspond pas à une offre de paris tennis Unibet, renvoie success=false et décris le problème dans 'error'.\n"
    "- Convertis les mentions de date relatives (\"aujourd'hui\", \"demain\", etc.) en format YYYY-MM-DD en utilisant la date actuelle fournie.\n"
    "- Les cotes doivent être des nombres décimaux (utilise un point).\n"
    "- Ne rajoute aucun texte en dehors du JSON.\n"
)


@dataclass
class OCRMatch:
    tournament: Optional[str]
    match_date: Optional[date]
    match_time: Optional[str]
    player1: str
    player2: str
    odds_player1: float
    odds_player2: float
    raw: Dict[str, Any]


class OCRExtractionError(RuntimeError):
    """Raised when the OCR step fails or returns no usable matches."""


def _ensure_client() -> Any:
    if openai is None:  # pragma: no cover - will fail at runtime if SDK missing
        raise ImportError("The 'openai' package is required for OCR extraction.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OCRExtractionError("Environment variable OPENAI_API_KEY is not set.")
    openai.api_key = api_key
    return openai


def _decode_json_from_response(text: str) -> Dict[str, Any]:
    json_match = re.search(r"(\{[\s\S]*\})", text)
    if not json_match:
        raise OCRExtractionError("Le modèle n'a pas renvoyé de JSON exploitable.")
    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on API response
        raise OCRExtractionError(f"Réponse JSON invalide : {exc}") from exc


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _parse_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.replace(",", ".").strip()
        return float(clean)
    raise ValueError(f"Valeur non numérique pour une cote : {value!r}")


def extract_matches_from_image(
    image_path: Path | str,
    *,
    model: str = "gpt-4o-mini",
    today: Optional[date] = None,
) -> Tuple[List[OCRMatch], Dict[str, Any]]:
    """Run OCR via GPT on the provided screenshot and return parsed matches."""
    client = _ensure_client()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    today_value = today or datetime.now().date()

    with path.open("rb") as fh:
        payload = base64.b64encode(fh.read()).decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(today=today_value.isoformat())

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Assistant OCR spécialisé tennis"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
                ],
            },
        ],
        temperature=0,
        max_tokens=2048,
    )

    raw_text = response.choices[0].message.content.strip()
    parsed = _decode_json_from_response(raw_text)

    if not parsed.get("success"):
        error_message = parsed.get("error") or "OCR non concluant"
        raise OCRExtractionError(error_message)

    matches: List[OCRMatch] = []
    for entry in parsed.get("matches", []):
        if not isinstance(entry, dict):
            continue
        try:
            match = OCRMatch(
                tournament=(entry.get("tournament") or None),
                match_date=_parse_date(entry.get("match_date")),
                match_time=(entry.get("match_time") or None),
                player1=str(entry.get("player1") or entry.get("player_1") or "").strip(),
                player2=str(entry.get("player2") or entry.get("player_2") or "").strip(),
                odds_player1=_parse_float(entry.get("odds_player1") or entry.get("odds_player_1")),
                odds_player2=_parse_float(entry.get("odds_player2") or entry.get("odds_player_2")),
                raw=entry,
            )
        except Exception as exc:  # noqa: BLE001
            raise OCRExtractionError(f"Impossible de parser le match OCR : {entry!r}") from exc
        if not match.player1 or not match.player2:
            continue
        matches.append(match)

    if not matches:
        raise OCRExtractionError("Aucun match valable détecté dans le screenshot.")

    return matches, parsed


__all__ = ["OCRMatch", "OCRExtractionError", "extract_matches_from_image"]
