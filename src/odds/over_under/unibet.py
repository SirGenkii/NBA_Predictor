from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dateutil import tz

from .true_odds import true_odds_from_market

# ---------------------------------------------------------------------------
# Constantes de configuration
# ---------------------------------------------------------------------------

# Plusieurs endpoints Kambi fonctionnent selon la marque/pays.
# ubfr renvoie souvent 400 via eu1.offering-api, donc on essaie aussi
# les endpoints internationaux accessibles sans login/geo.
KAMBI_BASES = [
    "https://eu-offering-api.kambicdn.com/offering/v2018",
    "https://eu1.offering-api.kambicdn.com/offering/v2018",
]
BRANDS = [
    "unibetfr"
    "ubfr",  # Unibet France (renvoie 400 actuellement)
    "ubuk",  # Unibet UK
    "ubde",
    "ubdk",
    "ubse",
]

NBA_SPORT_PATHS_CANDIDATES = [
    "basketball/nba/all/all/matches.json",
    "basketball/nba/all/matches.json",
    "basketball/nba/matches.json",
]

COMMON_QS: Dict[str, Any] = {
    "lang": "fr_FR",
    "market": "FR",
    "client_id": "2",
    "channel_id": "1",
    "useCombined": "true",
    "includeParticipants": "true",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://www.unibet.fr",
    "Referer": "https://www.unibet.fr/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SelectionOdds:
    name: Optional[str]
    odds_decimal: Optional[float]
    odds_fractional: Optional[str]
    popularity: Optional[float] = None
    true_probability_or: Optional[float] = None
    true_odds_or: Optional[float] = None
    true_probability_mpo: Optional[float] = None
    true_odds_mpo: Optional[float] = None
    book_overround: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "odds_decimal": self.odds_decimal,
            "odds_fractional": self.odds_fractional,
            "popularity": self.popularity,
            "true_probability_or": self.true_probability_or,
            "true_odds_or": self.true_odds_or,
            "true_probability_mpo": self.true_probability_mpo,
            "true_odds_mpo": self.true_odds_mpo,
            "book_overround": self.book_overround,
            "raw": self.raw,
        }


@dataclass
class MarketSnapshot:
    event_id: str
    market_id: str
    event_name: Optional[str]
    competition: Optional[str]
    market_name: Optional[str]
    start_utc: Optional[datetime]
    local_start: Optional[datetime]
    will_be_live: Optional[bool]
    total_markets: Optional[int]
    selections: List[SelectionOdds]
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unibet"
    brand: Optional[str] = None  # quelle marque Unibet (ubuk, ubfr...) a été utilisée
    base_url: Optional[str] = None  # endpoint Kambi utilisé
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _iso(value: Optional[datetime]) -> Optional[str]:
            return value.isoformat() if value else None

        return {
            "event_id": self.event_id,
            "market_id": self.market_id,
            "event_name": self.event_name,
            "competition": self.competition,
            "market_name": self.market_name,
            "start_utc": _iso(self.start_utc),
            "local_start": _iso(self.local_start),
            "will_be_live": self.will_be_live,
            "total_markets": self.total_markets,
            "selections": [sel.to_dict() for sel in self.selections],
            "captured_at": _iso(self.captured_at),
            "source": self.source,
            "brand": self.brand,
            "base_url": self.base_url,
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# Helpers génériques
# ---------------------------------------------------------------------------

async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    r = await client.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def _as_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Kambi en ms
        ts = value / 1000.0 if value > 1e10 else float(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _fractional_to_decimal(up: Any, down: Any) -> Optional[float]:
    try:
        up_val = float(up)
        down_val = float(down)
        if down_val <= 0:
            return None
        return round(up_val / down_val + 1.0, 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _extract_fractional_from_outcome(
    outcome: Dict[str, Any],
) -> tuple[Optional[str], Optional[float]]:
    """
    Essaie de récupérer (\"up/down\", decimal) à partir d'un outcome Kambi/Unibet.
    Gère à la fois les champs oddsUp/oddsDown et currentPriceUp/currentPriceDown.
    """
    up = outcome.get("oddsUp") or outcome.get("currentPriceUp")
    down = outcome.get("oddsDown") or outcome.get("currentPriceDown")
    if up is not None and down is not None:
        frac = f"{up}/{down}"
        return frac, _fractional_to_decimal(up, down)

    frac_raw = outcome.get("oddsFractional")
    if isinstance(frac_raw, str) and "/" in frac_raw:
        up_str, down_str = frac_raw.split("/", 1)
        return frac_raw, _fractional_to_decimal(up_str, down_str)

    return None, None


def _tournament_path(ev: Dict[str, Any]) -> str:
    path = ev.get("path") or []
    if isinstance(path, list) and path:
        return " > ".join(
            node.get("name", "") for node in path if isinstance(node, dict)
        )
    return (ev.get("event") or {}).get("group", "")


async def _list_direct_matches(
    client: httpx.AsyncClient,
    path_candidates: List[str],
) -> tuple[List[Dict[str, Any]], Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    Essaie d'abord avec ubfr, sinon bascule sur d'autres marques Unibet accessibles.
    Teste toutes les combinaisons base/brand/path et renvoie aussi le log des tentatives.
    Retourne (events, base, brand, attempts).
    """
    attempts: List[Dict[str, Any]] = []
    chosen_events: List[Dict[str, Any]] = []
    chosen_base: Optional[str] = None
    chosen_brand: Optional[str] = None

    for base in KAMBI_BASES:
        for brand in BRANDS:
            for suffix in path_candidates:
                url = f"{base}/{brand}/listView/{suffix}"
                status = None
                events_count = 0
                error: Optional[str] = None
                try:
                    data = await _fetch_json(client, url, COMMON_QS)
                    status = "ok"
                    events = data.get("events") or []
                    events_count = len(events)
                    if events and not chosen_events:
                        chosen_events = events
                        chosen_base = base
                        chosen_brand = brand
                except httpx.HTTPStatusError as exc:
                    status = f"http_{exc.response.status_code}"
                    error = str(exc)
                    if exc.response.status_code >= 500:
                        break
                except httpx.HTTPError as exc:
                    status = "error"
                    error = str(exc)

                attempts.append(
                    {
                        "base": base,
                        "brand": brand,
                        "path": suffix,
                        "status": status,
                        "events": events_count,
                        "error": error,
                    }
                )

    return chosen_events, chosen_base, chosen_brand, attempts


async def _list_nba_events(
    client: httpx.AsyncClient,
) -> Tuple[
    List[Dict[str, Any]], Optional[str], Optional[str], List[Dict[str, Any]]
]:
    events, base, brand, attempts = await _list_direct_matches(
        client, NBA_SPORT_PATHS_CANDIDATES
    )
    return events, base, brand, attempts


# ---------------------------------------------------------------------------
# NBA : marchés Over/Under (Total Points)
# ---------------------------------------------------------------------------

async def fetch_nba_totals_for_day(
    *,
    days_ahead: int = 0,
    tz_name: str = "Europe/Copenhagen",
    fallback_to_next_available: bool = True,
    debug_probe_brands: bool = False,
) -> List[MarketSnapshot]:
    """
    Récupère les marchés Over/Under (Total Points) NBA pour un jour donné.

    - days_ahead=0 : aujourd'hui
    - days_ahead=1 : demain
    etc.
    - fallback_to_next_available=True : si aucun match trouvé ce jour-là,
      on prend la prochaine journée disponible dans le flux Kambi (utile
      hors-saison).
    - debug_probe_brands=True : affiche le détail des tests base/brand/path
      pour comprendre pourquoi certaines marques ne renvoient pas d'événements.
    """
    local_tz = tz.gettz(tz_name)
    now_local = datetime.now(local_tz)
    start_local = (now_local + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    results: List[MarketSnapshot] = []

    async with httpx.AsyncClient(headers=HEADERS, http2=True) as client:
        events, base_url, brand, attempts = await _list_nba_events(client)

        if debug_probe_brands:
            for att in attempts:
                base_short = att["base"].replace("https://", "")
                print(
                    f"[probe] base={base_short} brand={att['brand']} path={att['path']} "
                    f"status={att['status']} events={att['events']}"
                    + (f" error={att['error']}" if att["error"] else "")
                )

        if not events or not base_url or not brand:
            return []

        parsed_events = []
        for ev in events:
            event_obj = ev.get("event") or ev
            eid = event_obj.get("id") or ev.get("id")
            if not eid:
                continue
            dt = _as_utc(
                event_obj.get("start")
                or ev.get("start")
                or event_obj.get("openDate")
                or ev.get("openDate")
            )
            parsed_events.append((dt, ev, event_obj, eid))

        async def process_window(
            win_start_utc: datetime, win_end_utc: datetime
        ) -> List[MarketSnapshot]:
            window_results: List[MarketSnapshot] = []
            for dt, ev, event_obj, eid in parsed_events:
                if not dt or not (win_start_utc <= dt < win_end_utc):
                    continue

                event_name = event_obj.get("name", "")
                competition = _tournament_path(ev)

                bet_url = f"{base_url}/{brand}/betoffer/event/{eid}.json"
                try:
                    bet_data = await _fetch_json(client, bet_url, COMMON_QS)
                except httpx.HTTPError:
                    continue

                bet_offers = bet_data.get("betOffers") or []
                if not bet_offers:
                    continue

                for offer in bet_offers:
                    offer_label = (
                        offer.get("name")
                        or (offer.get("criterion") or {}).get("label")
                        or ""
                    )
                    offer_name = offer_label.lower()
                    # on cible les marchés de total points
                    if (
                        "over/under" not in offer_name
                        and "total points" not in offer_name
                        and "total de points" not in offer_name
                    ):
                        continue

                    selections_data = offer.get("outcomes") or []
                    selections: List[SelectionOdds] = []

                    for outcome in selections_data:
                        label = (
                            outcome.get("label")
                            or outcome.get("englishLabel")
                            or outcome.get("name")
                        )
                        frac, dec = _extract_fractional_from_outcome(outcome)

                        selections.append(
                            SelectionOdds(
                                name=label,
                                odds_decimal=dec,
                                odds_fractional=frac,
                                popularity=None,
                                raw=outcome,
                            )
                        )

                    valid = [
                        s for s in selections
                        if s.odds_decimal is not None and s.odds_decimal > 1.0
                    ]
                    if len(valid) >= 2:
                        first_sel, second_sel = valid[:2]
                        try:
                            odds_ratio = true_odds_from_market(
                                first_sel.odds_decimal,
                                second_sel.odds_decimal,
                                method="or",
                            )
                            mpo_values = true_odds_from_market(
                                first_sel.odds_decimal,
                                second_sel.odds_decimal,
                                method="mpo",
                            )
                        except ValueError:
                            pass
                        else:
                            for selection, prob_key, odds_key in (
                                (first_sel, "p1", "t1"),
                                (second_sel, "p2", "t2"),
                            ):
                                selection.true_probability_or = odds_ratio[prob_key]
                                selection.true_odds_or = odds_ratio[odds_key]
                                selection.true_probability_mpo = mpo_values[prob_key]
                                selection.true_odds_mpo = mpo_values[odds_key]
                                selection.book_overround = odds_ratio["overround"]

                    snapshot = MarketSnapshot(
                        event_id=str(eid),
                        market_id=str(offer.get("id")),
                        event_name=event_name,
                        competition=competition,
                        market_name=offer_label or offer.get("name"),
                        start_utc=dt,
                        local_start=dt.astimezone(local_tz) if local_tz else dt,
                        will_be_live=None,
                        total_markets=None,
                        selections=selections,
                        brand=brand,
                        base_url=base_url,
                        raw=offer,
                    )
                    window_results.append(snapshot)
            return window_results

        results = await process_window(start_utc, end_utc)

        if not results and fallback_to_next_available:
            future = [p for p in parsed_events if p[0] and p[0] >= start_utc]
            if future:
                first_dt = min(future, key=lambda x: x[0])[0]
                if first_dt:
                    ref_local = first_dt.astimezone(local_tz) if local_tz else first_dt
                    alt_start_local = ref_local.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    alt_end_local = alt_start_local + timedelta(days=1)
                    alt_start_utc = alt_start_local.astimezone(timezone.utc)
                    alt_end_utc = alt_end_local.astimezone(timezone.utc)
                    results = await process_window(alt_start_utc, alt_end_utc)

    return results


# ---------------------------------------------------------------------------
# Wrapper synchrone
# ---------------------------------------------------------------------------

def fetch_nba_totals_for_day_sync(**kwargs: Any) -> List[MarketSnapshot]:
    """
    Wrapper sync pratique pour CLI / scripts.
    Exemple :
        from .unibet_nba import fetch_nba_totals_for_day_sync
        markets = fetch_nba_totals_for_day_sync(days_ahead=0)
    """
    return asyncio.run(fetch_nba_totals_for_day(**kwargs))


__all__ = [
    "SelectionOdds",
    "MarketSnapshot",
    "fetch_nba_totals_for_day",
    "fetch_nba_totals_for_day_sync",
]
