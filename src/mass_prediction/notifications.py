from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

import requests
from dotenv import load_dotenv

from src.config import (
    DISCORD_ADMIN_ROLE_MENTION,
    DISCORD_FREE_PREDICTION_BOT_TOKEN,
    DISCORD_FREE_PREDICTION_CHANNEL_ID,
    DISCORD_SUBSCRIBER_PREDICTION_BOT_TOKEN,
    DISCORD_SUBSCRIBER_PREDICTION_CHANNEL_ID,
    DISCORD_SUBSCRIBER_ROLE_MENTION,
    DISCORD_WATCHER_DEBUG_BOT_TOKEN,
    DISCORD_WATCHER_DEBUG_CHANNEL_ID,
    MACHINE_IDENTIFIER,
)
from .betting import recommend_bet

if TYPE_CHECKING:  # pragma: no cover
    from .ocr import OCRMatch
    from .runner import ProcessedMatch
else:
    OCRMatch = object
    ProcessedMatch = object

load_dotenv()

LOGGER = logging.getLogger(__name__)

MODEL_KEY_LABELS = {
    "P_WEIGHTED": "Weighted Blend (raw)",
    "P_BLEND_PLATT": "Meta Blend (Platt)",
    "P_BLEND": "Meta Blend (logit)",
    "P_BLEND_ISO": "Meta Blend (Isotonic)",
    "P_XGB": "XGBoost",
    "P_CAT": "CatBoost",
    "P_LGB": "LightGBM",
    "P_HGB": "HistGradientBoosting",
}

WATCHLIST_ODDS_DIFF_THRESHOLD = 0.10
KELLY_DISPLAY_SCALING = 2 / 3
KELLY_DISPLAY_LABEL = "2/3"


def _label_for_model_key(key: str) -> str:
    return MODEL_KEY_LABELS.get(key, key)

PLAIN_MESSAGE_SAFE_LENGTH = 1800


@dataclass
class SummaryItem:
    title: str
    embed_lines: list[str]
    text_lines: list[str]


@dataclass
class FreePickCandidate:
    player_name: str
    opponent_name: str
    player_probability: float
    opponent_probability: float
    player_fair_odds: float
    opponent_fair_odds: float
    player_book_odds: Optional[float]
    opponent_book_odds: Optional[float]
    edge: float
    match_header: str
    match_date_label: str
    tournament: Optional[str]


@dataclass
class SubscriberEntry:
    header: str
    lines: list[str]


@dataclass
class SummaryData:
    bets: list[SummaryItem]
    watch: list[SummaryItem]
    no_bet: list[SummaryItem]
    total_kelly_fraction: float
    total_scaled_fraction: float
    total_matches: int
    subscriber_bets: list[SubscriberEntry]
    subscriber_watch: list[SubscriberEntry]
    subscriber_watch_fallback: list[SubscriberEntry]
    free_pick: Optional[FreePickCandidate]


def _send_discord_embeds(
    token: Optional[str],
    channel_id: Optional[str],
    embeds: list[dict],
    *,
    mentions: Optional[Sequence[str]] = None,
    log_context: str,
) -> bool:
    if not embeds:
        return False

    if not token or not channel_id:
        LOGGER.debug("Discord notification skipped (missing env vars)")
        return False

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    payload = {"embeds": embeds}

    if mentions:
        mention_lines = [m for m in mentions if m]
        if mention_lines:
            payload["content"] = "\n".join(mention_lines)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code >= 400:
            LOGGER.error(
                "Discord notification failed (%s): %s",
                response.status_code,
                response.text,
            )
            return False
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Discord notification error: %s", exc)
        return False

    LOGGER.info("Discord notification sent (%s) [%s]", channel_id, log_context)
    return True


def _format_sent_at() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _build_metadata_description(
    *,
    run_id: Optional[str] = None,
    screenshot_path: Optional[Path] = None,
    extra_lines: Optional[Iterable[str]] = None,
) -> str:
    lines = [f"Envoyé : `{_format_sent_at()}`"]
    if run_id:
        lines.append(f"Run : `{run_id}`")
    if screenshot_path:
        lines.append(f"Screenshot : `{screenshot_path.name}`")
    if extra_lines:
        for line in extra_lines:
            if line:
                lines.append(line)
    lines.append(f"Machine : `{MACHINE_IDENTIFIER}`")
    return "\n".join(lines)


def _prepare_summary_data(matches: Iterable[ProcessedMatch]) -> SummaryData:
    processed_matches = list(matches)
    bets: list[SummaryItem] = []
    watch: list[SummaryItem] = []
    no_bet: list[SummaryItem] = []
    subscriber_bets: list[SubscriberEntry] = []
    subscriber_watch: list[SubscriberEntry] = []
    subscriber_watch_fallback: list[SubscriberEntry] = []
    best_free_pick: Optional[FreePickCandidate] = None
    total_kelly_fraction = 0.0
    total_scaled_fraction = 0.0

    def _format_percentage(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1f}%"

    def _format_odds(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, (int, float)) and not math.isnan(value):
            return f"{value:.2f}"
        return "n/a"

    def _safe_number(value: object) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    def _register_free_candidate(
        *,
        player_prob: Optional[float],
        opponent_prob: Optional[float],
        player_fair: Optional[float],
        opponent_fair: Optional[float],
        player_book: Optional[float],
        opponent_book: Optional[float],
        player_name: str,
        opponent_name: str,
        match_header: str,
        match_date_label: str,
        tournament_label: Optional[str],
    ) -> None:
        nonlocal best_free_pick

        if player_prob is None or player_book is None or player_fair is None:
            return

        edge = player_prob * player_book - 1.0
        opponent_probability = (
            opponent_prob if opponent_prob is not None else max(0.0, 1.0 - player_prob)
        )
        opponent_fair_value = opponent_fair if opponent_fair is not None else max(player_fair, 1.0)

        candidate = FreePickCandidate(
            player_name=player_name,
            opponent_name=opponent_name,
            player_probability=player_prob,
            opponent_probability=opponent_probability,
            player_fair_odds=player_fair,
            opponent_fair_odds=opponent_fair_value,
            player_book_odds=player_book,
            opponent_book_odds=opponent_book,
            edge=edge,
            match_header=match_header,
            match_date_label=match_date_label,
            tournament=tournament_label if tournament_label and tournament_label != "?" else None,
        )

        if best_free_pick is None or edge > best_free_pick.edge:
            best_free_pick = candidate

    for match in processed_matches:
        pred = match.prediction
        req = pred.request
        tournament = (pred.tournament.get("name") if pred.tournament else req.tournament_name) or "?"
        match_date = req.match_date.isoformat() if req.match_date else "?"
        match_time = req.match_time or ""
        match_time_str = str(match_time).strip()
        odds_player1 = match.ocr.odds_player1 if hasattr(match, "ocr") else None
        odds_player2 = match.ocr.odds_player2 if hasattr(match, "ocr") else None
        player_vs = f"{pred.player1.display_name} vs {pred.player2.display_name}"
        header_parts = [player_vs]
        if tournament and tournament != "?":
            header_parts.append(tournament)
        datetime_part = match_date
        if match_time_str:
            datetime_part = f"{match_date} {match_time_str}".strip()
        if datetime_part and datetime_part != "?":
            header_parts.append(datetime_part)
        header_text = " - ".join(part for part in header_parts if part).strip()

        prob_line = f"• Probabilités : {pred.prob_player1 * 100:.1f}% / {pred.prob_player2 * 100:.1f}%"
        fair_line = f"• Cotes justes    : {pred.fair_odds_player1:.2f} / {pred.fair_odds_player2:.2f}"
        if odds_player1 is not None and odds_player2 is not None:
            book_line = f"• Cotes bookmaker : {odds_player1:.2f} / {odds_player2:.2f}"
        else:
            book_line = "• Cotes bookmaker : n/a"
        
        model_line = f"• Modèle : {_label_for_model_key(pred.model_key)} ({pred.model_key})"

        core_lines = [prob_line, fair_line, book_line]
        common_text_lines = [header_text] if header_text else []
        common_text_lines.extend(core_lines)

        prob1_num = _safe_number(pred.prob_player1)
        prob2_num = _safe_number(pred.prob_player2)
        fair1_num = _safe_number(pred.fair_odds_player1)
        fair2_num = _safe_number(pred.fair_odds_player2)
        book1_num = _safe_number(odds_player1)
        book2_num = _safe_number(odds_player2)

        free_match_header = header_text or player_vs
        free_date_label = datetime_part if datetime_part and datetime_part != "?" else match_date

        _register_free_candidate(
            player_prob=prob1_num,
            opponent_prob=prob2_num,
            player_fair=fair1_num,
            opponent_fair=fair2_num,
            player_book=book1_num,
            opponent_book=book2_num,
            player_name=pred.player1.display_name,
            opponent_name=pred.player2.display_name,
            match_header=free_match_header,
            match_date_label=free_date_label or "?",
            tournament_label=tournament,
        )
        _register_free_candidate(
            player_prob=prob2_num,
            opponent_prob=prob1_num,
            player_fair=fair2_num,
            opponent_fair=fair1_num,
            player_book=book2_num,
            opponent_book=book1_num,
            player_name=pred.player2.display_name,
            opponent_name=pred.player1.display_name,
            match_header=free_match_header,
            match_date_label=free_date_label or "?",
            tournament_label=tournament,
        )

        recommendation = recommend_bet(
            pred.prob_player1,
            pred.prob_player2,
            odds_player1,
            odds_player2,
        )

        if recommendation is not None:
            player_name = (
                pred.player1.display_name
                if recommendation["side"] == "player1"
                else pred.player2.display_name
            )
            kelly_line = (
                "• Kelly : "
                f"**{player_name} ({recommendation['fraction'] * 100:.1f}% bankroll)** "
                f"@ {recommendation['odds']:.2f} (edge {recommendation['edge'] * 100:.1f}%)"
            )
            scaled_fraction = recommendation["fraction"] * KELLY_DISPLAY_SCALING
            scaled_line = (
                f"• Kelly ({KELLY_DISPLAY_LABEL}) : "
                f"**{player_name} ({scaled_fraction * 100:.1f}% bankroll)** "
                f"@ {recommendation['odds']:.2f} (edge {recommendation['edge'] * 100:.1f}%)"
            )

            embed_lines = core_lines + [kelly_line, scaled_line, model_line]
            text_lines = list(common_text_lines)
            text_lines.append(kelly_line)
            text_lines.append(scaled_line)

            bets.append(SummaryItem(title=player_vs, embed_lines=embed_lines, text_lines=text_lines))
            total_kelly_fraction += recommendation["fraction"]
            total_scaled_fraction += scaled_fraction

            prob1_pct = pred.prob_player1 * 100
            prob2_pct = pred.prob_player2 * 100
            fair1 = _format_odds(pred.fair_odds_player1)
            fair2 = _format_odds(pred.fair_odds_player2)

            suffix_parts: list[str] = []
            if tournament and tournament != "?":
                suffix_parts.append(tournament)
            date_component = ""
            if match_date and match_date != "?":
                date_component = f"**{match_date}**"
            if match_time_str:
                date_component = f"{date_component} {match_time_str}".strip()
            if date_component:
                suffix_parts.append(date_component)
            suffix = " - ".join(suffix_parts)
            header = (
                "🎾 "
                f"**{pred.player1.display_name} ({prob1_pct:.1f}% : @{fair1}) "
                f"vs {pred.player2.display_name} ({prob2_pct:.1f}% : @{fair2})**"
            )
            if suffix:
                header += f" - {suffix}"

            if odds_player1 is not None and odds_player2 is not None:
                bookmaker_suffix = f"(cotes bookmaker @{odds_player1:.2f} / @{odds_player2:.2f})"
            else:
                bookmaker_suffix = "(cotes bookmaker n/a)"
            edge_line = (
                f"- Edge de {recommendation['edge'] * 100:.1f}% sur {player_name} {bookmaker_suffix}"
            )
            kelly_line_fmt = (
                f"- Kelly : 💵💵 **{player_name}** - **{recommendation['fraction'] * 100:.1f}%** bankroll "
                f"**@{recommendation['odds']:.2f}**"
            )
            scaled_line_fmt = (
                f"- Kelly ({KELLY_DISPLAY_LABEL}) : 💵 **{player_name}** - **{scaled_fraction * 100:.1f}%** "
                f"bankroll **@{recommendation['odds']:.2f}**"
            )

            subscriber_bets.append(
                SubscriberEntry(
                    header=header,
                    lines=[edge_line, kelly_line_fmt, scaled_line_fmt],
                )
            )
            continue

        watch_candidates: list[dict[str, object]] = []
        if book1_num is not None and fair1_num is not None:
            diff_signed = fair1_num - book1_num
            watch_candidates.append(
                {
                    "side": "player1",
                    "abs_diff": abs(diff_signed),
                    "diff_signed": diff_signed,
                    "player_name": pred.player1.display_name,
                    "fair_odds": fair1_num,
                }
            )
        if book2_num is not None and fair2_num is not None:
            diff_signed = fair2_num - book2_num
            watch_candidates.append(
                {
                    "side": "player2",
                    "abs_diff": abs(diff_signed),
                    "diff_signed": diff_signed,
                    "player_name": pred.player2.display_name,
                    "fair_odds": fair2_num,
                }
            )

        watch_focus: dict[str, object] | None = None
        if watch_candidates:
            watch_focus = min(watch_candidates, key=lambda item: item["abs_diff"])

        min_diff = float(watch_focus["abs_diff"]) if watch_focus else None

        if min_diff is None:
            extra_line = "• Cotes bookmaker indisponibles"
            embed_lines = core_lines + [model_line, extra_line]
            text_lines = list(common_text_lines)
            text_lines.extend([model_line, extra_line])
            no_bet.append(SummaryItem(title=player_vs, embed_lines=embed_lines, text_lines=text_lines))
            continue

        diff_display = None
        watch_line_embed: Optional[str] = None
        watch_line_text: Optional[str] = None
        if watch_focus is not None:
            diff_signed = float(watch_focus["diff_signed"])
            diff_display = f"{diff_signed:+.2f}"
            watch_line_embed = (
                "• Bet à surveiller : "
                f"**{watch_focus['player_name']} @{watch_focus['fair_odds']:.2f}** "
                f"({diff_display})"
            )
            watch_line_text = (
                "- Bet à surveiller : "
                f"**{watch_focus['player_name']} @{watch_focus['fair_odds']:.2f}** "
                f"({diff_display})"
            )

        extra_line = f"• Écart min bookmaker vs fair : {min_diff:.2f}"
        diff_line_text = (
            f"- Plus petit écart actuel en cotes justes et bookmaker : **{min_diff:.2f}**"
        )

        prob1_pct = pred.prob_player1 * 100
        prob2_pct = pred.prob_player2 * 100
        fair1 = _format_odds(pred.fair_odds_player1)
        fair2 = _format_odds(pred.fair_odds_player2)

        suffix_parts_watch: list[str] = []
        if tournament and tournament != "?":
            suffix_parts_watch.append(tournament)
        datetime_component = ""
        if match_date and match_date != "?":
            datetime_component = match_date
        if match_time_str:
            datetime_component = f"{datetime_component} {match_time_str}".strip()
        if datetime_component:
            suffix_parts_watch.append(datetime_component)
        suffix_watch = " - ".join(suffix_parts_watch)
        watch_header = (
            "🔍 "
            f"**{pred.player1.display_name} ({prob1_pct:.1f}% : @{fair1}) "
            f"vs {pred.player2.display_name} ({prob2_pct:.1f}% : @{fair2})**"
        )
        if suffix_watch:
            watch_header += f" - {suffix_watch}"
        if book1_num is not None and book2_num is not None:
            bookmaker_line = (
                f"- Cotes des bookmaker défavorable : **{pred.player1.display_name} @{book1_num:.2f}** "
                f"vs **{pred.player2.display_name} @{book2_num:.2f}**"
            )
        else:
            bookmaker_line = "- Cotes des bookmaker défavorable : données indisponibles"

        if min_diff <= WATCHLIST_ODDS_DIFF_THRESHOLD:
            embed_lines = core_lines + [model_line]
            if watch_line_embed:
                embed_lines.append(watch_line_embed)
            embed_lines.append(extra_line)
            text_lines = list(common_text_lines)
            if watch_line_embed:
                text_lines.append(watch_line_embed)
            text_lines.append(extra_line)
            watch.append(SummaryItem(title=player_vs, embed_lines=embed_lines, text_lines=text_lines))
            subscriber_watch.append(
                SubscriberEntry(
                    header=watch_header,
                    lines=[
                        bookmaker_line,
                        watch_line_text or diff_line_text,
                    ],
                )
            )
        else:
            extra_line = f"• Pas de value (écart min {min_diff:.2f})"
            embed_lines = core_lines + [model_line, extra_line]
            text_lines = list(common_text_lines)
            text_lines.extend([model_line, extra_line])
            no_bet.append(SummaryItem(title=player_vs, embed_lines=embed_lines, text_lines=text_lines))
            subscriber_watch_fallback.append(
                SubscriberEntry(
                    header=watch_header,
                    lines=[
                        bookmaker_line,
                        watch_line_text or diff_line_text,
                    ],
                )
            )

    return SummaryData(
        bets=bets,
        watch=watch,
        no_bet=no_bet,
        total_kelly_fraction=total_kelly_fraction,
        total_scaled_fraction=total_scaled_fraction,
        total_matches=len(processed_matches),
        subscriber_bets=subscriber_bets,
        subscriber_watch=subscriber_watch,
        subscriber_watch_fallback=subscriber_watch_fallback,
        free_pick=best_free_pick,
    )


def _build_summary_embeds(
    run_id: str,
    screenshot_path: Path,
    summary: SummaryData,
) -> list[dict]:
    total_matches = summary.total_matches or (len(summary.bets) + len(summary.watch) + len(summary.no_bet))
    if total_matches == 0:
        return []

    def _make_embeds_for_category(
        title: str,
        color: int,
        items: list[SummaryItem],
        extra_fields: list[dict] | None = None,
    ) -> list[dict]:
        embeds_chunked: list[dict] = []
        total = len(items)
        for idx in range(0, total, 25):
            chunk = items[idx : idx + 25]
            embed = {
                "title": f"{title} ({total})" if idx == 0 else f"{title} (suite)",
                "color": color,
                "description": _build_metadata_description(
                    run_id=run_id,
                    screenshot_path=screenshot_path,
                ),
                "fields": [],
                "footer": {"text": f"Matches traités: {total_matches}"},
            }
            for item in chunk:
                embed["fields"].append(
                    {
                        "name": item.title,
                        "value": "\n".join(item.embed_lines)[:1024],
                        "inline": False,
                    }
                )
            if extra_fields and idx == 0:
                embed["fields"].extend(extra_fields)
            embeds_chunked.append(embed)
        return embeds_chunked

    embeds: list[dict] = []
    if summary.bets:
        extra = [
            {
                "name": "Total Kelly recommandé",
                "value": (
                    f"≈ {summary.total_kelly_fraction * 100:.1f}% de la bankroll"
                    f" (et {KELLY_DISPLAY_LABEL}: ~{summary.total_scaled_fraction * 100:.1f}%)"
                ),
                "inline": False,
            }
        ]
        embeds.extend(_make_embeds_for_category("Bets recommandés", 0x2ECC71, summary.bets, extra))
    if summary.watch:
        embeds.extend(_make_embeds_for_category("Matchs à surveiller", 0xF1C40F, summary.watch))
    if summary.no_bet:
        embeds.extend(_make_embeds_for_category("Pas de paris", 0x95A5A6, summary.no_bet))

    return embeds


def send_discord_free_pick(
    run_id: str,
    screenshot_path: Path,
    matches: Iterable[ProcessedMatch],
) -> bool:
    matches = list(matches)
    if not matches:
        LOGGER.debug("No matches provided for free pick run %s", run_id)
        return False

    summary = _prepare_summary_data(matches)
    candidate = summary.free_pick
    if candidate is None:
        LOGGER.debug("No free pick candidate for run %s", run_id)
        return False

    def _fmt_odds(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"@{value:.2f}"

    def _fmt_prob(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    edge_pct = candidate.edge * 100
    edge_emoji = "🟢" if edge_pct > 0 else "🟠"

    header_lines: list[str] = [
        f"🎁 **Free Pick du jour** — {candidate.match_header}",
    ]

    body_lines: list[str] = [
        f"- {edge_emoji} **{candidate.player_name}** vs {candidate.opponent_name}",
        f"- Probabilités : { _fmt_prob(candidate.player_probability)} / {_fmt_prob(candidate.opponent_probability)}",
        f"- Cotes justes : {candidate.player_name} {_fmt_odds(candidate.player_fair_odds)} / {candidate.opponent_name} {_fmt_odds(candidate.opponent_fair_odds)}",
    ]

    if candidate.player_book_odds is not None or candidate.opponent_book_odds is not None:
        body_lines.append(
            "- Cotes bookmaker : "
            f"{candidate.player_name} {_fmt_odds(candidate.player_book_odds)} / "
            f"{candidate.opponent_name} {_fmt_odds(candidate.opponent_book_odds)}"
        )
    else:
        body_lines.append("- Cotes bookmaker : données indisponibles")

    body_lines.append(f"- Edge estimée : {edge_pct:+.1f}%")

    if edge_pct <= 0:
        body_lines.append("⚠️ Pas de value positive aujourd'hui — on partage le meilleur compromis disponible.")

    if candidate.tournament:
        body_lines.append(f"- Tournoi : {candidate.tournament}")
    if candidate.match_date_label and candidate.match_date_label != "?":
        body_lines.append(f"- Horaire estimé : {candidate.match_date_label}")

    content = "\n".join(header_lines + [""] + body_lines).strip()

    return _send_discord_plain_message(
        DISCORD_FREE_PREDICTION_BOT_TOKEN,
        DISCORD_FREE_PREDICTION_CHANNEL_ID,
        content,
    )


def _format_subscriber_message_lines(
    run_id: str,
    screenshot_path: Path,
    summary: SummaryData,
) -> list[str]:
    lines: list[str] = [
        #f"{_format_sent_at()} – Run {run_id}",
        #f"Screenshot : {screenshot_path.name}",
    ]

    total_matches = summary.total_matches or (len(summary.bets) + len(summary.watch) + len(summary.no_bet))

    lines.append("")
    lines.append(f"**Paris EV+ ({len(summary.subscriber_bets)} / {total_matches} matchs)**")
    total_line = (
        f"Total Kelly recommandé : ≈ **{summary.total_kelly_fraction * 100:.1f}%** de la bankroll"
        f" (et **{KELLY_DISPLAY_LABEL}: ~ {summary.total_scaled_fraction * 100:.1f}%**)"
    )
    lines.append(total_line)
    if summary.subscriber_bets:
        for entry in summary.subscriber_bets:
            lines.append("")
            lines.append(entry.header)
            lines.extend(entry.lines)
    else:
        lines.append("")
        lines.append("**Aucun pari EV+ recommandé.**")

    lines.append("")
    watch_entries = summary.subscriber_watch
    if not watch_entries and summary.subscriber_watch_fallback:
        watch_entries = summary.subscriber_watch_fallback
    lines.append(f"**Matchs à surveiller ({len(watch_entries)} / {total_matches})**")
    if watch_entries:
        for entry in watch_entries:
            lines.append("")
            lines.append(entry.header)
            lines.extend(entry.lines)
    else:
        lines.append("")
        lines.append("Aucun match à surveiller pour le moment.")

    return lines


def _chunk_lines(lines: list[str], max_length: int = PLAIN_MESSAGE_SAFE_LENGTH) -> list[str]:
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        if not current_lines and not line:
            continue

        projected_len = len(line) if not current_lines else current_len + 1 + len(line)

        if len(line) > max_length:
            if current_lines:
                chunk = "\n".join(current_lines).strip()
                if chunk:
                    chunks.append(chunk)
                current_lines = []
                current_len = 0
            for start in range(0, len(line), max_length):
                slice_ = line[start : start + max_length]
                if len(slice_) == max_length:
                    chunks.append(slice_)
                else:
                    current_lines = [slice_]
                    current_len = len(slice_)
            continue

        if current_lines and projected_len > max_length:
            chunk = "\n".join(current_lines).strip()
            if chunk:
                chunks.append(chunk)
            current_lines = []
            current_len = 0
            projected_len = len(line)

        current_lines.append(line)
        current_len = projected_len

    if current_lines:
        chunk = "\n".join(current_lines).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _send_discord_plain_message(
    token: Optional[str],
    channel_id: Optional[str],
    content: str,
    *,
    mentions: Optional[Sequence[str]] = None,
) -> bool:
    if not token or not channel_id:
        LOGGER.debug("Discord plain notification skipped (missing env vars)")
        return False

    if mentions:
        mention_lines = [m for m in mentions if m]
        if mention_lines:
            content = "\n".join(mention_lines + [content])

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    payload = {"content": content}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code >= 400:
            LOGGER.error(
                "Discord subscriber notification failed (%s): %s",
                response.status_code,
                response.text,
            )
            return False
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Discord subscriber notification error: %s", exc)
        return False

    LOGGER.info("Discord notification sent (%s) [plain]", channel_id)
    return True

def send_discord_mention(
    token: Optional[str],
    channel_id: Optional[str],
    mention_lines: list[str],
    content: str
) -> bool:

    if _send_discord_plain_message(
        token,
        channel_id,
        content,
        mentions=mention_lines,
    ):
        return True
    return False


def send_discord_debug_summary(
    run_id: str,
    screenshot_path: Path,
    matches: Iterable[ProcessedMatch],
    mention_admin: bool = False,
    mention_subscriber: bool = False,
) -> bool:
    matches = list(matches)
    if not matches:
        return False

    summary = _prepare_summary_data(matches)
    embeds = _build_summary_embeds(run_id, screenshot_path, summary)
    if not embeds:
        return False

    mention_lines: list[str] = []
    if mention_admin:
        mention_lines.append(DISCORD_ADMIN_ROLE_MENTION)
    if mention_subscriber:
        mention_lines.append(DISCORD_SUBSCRIBER_ROLE_MENTION)

    any_sent = False
    for embed in embeds:
        if _send_discord_embeds(
            DISCORD_WATCHER_DEBUG_BOT_TOKEN,
            DISCORD_WATCHER_DEBUG_CHANNEL_ID,
            [embed],
            mentions=[m for m in mention_lines if m],
            log_context="debug_summary",
        ):
            any_sent = True
    return any_sent


def send_discord_subscriber_summary(
    run_id: str,
    screenshot_path: Path,
    matches: Iterable[ProcessedMatch],
    mention_subscribers: bool = True,
) -> bool:
    matches = list(matches)
    if not matches:
        return False

    summary = _prepare_summary_data(matches)
    if not summary.bets and not summary.watch and not summary.subscriber_watch_fallback:
        LOGGER.debug("No subscriber-facing matches for run %s", run_id)
        return False

    lines = _format_subscriber_message_lines(run_id, screenshot_path, summary)
    chunks = _chunk_lines(lines)
    if not chunks:
        LOGGER.debug("Subscriber message empty for run %s", run_id)
        return False

    any_sent = False
    for idx, chunk in enumerate(chunks):
        mention = mention_subscribers and idx == 0
        mention_lines = [DISCORD_SUBSCRIBER_ROLE_MENTION] if mention and DISCORD_SUBSCRIBER_ROLE_MENTION else None
        if _send_discord_plain_message(
            DISCORD_SUBSCRIBER_PREDICTION_BOT_TOKEN,
            DISCORD_SUBSCRIBER_PREDICTION_CHANNEL_ID,
            chunk,
            mentions=mention_lines,
        ):
            any_sent = True
    return any_sent


def send_discord_pending_matches(run_id: str, matches: Iterable[OCRMatch]) -> bool:
    matches = list(matches)
    if not matches:
        return False

    embed = {
        "title": "OCR terminé - matchs détectés",
        "color": 0xF1C40F,
        "description": _build_metadata_description(
            run_id=run_id,
            extra_lines=[f"Matchs détectés : {len(matches)}"],
        ),
        "fields": [],
    }

    for match in matches:
        p1 = getattr(match, "player1", "?")
        p2 = getattr(match, "player2", "?")
        tournament = getattr(match, "tournament", None) or "?"
        match_date = getattr(match, "match_date", None)
        formatted_date = match_date.isoformat() if match_date else "?"
        match_time = getattr(match, "match_time", None)

        line_top = f"**{p1}** vs **{p2}** - {tournament} - {formatted_date} {match_time}"
        #line_top = f"_____________________________________________________"
        #line_bottom_parts = [tournament, formatted_date]
        #if match_time:
        #    line_bottom_parts.append(match_time)
        #line_bottom = " - ".join(filter(None, line_bottom_parts))
        
        
        line_bottom = "-----------------------------------------"

        field = {
            "name": line_top,
            "value": line_bottom,
            "inline": False,
        }
        embed["fields"].append(field)

    return _send_discord_embeds(
        DISCORD_WATCHER_DEBUG_BOT_TOKEN,
        DISCORD_WATCHER_DEBUG_CHANNEL_ID,
        [embed],
        log_context="pending_matches",
    )


def send_discord_engine_startup() -> bool:
    embed = {
        "title": "Initialisation du moteur de prédiction",
        "color": 0x3498DB,
        "description": _build_metadata_description(extra_lines=["Démarrage de l'engine..."]),
    }
    return _send_discord_embeds(
        DISCORD_WATCHER_DEBUG_BOT_TOKEN,
        DISCORD_WATCHER_DEBUG_CHANNEL_ID,
        [embed],
        log_context="engine_start",
    )


__all__ = [
    "send_discord_debug_summary",
    "send_discord_subscriber_summary",
    "send_discord_free_pick",
    "send_discord_pending_matches",
    "send_discord_engine_startup",
]
