"""Helpers to remove bookmaker margin from two-way markets."""
from __future__ import annotations

from typing import Dict, Literal


def true_odds_from_market(
    o1: float,
    o2: float,
    method: Literal["or", "mpo"] = "or",
) -> Dict[str, float]:
    """Return true probabilities/odds derived from two decimal odds."""
    if o1 <= 1.0 or o2 <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0 to compute true odds")

    q1 = 1.0 / o1
    q2 = 1.0 / o2
    s = q1 + q2
    if s <= 0:
        raise ValueError("Invalid implied probabilities; sum must be positive")

    overround = s - 1.0

    if abs(overround) < 1e-12:
        p1 = q1 / s
        p2 = q2 / s
    elif method == "or":
        p1 = q1 / s
        p2 = q2 / s
    elif method == "mpo":
        weight_denominator = o1 + o2
        w1 = o1 / weight_denominator
        w2 = o2 / weight_denominator
        p1 = q1 - overround * w1
        p2 = q2 - overround * w2
        if p1 <= 0 or p2 <= 0:
            p1 = q1 / s
            p2 = q2 / s
    else:
        raise ValueError("method must be either 'or' or 'mpo'")

    t1 = 1.0 / p1
    t2 = 1.0 / p2
    return {"p1": p1, "p2": p2, "t1": t1, "t2": t2, "overround": overround}


__all__ = ["true_odds_from_market"]
