"""Small, dependency-light helpers shared across modules."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

from src.utils.logger import get_logger

logger = get_logger("helpers")


def stable_hash(*parts: Any) -> str:
    """Deterministic short hash -- used for the NL-to-SQL response cache key
    and for rule IDs. `hash()` is not used because it's salted per-process
    in Python and would break caching across restarts."""
    joined = "||".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Usage: `with timed('training'): ...` -- logs elapsed wall time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s took %.2fs", label, elapsed)


def risk_band(probability: float, low_max: float, medium_max: float) -> str:
    """Map a calibrated default probability to a business risk band.

    Bands are configurable (see ModelConfig) rather than hardcoded so a
    credit-policy team can retune them without touching model code.
    """
    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default
