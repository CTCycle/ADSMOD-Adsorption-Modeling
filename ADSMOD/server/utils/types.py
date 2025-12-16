from __future__ import annotations

from collections.abc import Iterable
from typing import Any


###############################################################################
# -------------------------------------------------------------------------
def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


# -------------------------------------------------------------------------
def coerce_int(
    value: Any, default: int, minimum: int | None = None, maximum: int | None = None
) -> int:
    candidate: int
    if isinstance(value, bool):
        candidate = int(value)
    else:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = default
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate


# -------------------------------------------------------------------------
def coerce_float(
    value: Any,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = default
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate


# -------------------------------------------------------------------------
def coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    if value is None:
        return default
    return str(value).strip() or default


# -------------------------------------------------------------------------
def coerce_str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


# -------------------------------------------------------------------------
def coerce_str_sequence(value: Any, default: Iterable[str]) -> tuple[str, ...]:
    items: list[str] = []
    if isinstance(value, str):
        candidates = [
            segment.strip() for segment in value.split(",") if segment.strip()
        ]
    elif isinstance(value, Iterable):
        candidates = []
        for item in value:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    candidates.append(trimmed)
    else:
        candidates = list(default)
    seen: set[str] = set()
    for candidate in candidates or default:
        lowered = candidate.lower()
        if lowered not in seen:
            seen.add(lowered)
            items.append(lowered)
    return tuple(items)
