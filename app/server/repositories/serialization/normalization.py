from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import hashlib
import json

import pandas as pd

from app.server.common.constants import (
    COLUMN_ADSORBATE,
    COLUMN_ADSORBENT,
    COLUMN_PRESSURE_PA,
    COLUMN_TEMPERATURE_K,
    COLUMN_UPTAKE_MOL_G,
)


# -----------------------------------------------------------------------------
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# -----------------------------------------------------------------------------
def normalize_lower(value: Any) -> str:
    return normalize_text(value).lower()


# -----------------------------------------------------------------------------
def normalize_model_key(model_name: Any) -> str:
    return (
        str(model_name)
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("-", "_")
        .replace(" ", "_")
        .upper()
        .strip("_")
    )


# -----------------------------------------------------------------------------
def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


# -----------------------------------------------------------------------------
def to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [entry.strip() for entry in stripped.split(",") if entry.strip()]
        return [float(entry) for entry in parsed if not pd.isna(entry)]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        values: list[float] = []
        for entry in value:
            parsed = to_float(entry)
            if parsed is None:
                continue
            values.append(parsed)
        return values
    parsed = to_float(value)
    return [parsed] if parsed is not None else []


# -----------------------------------------------------------------------------
def build_processed_key_from_values(
    adsorbent: str,
    adsorbate: str,
    temperature_k: float | None,
    pressure_series: list[float],
    uptake_series: list[float],
) -> str:
    payload = {
        COLUMN_ADSORBENT: adsorbent,
        COLUMN_ADSORBATE: adsorbate,
        COLUMN_TEMPERATURE_K: temperature_k,
        COLUMN_PRESSURE_PA: pressure_series,
        COLUMN_UPTAKE_MOL_G: uptake_series,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
