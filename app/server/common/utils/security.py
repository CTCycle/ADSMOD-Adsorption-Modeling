from __future__ import annotations

import os
import re


SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
CHECKPOINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


# -----------------------------------------------------------------------------
def ensure_safe_sql_identifier(identifier: str, field_name: str = "identifier") -> str:
    normalized = str(identifier).strip()
    if not normalized or not SQL_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid {field_name}.")
    return normalized


# -----------------------------------------------------------------------------
def ensure_safe_checkpoint_name(checkpoint_name: str) -> str:
    normalized = str(checkpoint_name).strip()
    if not normalized or not CHECKPOINT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid checkpoint name.")
    return normalized


# -----------------------------------------------------------------------------
def resolve_checkpoint_path(base_path: str, checkpoint_name: str) -> str:
    safe_name = ensure_safe_checkpoint_name(checkpoint_name)
    absolute_base = os.path.abspath(base_path)
    resolved = os.path.abspath(os.path.join(absolute_base, safe_name))
    if os.path.commonpath([absolute_base, resolved]) != absolute_base:
        raise ValueError("Invalid checkpoint path.")
    return resolved
