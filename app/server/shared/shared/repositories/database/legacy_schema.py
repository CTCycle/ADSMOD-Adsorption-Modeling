"""Frozen structural checks for adopting the pre-Alembic database."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy.engine import Inspector


BASELINE_REVISION = "23f1110c64a9"
BASELINE_SCHEMA_HASH = "b6b2508ed54b61e4d79aa81d1e4d89837150e52ee56705c09f7661946f139c7a"
BASELINE_TABLES = frozenset(
    {
        "datasets",
        "dataset_imports",
        "adsorbates",
        "adsorbents",
        "isotherms",
        "isotherm_components",
        "observations",
        "fitting_runs",
        "fit_results",
        "fit_parameters",
        "training_datasets",
        "training_samples",
    }
)

###############################################################################
def _expression(value: object) -> str:
    expression = re.sub(r"\s+", " ", str(value).strip().lower())
    expression = re.sub(r'"([a-z_][a-z0-9_]*)"', r"\1", expression)
    expression = re.sub(
        r"::(?:double precision|character varying|timestamp with time zone|"
        r"timestamp without time zone|text|integer|smallint|bigint|numeric|real|"
        r"boolean|[a-z_][a-z0-9_]*)(?:\[\])?",
        "",
        expression,
    )
    expression = re.sub(
        r"= any\s*\(\s*array\s*\[(.*?)\]\s*\)",
        r"in (\1)",
        expression,
    )
    # PostgreSQL may omit parentheses that SQLite retains around a logical
    # sub-expression. Remove only redundant groups containing AND/OR, keeping
    # function and IN-list parentheses intact.
    while True:
        normalized = re.sub(
            r"\(([^()]*\b(?:and|or)\b[^()]*)\)",
            r"\1",
            expression,
        )
        if normalized == expression:
            break
        expression = normalized
    while expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1].strip()
    return expression

###############################################################################
def _type_signature(type_: Any) -> tuple[Any, ...]:
    name = type_.__class__.__name__.lower()
    if name in {"json", "jsonb"}:
        return ("json",)
    if "datetime" in name:
        # SQLite reflects the storage affinity without timezone metadata, but
        # the application type always represents UTC-aware values.
        return ("datetime", True)
    if name in {"string", "varchar", "nvarchar", "text"}:
        return ("string", getattr(type_, "length", None))
    if name in {"integer", "bigint", "smallint"}:
        return ("integer",)
    if name in {"float", "real", "double", "doubleprecision", "double_precision"}:
        return ("float",)
    if name in {"timestamp", "timestamptz", "datetime"}:
        return ("datetime", True)
    return (name, getattr(type_, "length", None))

###############################################################################
def inspector_signature(inspector: Inspector) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for table_name in sorted(BASELINE_TABLES):
        columns = inspector.get_columns(table_name)
        primary_key = tuple(
            inspector.get_pk_constraint(table_name).get("constrained_columns") or ()
        )
        unique_sets = {
            tuple(item.get("column_names") or ())
            for item in inspector.get_unique_constraints(table_name)
        }
        foreign_keys = []
        for item in inspector.get_foreign_keys(table_name):
            columns_for_key = tuple(item.get("constrained_columns") or ())
            options = item.get("options") or {}
            foreign_keys.append(
                {
                    "columns": columns_for_key,
                    "referred_table": item.get("referred_table"),
                    "referred_columns": tuple(item.get("referred_columns") or ()),
                    "ondelete": str(options.get("ondelete") or "").upper(),
                    "onupdate": str(options.get("onupdate") or "").upper(),
                }
            )
        checks = [
            _expression(item.get("sqltext") or "")
            for item in inspector.get_check_constraints(table_name)
        ]
        indexes = [
            (tuple(item.get("column_names") or ()), bool(item.get("unique")))
            for item in inspector.get_indexes(table_name)
            if not item.get("duplicates_constraint")
        ]
        result[table_name] = {
            "columns": [
                {
                    "name": column["name"],
                    "type": _type_signature(column["type"]),
                    "nullable": bool(column["nullable"]),
                    "primary_key": column["name"] in primary_key,
                }
                for column in columns
            ],
            "primary_key": primary_key,
            "unique": sorted(unique_sets),
            "foreign_keys": sorted(foreign_keys, key=repr),
            "checks": sorted(checks),
            "indexes": sorted(indexes),
        }
    return result

###############################################################################
def schema_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

###############################################################################
def schema_matches_baseline(inspector: Inspector) -> bool:
    # A legacy database is adopted only when the complete user-table set is
    # present.  This rejects partial schemas and unrelated tables instead of
    # silently adding the packaged schema around them.
    available = set(inspector.get_table_names()) - {"alembic_version"}
    return (
        available == BASELINE_TABLES
        and schema_hash(inspector_signature(inspector)) == BASELINE_SCHEMA_HASH
    )

###############################################################################
def missing_baseline_tables(inspector: Inspector) -> tuple[str, ...]:
    available = set(inspector.get_table_names())
    return tuple(sorted(BASELINE_TABLES - available))


__all__ = [
    "BASELINE_REVISION",
    "BASELINE_SCHEMA_HASH",
    "BASELINE_TABLES",
    "inspector_signature",
    "missing_baseline_tables",
    "schema_hash",
    "schema_matches_baseline",
]
