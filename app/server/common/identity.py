# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Pure rules for comparing adsorbate and adsorbent names.

Nothing here touches a database or network. A name comparison can only ever
propose or block a match: the guard signature exists so that names which look
alike but describe different species or samples are never treated as equal.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from server.common.identity_catalog import GENERIC_ADSORBENT_NAMES

FUZZY_REVIEW_THRESHOLD = 0.9
_MIN_TYPO_TOKEN_LENGTH = 6

STATUS_RESOLVED = "resolved"
STATUS_PROVISIONAL = "provisional"
STATUS_GENERIC = "generic"

_INCHI_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−�"), "-")
_NON_ALPHANUMERIC = re.compile(r"[\W_]+", re.UNICODE)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"\d*\.\d+|\d+")
_TRAILING_QUALIFIER = re.compile(r"\s*\([a-z ]+\)$")
_FORMULA = re.compile(r"^(?:\[?[A-Z][a-z]?\d*(?:\.\d+)?\]?)+$")
_METAL = re.compile(
    r"(?<![A-Za-z])(Li|Na|K|Rb|Cs|Be|Mg|Ca|Sr|Ba|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Y|Zr"
    r"|Nb|Mo|Ru|Rh|Pd|Ag|Cd|In|Sn|Hf|Ta|W|Re|Ir|Pt|Au|Hg|Tl|Pb|Bi|Al|Ga|La|Ce|Eu"
    r"|Gd|Tb|Dy|Yb|U)(?![a-z])"
)
_ISOMER_MARKER = re.compile(
    r"(?<![a-z])(?:(ortho|meta|para|cis|trans|iso|neo|sec|tert)(?=[a-z\- ])"
    r"|([ompn])(?=[- ])|([itdlezrs])(?=-)|\(([ezrs+-])\))"
)
_OXIDATION_STATE = re.compile(r"\((i{1,3}|iv|v|vi{1,3}|ix|x)\)")
_CHARGE = re.compile(r"\d?[+-]$")
_COMPOSITE_MARKERS = frozenset("/@:+")
_QUALIFIERS = re.compile(
    r"(?<![a-z])(activated|synthetic|natural|treated|untreated|calcined|washed"
    r"|hydrated|dehydrated|desolvated|solvated|functionali[sz]ed|modified|doped"
    r"|impregnated|guest|humidity|photoswitched|wt|in|on|with)(?![a-z])"
)


###############################################################################
def stable_material_key(kind: str, name: str, external: str | None = None) -> str:
    identity = " ".join((external or name).casefold().split())
    if not identity:
        raise ValueError("Identity values must not be empty.")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{digest}"


###############################################################################
def is_inchi_key(value: str | None) -> bool:
    return bool(value and _INCHI_KEY.fullmatch(value))


###############################################################################
def clean_name(value: str) -> str:
    """Repair encoding and spacing without changing what the name refers to."""
    normalized = unicodedata.normalize("NFKC", value).translate(_DASHES)
    return " ".join(normalized.split())


###############################################################################
def exact_key(value: str) -> str:
    return clean_name(value).casefold()


###############################################################################
def separator_key(value: str) -> str:
    """Comparison form ignoring case, spacing, and punctuation."""
    return _NON_ALPHANUMERIC.sub("", clean_name(value).casefold())


###############################################################################
def guard_signature(value: str) -> tuple[object, ...]:
    """Features that must be equal before two names can be considered alike.

    Metals are read case-sensitively, so 'Cd-MOF-1' and 'CDMOF-1' differ.
    """
    cleaned = clean_name(value)
    folded = cleaned.casefold()
    return (
        tuple(_NUMBER.findall(cleaned)),
        tuple(sorted(set(_METAL.findall(cleaned)))),
        frozenset(cleaned) & _COMPOSITE_MARKERS,
        tuple(sorted({"".join(found) for found in _ISOMER_MARKER.findall(folded)})),
        tuple(_OXIDATION_STATE.findall(folded)),
        tuple(_CHARGE.findall(cleaned)),
        tuple(sorted(set(_QUALIFIERS.findall(folded)))),
    )


###############################################################################
def names_compatible(left: str, right: str) -> bool:
    return guard_signature(left) == guard_signature(right)


###############################################################################
def same_except_separators(left: str, right: str) -> bool:
    return separator_key(left) == separator_key(right) and names_compatible(left, right)


###############################################################################
def fuzzy_score(left: str, right: str) -> float:
    """Similarity in [0, 1] for names that may be misspellings of one another.

    Sample names routinely differ by a short meaningful fragment such as
    '-Me' and '-OMe', so only a single long word may differ, and only slightly.
    """
    if not names_compatible(left, right):
        return 0.0
    if separator_key(left) == separator_key(right):
        return 1.0
    left_tokens = _TOKEN.findall(clean_name(left).casefold())
    right_tokens = _TOKEN.findall(clean_name(right).casefold())
    if len(left_tokens) != len(right_tokens):
        return 0.0
    different = [pair for pair in zip(left_tokens, right_tokens) if pair[0] != pair[1]]
    if len(different) != 1 or min(map(len, different[0])) < _MIN_TYPO_TOKEN_LENGTH:
        return 0.0
    score = SequenceMatcher(None, *different[0]).ratio()
    return score if score >= FUZZY_REVIEW_THRESHOLD else 0.0


###############################################################################
def looks_like_formula(value: str) -> bool:
    """True for element-only strings, which do not identify an isomer by themselves."""
    cleaned = clean_name(value)
    return bool(_FORMULA.fullmatch(cleaned)) and (
        any(character.isdigit() for character in cleaned)
        or sum(character.isupper() for character in cleaned) > 1
    )


###############################################################################
def names_ion_or_oxidation_state(value: str) -> bool:
    """True when a name selects a charge state that a name search may not honour."""
    cleaned = clean_name(value)
    return bool(_OXIDATION_STATE.search(cleaned.casefold()) or _CHARGE.search(cleaned))


###############################################################################
def is_generic_adsorbent(value: str) -> bool:
    """True for a material class name that does not identify one sample."""
    return _TRAILING_QUALIFIER.sub("", exact_key(value)) in GENERIC_ADSORBENT_NAMES


###############################################################################
def adsorbate_record_key(record: dict[str, Any]) -> str:
    return record.get("key") or stable_material_key(
        "adsorbate", clean_name(record["name"]), record.get("inchi_key")
    )


###############################################################################
def adsorbent_record_key(record: dict[str, Any], dataset_id: int | None) -> str:
    """Generic names are keyed per dataset so unrelated samples stay apart."""
    name = clean_name(record["name"])
    external = record.get("external_identifier")
    if not external and is_generic_adsorbent(name):
        return stable_material_key("adsorbent", name, f"dataset:{dataset_id}:{name}")
    return record.get("key") or stable_material_key("adsorbent", name, external)
