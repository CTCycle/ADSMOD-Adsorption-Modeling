# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from server.common.units import UnitRegistry, parse_number
from server.domain.canonical import normalize_experiment
from server.common.utils.logger import logger
from server.common.identity import is_inchi_key, stable_material_key


###############################################################################
def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError, ValueError:
        pass
    return str(value).strip()


# NIST gives species without a public InChIKey a placeholder such as
# "NIST-ISODB-...". It identifies the NIST record but is not an InChIKey.

# NIST ISODB reports mixture composition as gas-phase mole fractions; an
# absent compositionType follows that convention.
COMPOSITION_TYPES = {"", "molefraction"}


###############################################################################
def _kelvin(value: object) -> float:
    temperature = parse_number(value, ".")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"NIST temperature must be a positive number of kelvin (got {temperature})."
        )
    return temperature


###############################################################################
def _inchi_key(value: str) -> str | None:
    return value if is_inchi_key(value) else None


###############################################################################
class NISTCanonicalMapper:
    """Map known NIST provider frames to canonical persistence records."""

    # -------------------------------------------------------------------------
    def material_records(
        self, frame: pd.DataFrame | None, kind: str
    ) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        records: list[dict[str, Any]] = []
        for row in frame.where(frame.notna(), None).to_dict(orient="records"):
            name = _text(row.get("name"))
            if not name:
                continue
            external = _text(
                row.get("InChIKey") if kind == "adsorbate" else row.get("hashkey")
            )
            records.append(
                {
                    "key": stable_material_key(kind, name, external or None),
                    "name": name,
                    **(
                        {
                            "inchi_key": _inchi_key(external),
                            "nist_identifier": external or None,
                            "inchi": _text(row.get("inchi")) or None,
                        }
                        if kind == "adsorbate"
                        else {"external_identifier": external or None}
                    ),
                    "formula": _text(row.get("molecular_formula")) or None,
                    "molar_mass_g_mol": (
                        float(row["molecular_weight"])
                        if row.get("molecular_weight") is not None
                        else None
                    ),
                    "smiles": _text(row.get("smile_code")) or None,
                }
            )
        return records

    # -------------------------------------------------------------------------
    @staticmethod
    def attach_synonyms(
        records: list[dict[str, Any]],
        payloads: list[dict[str, Any]],
        payload_identifier: str,
        record_identifier: str,
    ) -> None:
        """Keep the synonyms NIST publishes so later uploads can match them."""
        synonyms = {
            _text(payload.get(payload_identifier)): payload.get("synonyms")
            for payload in payloads
        }
        for record in records:
            values = synonyms.get(_text(record.get(record_identifier)))
            if isinstance(values, list):
                record["synonyms"] = [_text(value) for value in values if _text(value)]

    # -------------------------------------------------------------------------
    @staticmethod
    def attach_source_identities(
        records: list[dict[str, Any]], payloads: list[dict[str, Any]]
    ) -> None:
        """Copy NIST identifiers and DOI from raw isotherm payloads to records.

        The flattened measurement frames drop the adsorbent hashkey, adsorbate
        InChIKeys, and DOI; without them species and materials would be keyed
        by display name and duplicate the NIST catalogue rows.
        """
        by_filename = {
            _text(payload.get("filename")).casefold(): payload
            for payload in payloads
            if _text(payload.get("filename"))
        }
        for record in records:
            payload = by_filename.get(_text(record["external_key"]).casefold())
            if payload is None:
                continue
            adsorbent = payload.get("adsorbent")
            hashkey = (
                _text(adsorbent.get("hashkey")) if isinstance(adsorbent, dict) else ""
            )
            if hashkey:
                record["adsorbent"]["external_identifier"] = hashkey
                record["adsorbent"]["name"] = (
                    _text(adsorbent.get("name")) or record["adsorbent"]["name"]
                )
            inchi_keys = {
                _text(item.get("name")).casefold(): _text(item.get("InChIKey"))
                for item in payload.get("adsorbates") or []
                if isinstance(item, dict)
            }
            for adsorbate in record["adsorbates"]:
                identifier = inchi_keys.get(_text(adsorbate["name"]).casefold())
                if identifier:
                    adsorbate["key"] = stable_material_key(
                        "adsorbate", adsorbate["name"], identifier
                    )
                    adsorbate["inchi_key"] = _inchi_key(identifier)
            doi = _text(payload.get("DOI"))
            if doi:
                record["provenance"]["doi"] = doi

    # -------------------------------------------------------------------------
    def experiment_records(
        self, measurements: pd.DataFrame | None
    ) -> list[dict[str, Any]]:
        records, _skipped = self.experiment_records_with_skips(measurements)
        return records

    # -------------------------------------------------------------------------
    def experiment_records_with_skips(
        self, measurements: pd.DataFrame | None
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Map the long NIST frame to canonical experiments, one per source id.

        Experiments the canonical model cannot represent (unsupported units or
        composition type, repeated pressures of one species) are skipped with a
        recorded reason instead of being altered.
        """
        records: list[dict[str, Any]] = []
        skipped: dict[str, str] = {}
        if measurements is None or measurements.empty:
            return records, skipped
        for name, rows in measurements.groupby("name", sort=False):
            source_id = str(name)
            try:
                records.append(
                    normalize_experiment(self._experiment(source_id, rows)).model_dump()
                )
            except ValueError as exc:
                logger.warning("Skipping NIST experiment %s: %s", source_id, exc)
                skipped[source_id] = str(exc) or "Unsupported NIST measurement data."
        return records, skipped

    # -------------------------------------------------------------------------
    @staticmethod
    def _experiment(source_id: str, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        pressure_unit = _text(first["pressure_units"])
        uptake_unit = _text(first["adsorption_units"])
        species = list(dict.fromkeys(_text(value) for value in rows["adsorbate"]))
        if len(species) > 1:
            composition = (
                _text(first.get("composition_type")).casefold().replace(" ", "")
            )
            if composition not in COMPOSITION_TYPES:
                raise ValueError(
                    f"Composition type '{first.get('composition_type')}' is not a "
                    "gas-phase mole fraction."
                )
        molar_mass = (
            float(first.get("adsorbate_molecular_weight"))
            if len(species) == 1 and _text(first.get("adsorbate_molecular_weight"))
            else None
        )
        observations: list[dict[str, Any]] = []
        for row in rows.itertuples(index=False):
            pressure_value = parse_number(row.pressure, ".")
            uptake_value = parse_number(row.adsorbed_amount, ".")
            pressure = UnitRegistry.convert_pressure(
                pressure_value, pressure_unit, "absolute"
            )
            uptake = UnitRegistry.convert_uptake(uptake_value, uptake_unit, molar_mass)
            fraction = (
                parse_number(row.gas_mole_fraction, ".")
                if len(species) > 1 and _text(row.gas_mole_fraction)
                else None
            )
            observations.append(
                {
                    "adsorbate": _text(row.adsorbate),
                    "source_row": int(row.point_index) + 1,
                    "pressure_original": pressure_value,
                    "pressure_original_unit": pressure.original_unit,
                    "pressure_canonical": pressure.canonical_value,
                    "pressure_canonical_unit": pressure.canonical_unit,
                    "uptake_original": uptake_value,
                    "uptake_original_unit": uptake.original_unit,
                    "uptake_mol_kg": uptake.canonical_value,
                    "gas_mole_fraction": fraction,
                    "conversion_metadata": {
                        "pressure": pressure.rule,
                        "uptake": uptake.rule,
                        "source": "NIST ISODB known schema",
                    },
                }
            )
        temperature = _kelvin(first["temperature"])
        return {
            "external_key": source_id,
            "name": source_id,
            "adsorbent": {"name": _text(first["adsorbent"])},
            "adsorbates": [
                {"name": name, "molar_mass_g_mol": molar_mass} for name in species
            ],
            "temperature_original": temperature,
            "temperature_original_unit": "K",
            "temperature_k": temperature,
            "pressure_basis": "absolute",
            "provenance": {
                "repository": "NIST ISODB",
                "source_identifier": source_id,
            },
            "observations": observations,
        }
