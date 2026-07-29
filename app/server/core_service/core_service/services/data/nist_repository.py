from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import func, select

from core_service.services.data.units import UnitRegistry, parse_number
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.datasets import DatasetRepository, stable_material_key
from shared.repositories.schemas.types import normalize_identity
from shared.repositories.materials import MaterialRepository
from shared.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)


NIST_DATASET_PREFIX = "NIST ISODB"


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _material_records(frame: pd.DataFrame, kind: str) -> list[dict[str, Any]]:
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
                    {"inchi_key": external or None}
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


class NISTCanonicalRepository:
    """Deterministic NIST-schema ingestion into the canonical dataset aggregate."""

    def __init__(
        self,
        *,
        database: DatabaseManager,
        datasets: DatasetRepository,
        materials: MaterialRepository,
    ) -> None:
        self.database = database
        self.datasets = datasets
        self.materials = materials

    def list_nist_experiment_ids(self) -> set[str]:
        with self.database.session_factory() as session:
            values = session.scalars(
                select(Isotherm.external_key)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            return {value.casefold() for value in values}

    def list_adsorbate_inchi_keys(self) -> set[str]:
        with self.database.session_factory() as session:
            values = session.scalars(
                select(Adsorbate.inchi_key).where(Adsorbate.inchi_key.is_not(None))
            )
            return {str(value).casefold() for value in values}

    def list_adsorbent_hash_keys(self) -> set[str]:
        with self.database.session_factory() as session:
            values = session.scalars(
                select(Adsorbent.external_identifier).where(
                    Adsorbent.external_identifier.is_not(None)
                )
            )
            return {str(value).casefold() for value in values}

    def count_local_records_by_category(self) -> dict[str, int]:
        with self.database.session_factory() as session:
            experiments = session.scalar(
                select(func.count(Isotherm.id))
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            guests = session.scalar(select(func.count(Adsorbate.id)))
            hosts = session.scalar(select(func.count(Adsorbent.id)))
        return {
            "experiments": int(experiments or 0),
            "guest": int(guests or 0),
            "host": int(hosts or 0),
        }

    def count_nist_rows(self) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(
                    func.count(Observation.id),
                    func.count(func.distinct(Isotherm.id)),
                )
                .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            ).one()
            components = session.scalar(
                select(func.count(IsothermComponent.id))
                .join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            mixture_ids = select(IsothermComponent.isotherm_id).join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id).join(Dataset, Dataset.id == Isotherm.dataset_id).where(Dataset.source == "nist").group_by(IsothermComponent.isotherm_id).having(func.count(IsothermComponent.id) > 1).subquery()
            binary = session.scalar(select(func.count()).select_from(mixture_ids))
        return {
            "single_component_rows": int(rows[0] or 0),
            "binary_mixture_rows": int(binary or 0),
            "isotherm_count": int(rows[1] or 0),
            "component_count": int(components or 0),
        }

    def save_materials_datasets(
        self,
        guest_data: pd.DataFrame | None,
        host_data: pd.DataFrame | None,
    ) -> None:
        if guest_data is not None and not guest_data.empty:
            self.materials.upsert_adsorbates(
                _material_records(guest_data, "adsorbate")
            )
        if host_data is not None and not host_data.empty:
            self.materials.upsert_adsorbents(
                _material_records(host_data, "adsorbent")
            )

    def load_adsorption_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with self.database.session_factory() as session:
            adsorption_rows = session.execute(
                select(
                    Isotherm.external_key,
                    Isotherm.temperature_k,
                    Adsorbent.name.label("adsorbent"),
                    Adsorbate.name.label("adsorbate"),
                    Observation.pressure_original,
                    Observation.pressure_original_unit,
                    Observation.uptake_original,
                    Observation.uptake_original_unit,
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .join(Observation, Observation.isotherm_id == Isotherm.id)
                .join(
                    IsothermComponent,
                    IsothermComponent.id == Observation.component_id,
                )
                .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                .where(Dataset.source == "nist")
                .order_by(Isotherm.id, Observation.sequence_index)
            ).mappings()
            guests = session.execute(select(Adsorbate)).scalars()
            hosts = session.execute(select(Adsorbent)).scalars()
            adsorption = pd.DataFrame([dict(row) for row in adsorption_rows])
            guest = pd.DataFrame(
                [
                    {
                        "name": item.name,
                        "InChIKey": item.inchi_key,
                        "molecular_weight": item.molar_mass_g_mol,
                        "molecular_formula": item.formula,
                        "smile_code": item.smiles,
                    }
                    for item in guests
                ]
            )
            host = pd.DataFrame(
                [
                    {
                        "name": item.name,
                        "hashkey": item.external_identifier,
                        "molecular_weight": item.molar_mass_g_mol,
                        "molecular_formula": item.formula,
                        "smile_code": item.smiles,
                    }
                    for item in hosts
                ]
            )
        return adsorption, guest, host

    @staticmethod
    def _single_experiment(
        source_id: str, rows: pd.DataFrame
    ) -> dict[str, Any]:
        first = rows.iloc[0]
        pressure_unit = _text(first["pressure_units"])
        uptake_unit = _text(first["adsorption_units"])
        temperature = parse_number(first["temperature"], ".")
        adsorbent = _text(first["adsorbent"])
        adsorbate = _text(first["adsorbate"])
        molar_mass = float(first.get("adsorbate_molecular_weight")) if first.get("adsorbate_molecular_weight") not in (None, "") else None
        observations: list[dict[str, Any]] = []
        for index, row in enumerate(rows.itertuples(index=False), start=0):
            pressure_value = parse_number(getattr(row, "pressure"), ".")
            uptake_value = parse_number(getattr(row, "adsorbed_amount"), ".")
            pressure = UnitRegistry.convert_pressure(
                pressure_value, pressure_unit, "absolute"
            )
            uptake = UnitRegistry.convert_uptake(uptake_value, uptake_unit, molar_mass)
            observations.append(
                {
                    "adsorbate": adsorbate,
                    "sequence_index": index,
                    "source_row": index + 1,
                    "pressure_original": pressure_value,
                    "pressure_original_unit": pressure.original_unit,
                    "pressure_canonical": pressure.canonical_value,
                    "pressure_canonical_unit": pressure.canonical_unit,
                    "uptake_original": uptake_value,
                    "uptake_original_unit": uptake.original_unit,
                    "uptake_mol_kg": uptake.canonical_value,
                    "conversion_metadata": {
                        "pressure": pressure.rule,
                        "uptake": uptake.rule,
                        "source": "NIST ISODB known schema",
                    },
                }
            )
        return {
            "external_key": source_id,
            "name": source_id,
            "adsorbent": {"name": adsorbent},
            "adsorbates": [{"name": adsorbate, "molar_mass_g_mol": molar_mass}],
            "temperature_original": temperature,
            "temperature_original_unit": "K",
            "temperature_k": temperature,
            "pressure_basis": "absolute",
            "conditions": {},
            "provenance": {
                "repository": "NIST ISODB",
                "source_identifier": source_id,
            },
            "observations": observations,
        }

    @staticmethod
    def _binary_experiment(
        source_id: str, rows: pd.DataFrame
    ) -> dict[str, Any]:
        first = rows.iloc[0]
        pressure_unit = _text(first["pressure_units"])
        uptake_unit = _text(first["adsorption_units"])
        temperature = parse_number(first["temperature"], ".")
        adsorbent = _text(first["adsorbent_name"])
        species = [_text(first["compound_1"]), _text(first["compound_2"])]
        observations: list[dict[str, Any]] = []
        for point_index, row in rows.reset_index(drop=True).iterrows():
            for position, species_name in enumerate(species, start=1):
                pressure_value = parse_number(
                    row[f"compound_{position}_pressure"], "."
                )
                uptake_value = parse_number(
                    row[f"compound_{position}_adsorption"], "."
                )
                pressure = UnitRegistry.convert_pressure(
                    pressure_value, pressure_unit, "partial"
                )
                uptake = UnitRegistry.convert_uptake(uptake_value, uptake_unit)
                observations.append(
                    {
                        "adsorbate": species_name,
                        "sequence_index": point_index,
                        "source_row": point_index + 1,
                        "pressure_original": pressure_value,
                        "pressure_original_unit": pressure.original_unit,
                        "pressure_canonical": pressure.canonical_value,
                        "pressure_canonical_unit": pressure.canonical_unit,
                        "uptake_original": uptake_value,
                        "uptake_original_unit": uptake.original_unit,
                        "uptake_mol_kg": uptake.canonical_value,
                        "conversion_metadata": {
                            "pressure": pressure.rule,
                            "uptake": uptake.rule,
                            "source": "NIST ISODB known schema",
                        },
                    }
                )
        return {
            "external_key": source_id,
            "name": source_id,
            "adsorbent": {"name": adsorbent},
            "adsorbates": [{"name": name} for name in species],
            "temperature_original": temperature,
            "temperature_original_unit": "K",
            "temperature_k": temperature,
            "pressure_basis": "partial",
            "conditions": {"mixture": True},
            "provenance": {
                "repository": "NIST ISODB",
                "source_identifier": source_id,
            },
            "observations": observations,
        }

    def save_adsorption_datasets(
        self,
        single_component: pd.DataFrame,
        binary_mixture: pd.DataFrame,
        _replace: bool = False,
    ) -> None:
        existing = self.list_nist_experiment_ids()
        with self.database.session_factory() as session:
            collection_id = session.scalar(select(Dataset.id).where(Dataset.source == "nist", Dataset.normalized_name == normalize_identity(NIST_DATASET_PREFIX)))
        if collection_id is None:
            collection_id = self.datasets.persist_canonical(name=NIST_DATASET_PREFIX, source="nist", provenance={"repository": "NIST ISODB", "ingestion": "deterministic known-schema"}, experiments=[])
        grouped: list[tuple[str, pd.DataFrame, str]] = []
        if single_component is not None and not single_component.empty:
            grouped.extend(
                (str(name), rows, "single")
                for name, rows in single_component.groupby("name", sort=False)
            )
        if binary_mixture is not None and not binary_mixture.empty:
            grouped.extend(
                (str(name), rows, "binary")
                for name, rows in binary_mixture.groupby("name", sort=False)
            )
        for source_id, rows, structure in grouped:
            if source_id.casefold() in existing:
                continue
            experiment = (
                self._single_experiment(source_id, rows)
                if structure == "single"
                else self._binary_experiment(source_id, rows)
            )
            self.datasets.persist_canonical(
                name=NIST_DATASET_PREFIX,
                source="nist",
                provenance={
                    "repository": "NIST ISODB",
                    "source_identifier": source_id,
                    "ingestion": "deterministic known-schema",
                },
                experiments=[experiment],
                dataset_id=collection_id,
            )
            existing.add(source_id.casefold())
