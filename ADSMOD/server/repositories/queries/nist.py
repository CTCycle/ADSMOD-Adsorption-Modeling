from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm import sessionmaker

from ADSMOD.server.common.constants import COLUMN_ADSORBATE, COLUMN_ADSORBENT
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.database.backend import database
from ADSMOD.server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    AdsorptionIsotherm,
    AdsorptionIsothermComponent,
    AdsorptionPoint,
    AdsorptionPointComponent,
    Dataset,
)


###############################################################################
class NISTDataSerializer:
    SINGLE_COMPONENT_UNIQUE_COLUMNS = [
        COLUMN_ADSORBATE,
        COLUMN_ADSORBENT,
        "temperature",
        "pressure",
        "adsorbed_amount",
    ]

    def __init__(self) -> None:
        self.engine = database.backend.engine
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    # -------------------------------------------------------------------------
    @staticmethod
    def _norm(value: object) -> str:
        return str(value or "").strip().lower()

    # -------------------------------------------------------------------------
    def _adsorbate_key(self, inchi_key: str | None, name: str) -> str:
        normalized_inchi = self._norm(inchi_key)
        if normalized_inchi:
            return f"inchi:{normalized_inchi}"
        digest = hashlib.sha1(self._norm(name).encode("utf-8")).hexdigest()[:24]
        return f"name:{digest}"

    # -------------------------------------------------------------------------
    def _adsorbent_key(self, hashkey: str | None, name: str) -> str:
        normalized_hash = self._norm(hashkey)
        if normalized_hash:
            return f"host:{normalized_hash}"
        digest = hashlib.sha1(self._norm(name).encode("utf-8")).hexdigest()[:24]
        return f"name:{digest}"

    # -------------------------------------------------------------------------
    @classmethod
    def deduplicate_single_component_rows(
        cls, single_component: pd.DataFrame
    ) -> pd.DataFrame:
        if single_component.empty:
            return single_component

        normalized = single_component.copy()
        for column in (COLUMN_ADSORBATE, COLUMN_ADSORBENT):
            if column in normalized.columns:
                normalized[column] = (
                    normalized[column].astype("string").str.strip().str.lower()
                )
        for column in ("temperature", "pressure", "adsorbed_amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        missing_columns = [
            column
            for column in cls.SINGLE_COMPONENT_UNIQUE_COLUMNS
            if column not in normalized.columns
        ]
        if missing_columns:
            logger.warning(
                "Skipping NIST single-component deduplication: missing columns %s",
                missing_columns,
            )
            return normalized

        before_count = len(normalized)
        deduplicated = normalized.drop_duplicates(
            subset=cls.SINGLE_COMPONENT_UNIQUE_COLUMNS,
            keep="first",
        ).reset_index(drop=True)
        removed_count = before_count - len(deduplicated)
        if removed_count > 0:
            logger.info(
                "Removed %d duplicate NIST single-component rows before upsert",
                removed_count,
            )
        return deduplicated

    # -------------------------------------------------------------------------
    def _load_single_component_rows(self) -> pd.DataFrame:
        with self.session_factory() as session:
            component_count = (
                select(
                    AdsorptionIsothermComponent.isotherm_id,
                    func.count(AdsorptionIsothermComponent.id).label("component_count"),
                )
                .group_by(AdsorptionIsothermComponent.isotherm_id)
                .subquery()
            )
            rows = session.execute(
                select(
                    AdsorptionIsotherm.source_record_id,
                    AdsorptionIsotherm.temperature_k,
                    AdsorptionIsotherm.adsorption_units,
                    AdsorptionIsotherm.pressure_units,
                    Adsorbent.name,
                    Adsorbate.name,
                    AdsorptionPoint.point_index,
                    AdsorptionPointComponent.original_pressure,
                    AdsorptionPointComponent.original_uptake,
                    AdsorptionPointComponent.partial_pressure_pa,
                    AdsorptionPointComponent.uptake_mol_g,
                )
                .join(Dataset, Dataset.id == AdsorptionIsotherm.dataset_id)
                .join(component_count, component_count.c.isotherm_id == AdsorptionIsotherm.id)
                .join(
                    AdsorptionIsothermComponent,
                    and_(
                        AdsorptionIsothermComponent.isotherm_id == AdsorptionIsotherm.id,
                        AdsorptionIsothermComponent.component_index == 1,
                    ),
                )
                .join(Adsorbate, Adsorbate.id == AdsorptionIsothermComponent.adsorbate_id)
                .join(Adsorbent, Adsorbent.id == AdsorptionIsotherm.adsorbent_id)
                .join(AdsorptionPoint, AdsorptionPoint.isotherm_id == AdsorptionIsotherm.id)
                .join(
                    AdsorptionPointComponent,
                    and_(
                        AdsorptionPointComponent.point_id == AdsorptionPoint.id,
                        AdsorptionPointComponent.component_id == AdsorptionIsothermComponent.id,
                    ),
                )
                .where(and_(Dataset.source == "nist", component_count.c.component_count == 1))
                .order_by(AdsorptionIsotherm.source_record_id, AdsorptionPoint.point_index)
            ).all()

        if not rows:
            return pd.DataFrame()

        records: list[dict[str, object]] = []
        for row in rows:
            pressure = row.original_pressure
            uptake = row.original_uptake
            if pressure is None:
                pressure = row.partial_pressure_pa
            if uptake is None:
                uptake = row.uptake_mol_g
            records.append(
                {
                    "name": row.source_record_id,
                    "temperature": row.temperature_k,
                    "adsorption_units": row.adsorption_units,
                    "pressure_units": row.pressure_units,
                    COLUMN_ADSORBENT: row.name,
                    COLUMN_ADSORBATE: row.name_1,
                    "pressure": pressure,
                    "adsorbed_amount": uptake,
                }
            )
        return pd.DataFrame.from_records(records)

    # -------------------------------------------------------------------------
    def _load_binary_rows(self) -> pd.DataFrame:
        with self.session_factory() as session:
            component_count = (
                select(
                    AdsorptionIsothermComponent.isotherm_id,
                    func.count(AdsorptionIsothermComponent.id).label("component_count"),
                )
                .group_by(AdsorptionIsothermComponent.isotherm_id)
                .subquery()
            )
            rows = session.execute(
                select(
                    AdsorptionIsotherm.source_record_id,
                    AdsorptionIsotherm.temperature_k,
                    AdsorptionIsotherm.adsorption_units,
                    AdsorptionIsotherm.pressure_units,
                    Adsorbent.name,
                    Adsorbate.name,
                    AdsorptionIsothermComponent.component_index,
                    AdsorptionPoint.point_index,
                    AdsorptionPointComponent.original_pressure,
                    AdsorptionPointComponent.original_uptake,
                    AdsorptionPointComponent.partial_pressure_pa,
                    AdsorptionPointComponent.uptake_mol_g,
                )
                .join(Dataset, Dataset.id == AdsorptionIsotherm.dataset_id)
                .join(component_count, component_count.c.isotherm_id == AdsorptionIsotherm.id)
                .join(AdsorptionIsothermComponent, AdsorptionIsothermComponent.isotherm_id == AdsorptionIsotherm.id)
                .join(Adsorbate, Adsorbate.id == AdsorptionIsothermComponent.adsorbate_id)
                .join(Adsorbent, Adsorbent.id == AdsorptionIsotherm.adsorbent_id)
                .join(AdsorptionPoint, AdsorptionPoint.isotherm_id == AdsorptionIsotherm.id)
                .join(
                    AdsorptionPointComponent,
                    and_(
                        AdsorptionPointComponent.point_id == AdsorptionPoint.id,
                        AdsorptionPointComponent.component_id == AdsorptionIsothermComponent.id,
                    ),
                )
                .where(and_(Dataset.source == "nist", component_count.c.component_count == 2))
                .order_by(
                    AdsorptionIsotherm.source_record_id,
                    AdsorptionPoint.point_index,
                    AdsorptionIsothermComponent.component_index,
                )
            ).all()

        if not rows:
            return pd.DataFrame()

        by_point: dict[tuple[str, int], dict[str, object]] = {}
        for row in rows:
            key = (str(row.source_record_id), int(row.point_index))
            entry = by_point.setdefault(
                key,
                {
                    "name": row.source_record_id,
                    "temperature": row.temperature_k,
                    "adsorption_units": row.adsorption_units,
                    "pressure_units": row.pressure_units,
                    "adsorbent_name": row.name,
                },
            )
            pressure = row.original_pressure
            uptake = row.original_uptake
            if pressure is None:
                pressure = row.partial_pressure_pa
            if uptake is None:
                uptake = row.uptake_mol_g
            if int(row.component_index) == 1:
                entry["compound_1"] = row.name_1
                entry["compound_1_pressure"] = pressure
                entry["compound_1_adsorption"] = uptake
                entry["compound_1_composition"] = None
            elif int(row.component_index) == 2:
                entry["compound_2"] = row.name_1
                entry["compound_2_pressure"] = pressure
                entry["compound_2_adsorption"] = uptake
                entry["compound_2_composition"] = None

        return pd.DataFrame.from_records(list(by_point.values()))

    # -------------------------------------------------------------------------
    def load_adsorption_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        adsorption_data = self._load_single_component_rows()
        guest_data = database.load_from_database("adsorbates")
        host_data = database.load_from_database("adsorbents")
        return adsorption_data, guest_data, host_data

    # -------------------------------------------------------------------------
    def count_nist_rows(self) -> dict[str, int]:
        single_rows = self._load_single_component_rows()
        binary_rows = self._load_binary_rows()
        return {
            "single_component_rows": int(len(single_rows)),
            "binary_mixture_rows": int(len(binary_rows)),
            "guest_rows": database.count_rows("adsorbates"),
            "host_rows": database.count_rows("adsorbents"),
        }

    # -------------------------------------------------------------------------
    def save_materials_datasets(
        self,
        guest_data: pd.DataFrame | None = None,
        host_data: pd.DataFrame | None = None,
    ) -> None:
        if isinstance(guest_data, pd.DataFrame) and not guest_data.empty:
            guests = guest_data.copy()
            guests["name"] = guests.get("name", pd.Series(dtype="string")).astype("string").str.strip().str.lower()
            guests["InChIKey"] = guests.get("InChIKey", pd.Series(dtype="string")).astype("string").str.strip()
            guests["adsorbate_key"] = guests.apply(
                lambda row: self._adsorbate_key(row.get("InChIKey"), row.get("name")),
                axis=1,
            )
            expected = {
                "adsorbate_key",
                "InChIKey",
                "name",
                "InChICode",
                "formula",
                "molecular_weight",
                "molecular_formula",
                "smile_code",
            }
            for column in expected:
                if column not in guests.columns:
                    guests[column] = pd.NA
            database.upsert_into_database(guests[list(expected)], "adsorbates")

        if isinstance(host_data, pd.DataFrame) and not host_data.empty:
            hosts = host_data.copy()
            hosts["name"] = hosts.get("name", pd.Series(dtype="string")).astype("string").str.strip().str.lower()
            hosts["hashkey"] = hosts.get("hashkey", pd.Series(dtype="string")).astype("string").str.strip()
            hosts["adsorbent_key"] = hosts.apply(
                lambda row: self._adsorbent_key(row.get("hashkey"), row.get("name")),
                axis=1,
            )
            expected = {
                "adsorbent_key",
                "hashkey",
                "name",
                "formula",
                "molecular_weight",
                "molecular_formula",
                "smile_code",
            }
            for column in expected:
                if column not in hosts.columns:
                    hosts[column] = pd.NA
            database.upsert_into_database(hosts[list(expected)], "adsorbents")

    # -------------------------------------------------------------------------
    def save_adsorption_datasets(
        self, single_component: pd.DataFrame, binary_mixture: pd.DataFrame
    ) -> None:
        single_component = self.deduplicate_single_component_rows(single_component)
        with self.session_factory() as session:
            dataset_entry = session.execute(
                select(Dataset).where(Dataset.dataset_name == "nist")
            ).scalar_one_or_none()
            if dataset_entry is None:
                dataset_entry = Dataset(
                    dataset_name="nist",
                    source="nist",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                session.add(dataset_entry)
                session.flush()

            session.query(AdsorptionIsotherm).filter(
                AdsorptionIsotherm.dataset_id == dataset_entry.id
            ).delete(synchronize_session=False)

            if isinstance(single_component, pd.DataFrame) and not single_component.empty:
                grouped = single_component.groupby(
                    ["name", COLUMN_ADSORBENT, COLUMN_ADSORBATE, "temperature", "pressure_units", "adsorption_units"],
                    dropna=False,
                )
                for (name, adsorbent_name, adsorbate_name, temperature, pressure_units, adsorption_units), frame in grouped:
                    adsorbent = session.execute(
                        select(Adsorbent).where(Adsorbent.name == self._norm(adsorbent_name))
                    ).scalar_one_or_none()
                    if adsorbent is None:
                        adsorbent = Adsorbent(
                            adsorbent_key=self._adsorbent_key(None, str(adsorbent_name)),
                            name=self._norm(adsorbent_name),
                        )
                        session.add(adsorbent)
                        session.flush()

                    adsorbate = session.execute(
                        select(Adsorbate).where(Adsorbate.name == self._norm(adsorbate_name))
                    ).scalar_one_or_none()
                    if adsorbate is None:
                        adsorbate = Adsorbate(
                            adsorbate_key=self._adsorbate_key(None, str(adsorbate_name)),
                            name=self._norm(adsorbate_name),
                        )
                        session.add(adsorbate)
                        session.flush()

                    experiment_name = f"nist:{name}:{adsorbent.name}:{adsorbate.name}:{temperature}"
                    isotherm = AdsorptionIsotherm(
                        dataset_id=dataset_entry.id,
                        source_record_id=str(name),
                        experiment_name=experiment_name,
                        adsorbent_id=adsorbent.id,
                        temperature_k=float(temperature),
                        pressure_units=self._norm(pressure_units),
                        adsorption_units=self._norm(adsorption_units),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    session.add(isotherm)
                    session.flush()

                    component = AdsorptionIsothermComponent(
                        isotherm_id=isotherm.id,
                        component_index=1,
                        adsorbate_id=adsorbate.id,
                    )
                    session.add(component)
                    session.flush()

                    for point_index, (_, point) in enumerate(frame.reset_index(drop=True).iterrows()):
                        point_row = AdsorptionPoint(isotherm_id=isotherm.id, point_index=point_index)
                        session.add(point_row)
                        session.flush()
                        pressure = float(point["pressure"])
                        uptake = float(point["adsorbed_amount"])
                        session.add(
                            AdsorptionPointComponent(
                                point_id=point_row.id,
                                component_id=component.id,
                                partial_pressure_pa=pressure,
                                uptake_mol_g=uptake,
                                original_pressure=pressure,
                                original_uptake=uptake,
                            )
                        )

            if isinstance(binary_mixture, pd.DataFrame) and not binary_mixture.empty:
                grouped = binary_mixture.groupby(
                    ["name", "adsorbent_name", "compound_1", "compound_2", "temperature", "pressure_units", "adsorption_units"],
                    dropna=False,
                )
                for (name, adsorbent_name, compound_1, compound_2, temperature, pressure_units, adsorption_units), frame in grouped:
                    adsorbent = session.execute(
                        select(Adsorbent).where(Adsorbent.name == self._norm(adsorbent_name))
                    ).scalar_one_or_none()
                    if adsorbent is None:
                        adsorbent = Adsorbent(
                            adsorbent_key=self._adsorbent_key(None, str(adsorbent_name)),
                            name=self._norm(adsorbent_name),
                        )
                        session.add(adsorbent)
                        session.flush()

                    guest_1 = session.execute(
                        select(Adsorbate).where(Adsorbate.name == self._norm(compound_1))
                    ).scalar_one_or_none()
                    if guest_1 is None:
                        guest_1 = Adsorbate(
                            adsorbate_key=self._adsorbate_key(None, str(compound_1)),
                            name=self._norm(compound_1),
                        )
                        session.add(guest_1)
                        session.flush()

                    guest_2 = session.execute(
                        select(Adsorbate).where(Adsorbate.name == self._norm(compound_2))
                    ).scalar_one_or_none()
                    if guest_2 is None:
                        guest_2 = Adsorbate(
                            adsorbate_key=self._adsorbate_key(None, str(compound_2)),
                            name=self._norm(compound_2),
                        )
                        session.add(guest_2)
                        session.flush()

                    experiment_name = f"nist-binary:{name}:{adsorbent.name}:{guest_1.name}:{guest_2.name}:{temperature}"
                    isotherm = AdsorptionIsotherm(
                        dataset_id=dataset_entry.id,
                        source_record_id=str(name),
                        experiment_name=experiment_name,
                        adsorbent_id=adsorbent.id,
                        temperature_k=float(temperature),
                        pressure_units=self._norm(pressure_units),
                        adsorption_units=self._norm(adsorption_units),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    session.add(isotherm)
                    session.flush()

                    component_1 = AdsorptionIsothermComponent(
                        isotherm_id=isotherm.id,
                        component_index=1,
                        adsorbate_id=guest_1.id,
                    )
                    component_2 = AdsorptionIsothermComponent(
                        isotherm_id=isotherm.id,
                        component_index=2,
                        adsorbate_id=guest_2.id,
                    )
                    session.add(component_1)
                    session.add(component_2)
                    session.flush()

                    for point_index, (_, point) in enumerate(frame.reset_index(drop=True).iterrows()):
                        point_row = AdsorptionPoint(isotherm_id=isotherm.id, point_index=point_index)
                        session.add(point_row)
                        session.flush()
                        p1 = float(point["compound_1_pressure"])
                        q1 = float(point["compound_1_adsorption"])
                        p2 = float(point["compound_2_pressure"])
                        q2 = float(point["compound_2_adsorption"])
                        session.add(
                            AdsorptionPointComponent(
                                point_id=point_row.id,
                                component_id=component_1.id,
                                partial_pressure_pa=p1,
                                uptake_mol_g=q1,
                                original_pressure=p1,
                                original_uptake=q1,
                            )
                        )
                        session.add(
                            AdsorptionPointComponent(
                                point_id=point_row.id,
                                component_id=component_2.id,
                                partial_pressure_pa=p2,
                                uptake_mol_g=q2,
                                original_pressure=p2,
                                original_uptake=q2,
                            )
                        )

            session.commit()
