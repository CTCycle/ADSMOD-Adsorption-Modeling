from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import hashlib
import json

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session, sessionmaker

from ADSMOD.server.entities.models import MODEL_SCHEMAS
from ADSMOD.server.common.constants import (
    COLUMN_ADSORBATE,
    COLUMN_ADSORBENT,
    COLUMN_BEST_MODEL,
    COLUMN_DATASET_NAME,
    COLUMN_EXPERIMENT,
    COLUMN_EXPERIMENT_NAME,
    COLUMN_ID,
    COLUMN_MAX_PRESSURE,
    COLUMN_MAX_UPTAKE,
    COLUMN_MEASUREMENT_COUNT,
    COLUMN_MIN_PRESSURE,
    COLUMN_MIN_UPTAKE,
    COLUMN_PRESSURE_PA,
    COLUMN_TEMPERATURE_K,
    COLUMN_UPTAKE_MOL_G,
    COLUMN_WORST_MODEL,
)
from ADSMOD.server.repositories.database.backend import database
from ADSMOD.server.repositories.queries.data import DataRepositoryQueries
from ADSMOD.server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    AdsorptionBestFit,
    AdsorptionFit,
    AdsorptionFitParam,
    AdsorptionIsotherm,
    AdsorptionIsothermComponent,
    AdsorptionPoint,
    AdsorptionPointComponent,
    AdsorptionProcessedIsotherm,
    Dataset,
)


###############################################################################
class DataSerializer:
    raw_table = "adsorption_data"
    processed_table = "adsorption_processed_data"
    best_fit_table = "adsorption_best_fit"
    raw_name_column = "name"
    fitting_name_column = "name"
    processed_key_column = "processed_key"
    processing_version = "v1"

    table_aliases = {
        "ADSORPTION_DATA": raw_table,
        "ADSORPTION_PROCESSED_DATA": processed_table,
        "ADSORPTION_BEST_FIT": best_fit_table,
        "ADSORPTION_FITS": "adsorption_fits",
        "ADSORPTION_FIT_PARAMS": "adsorption_fit_params",
        "ADSORPTION_ISOTHERMS": "adsorption_isotherms",
        "ADSORPTION_POINTS": "adsorption_points",
        "ADSORPTION_POINT_COMPONENTS": "adsorption_point_components",
        "ADSORPTION_ISOTHERM_COMPONENTS": "adsorption_isotherm_components",
        "ADSORPTION_PROCESSED_ISOTHERMS": "adsorption_processed_isotherms",
        "DATASETS": "datasets",
        "ADSORBATES": "adsorbates",
        "ADSORBENTS": "adsorbents",
        "TRAINING_DATASET": "training_dataset",
        "TRAINING_METADATA": "training_metadata",
    }

    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.engine = database.backend.engine
        self.session_factory = sessionmaker(bind=self.engine, future=True)
        self.model_prefixes = {
            self.normalize_model_key(schema["prefix"]): schema["prefix"]
            for schema in MODEL_SCHEMAS.values()
        }

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_table_name(cls, table_name: str) -> str:
        return cls.table_aliases.get(table_name, table_name)

    # -------------------------------------------------------------------------
    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_lower(value: Any) -> str:
        return DataSerializer.normalize_text(value).lower()

    # -------------------------------------------------------------------------
    @staticmethod
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

    # -------------------------------------------------------------------------
    @staticmethod
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

    # -------------------------------------------------------------------------
    @staticmethod
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
                parsed = [
                    entry.strip() for entry in stripped.split(",") if entry.strip()
                ]
            return [float(entry) for entry in parsed if not pd.isna(entry)]
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            values: list[float] = []
            for entry in value:
                parsed = DataSerializer.to_float(entry)
                if parsed is None:
                    continue
                values.append(parsed)
            return values
        parsed = DataSerializer.to_float(value)
        return [parsed] if parsed is not None else []

    # -------------------------------------------------------------------------
    def normalize_material_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        normalized = dataset.copy()
        rename_map: dict[str, str] = {}
        for column in normalized.columns:
            lowered = str(column).strip().lower()
            if lowered in {
                "adsorbent",
                "adsorbents",
                "adsorbent_name",
                "adsorbent name",
            }:
                rename_map[column] = COLUMN_ADSORBENT
            elif lowered in {
                "adsorbate",
                "adsorbates",
                "adsorbate_name",
                "adsorbate name",
            }:
                rename_map[column] = COLUMN_ADSORBATE
        if rename_map:
            normalized = normalized.rename(columns=rename_map)
        return normalized

    # -------------------------------------------------------------------------
    @staticmethod
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

    # -------------------------------------------------------------------------
    def material_key(
        self, prefix: str, name: str, external_key: str | None = None
    ) -> str:
        external = self.normalize_text(external_key)
        if external:
            return f"{prefix}:{external.lower()}"
        normalized_name = self.normalize_lower(name)
        if normalized_name:
            digest = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:24]
            return f"name:{digest}"
        digest = hashlib.sha1(self.now_iso().encode("utf-8")).hexdigest()[:24]
        return f"auto:{digest}"

    # -------------------------------------------------------------------------
    def _ensure_dataset(
        self, session: Session, dataset_name: str, source: str
    ) -> Dataset:
        normalized_name = self.normalize_text(dataset_name) or "default"
        dataset = session.execute(
            select(Dataset).where(Dataset.dataset_name == normalized_name)
        ).scalar_one_or_none()
        if dataset is None:
            dataset = Dataset(
                dataset_name=normalized_name,
                source=source,
                created_at=self.now_iso(),
            )
            session.add(dataset)
            session.flush()
        elif dataset.source != source:
            dataset.source = source
        return dataset

    # -------------------------------------------------------------------------
    def _ensure_adsorbate(
        self,
        session: Session,
        name: str,
        inchi_key: str | None = None,
        inchi_code: str | None = None,
        formula: str | None = None,
        molecular_weight: float | None = None,
        molecular_formula: str | None = None,
        smile_code: str | None = None,
    ) -> Adsorbate:
        normalized_name = self.normalize_lower(name)
        normalized_inchi = self.normalize_text(inchi_key)
        if normalized_inchi:
            query = select(Adsorbate).where(Adsorbate.InChIKey == normalized_inchi)
            key = self.material_key("inchi", normalized_name, normalized_inchi)
        else:
            key = self.material_key("inchi", normalized_name, None)
            query = select(Adsorbate).where(Adsorbate.adsorbate_key == key)

        adsorbate = session.execute(query).scalar_one_or_none()
        if adsorbate is None:
            adsorbate = Adsorbate(
                adsorbate_key=key,
                InChIKey=normalized_inchi or None,
                name=normalized_name or None,
                InChICode=self.normalize_text(inchi_code) or None,
                formula=self.normalize_text(formula) or None,
                molecular_weight=molecular_weight,
                molecular_formula=self.normalize_text(molecular_formula) or None,
                smile_code=self.normalize_text(smile_code) or None,
            )
            session.add(adsorbate)
            session.flush()
            return adsorbate

        if normalized_name and not adsorbate.name:
            adsorbate.name = normalized_name
        if inchi_code and not adsorbate.InChICode:
            adsorbate.InChICode = self.normalize_text(inchi_code)
        if formula and not adsorbate.formula:
            adsorbate.formula = self.normalize_text(formula)
        if molecular_weight is not None and adsorbate.molecular_weight is None:
            adsorbate.molecular_weight = molecular_weight
        if molecular_formula and not adsorbate.molecular_formula:
            adsorbate.molecular_formula = self.normalize_text(molecular_formula)
        if smile_code and not adsorbate.smile_code:
            adsorbate.smile_code = self.normalize_text(smile_code)
        return adsorbate

    # -------------------------------------------------------------------------
    def _ensure_adsorbent(
        self,
        session: Session,
        name: str,
        hashkey: str | None = None,
        formula: str | None = None,
        molecular_weight: float | None = None,
        molecular_formula: str | None = None,
        smile_code: str | None = None,
    ) -> Adsorbent:
        normalized_name = self.normalize_lower(name)
        normalized_hash = self.normalize_text(hashkey)
        if normalized_hash:
            key = self.material_key("host", normalized_name, normalized_hash)
            query = select(Adsorbent).where(Adsorbent.hashkey == normalized_hash)
        else:
            key = self.material_key("host", normalized_name, None)
            query = select(Adsorbent).where(Adsorbent.adsorbent_key == key)

        adsorbent = session.execute(query).scalar_one_or_none()
        if adsorbent is None:
            adsorbent = Adsorbent(
                adsorbent_key=key,
                hashkey=normalized_hash or None,
                name=normalized_name or None,
                formula=self.normalize_text(formula) or None,
                molecular_weight=molecular_weight,
                molecular_formula=self.normalize_text(molecular_formula) or None,
                smile_code=self.normalize_text(smile_code) or None,
            )
            session.add(adsorbent)
            session.flush()
            return adsorbent

        if normalized_name and not adsorbent.name:
            adsorbent.name = normalized_name
        if formula and not adsorbent.formula:
            adsorbent.formula = self.normalize_text(formula)
        if molecular_weight is not None and adsorbent.molecular_weight is None:
            adsorbent.molecular_weight = molecular_weight
        if molecular_formula and not adsorbent.molecular_formula:
            adsorbent.molecular_formula = self.normalize_text(molecular_formula)
        if smile_code and not adsorbent.smile_code:
            adsorbent.smile_code = self.normalize_text(smile_code)
        return adsorbent

    # -------------------------------------------------------------------------
    def _ensure_isotherm_component(
        self,
        session: Session,
        isotherm_id: int,
        component_index: int,
        adsorbate_id: int,
        mole_fraction: float | None = None,
    ) -> AdsorptionIsothermComponent:
        component = session.execute(
            select(AdsorptionIsothermComponent).where(
                and_(
                    AdsorptionIsothermComponent.isotherm_id == isotherm_id,
                    AdsorptionIsothermComponent.component_index == component_index,
                )
            )
        ).scalar_one_or_none()
        if component is None:
            component = AdsorptionIsothermComponent(
                isotherm_id=isotherm_id,
                component_index=component_index,
                adsorbate_id=adsorbate_id,
                mole_fraction=mole_fraction,
            )
            session.add(component)
            session.flush()
            return component

        component.adsorbate_id = adsorbate_id
        component.mole_fraction = mole_fraction
        return component

    # -------------------------------------------------------------------------
    def _insert_points_for_single_component(
        self,
        session: Session,
        isotherm_id: int,
        component_id: int,
        pressure_values: list[float],
        uptake_values: list[float],
        original_pressure_values: list[float] | None = None,
        original_uptake_values: list[float] | None = None,
    ) -> None:
        for idx, (pressure_pa, uptake_mol_g) in enumerate(
            zip(pressure_values, uptake_values, strict=False)
        ):
            point = AdsorptionPoint(isotherm_id=isotherm_id, point_index=idx)
            session.add(point)
            session.flush()
            original_pressure = None
            original_uptake = None
            if original_pressure_values and idx < len(original_pressure_values):
                original_pressure = original_pressure_values[idx]
            if original_uptake_values and idx < len(original_uptake_values):
                original_uptake = original_uptake_values[idx]
            session.add(
                AdsorptionPointComponent(
                    point_id=point.id,
                    component_id=component_id,
                    partial_pressure_pa=pressure_pa,
                    uptake_mol_g=uptake_mol_g,
                    original_pressure=original_pressure,
                    original_uptake=original_uptake,
                )
            )

    # -------------------------------------------------------------------------
    def _replace_dataset_isotherms(self, session: Session, dataset_id: int) -> None:
        session.query(AdsorptionIsotherm).filter(
            AdsorptionIsotherm.dataset_id == dataset_id
        ).delete(synchronize_session=False)

    # -------------------------------------------------------------------------
    def save_raw_dataset(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return

        normalized = self.normalize_material_columns(dataset)
        if (
            self.raw_name_column not in normalized.columns
            and COLUMN_DATASET_NAME in normalized.columns
        ):
            normalized = normalized.rename(
                columns={COLUMN_DATASET_NAME: self.raw_name_column}
            )

        required = [
            self.raw_name_column,
            COLUMN_EXPERIMENT,
            COLUMN_ADSORBENT,
            COLUMN_ADSORBATE,
            COLUMN_TEMPERATURE_K,
            COLUMN_PRESSURE_PA,
            COLUMN_UPTAKE_MOL_G,
        ]
        for column in required:
            if column not in normalized.columns:
                raise ValueError(f"Missing required upload column: {column}")

        normalized = normalized.dropna(subset=required).copy()
        if normalized.empty:
            return

        normalized[COLUMN_ADSORBENT] = (
            normalized[COLUMN_ADSORBENT].astype("string").str.strip().str.lower()
        )
        normalized[COLUMN_ADSORBATE] = (
            normalized[COLUMN_ADSORBATE].astype("string").str.strip().str.lower()
        )
        normalized[COLUMN_EXPERIMENT] = (
            normalized[COLUMN_EXPERIMENT].astype("string").str.strip()
        )
        normalized[self.raw_name_column] = (
            normalized[self.raw_name_column].astype("string").str.strip()
        )

        with self.session_factory() as session:
            dataset_names = [
                name
                for name in normalized[self.raw_name_column].dropna().unique().tolist()
                if str(name).strip()
            ]
            for dataset_name in dataset_names:
                subset = normalized[
                    normalized[self.raw_name_column] == dataset_name
                ].copy()
                dataset_entry = self._ensure_dataset(
                    session, str(dataset_name), "uploaded"
                )
                self._replace_dataset_isotherms(session, dataset_entry.id)

                grouped = subset.groupby(
                    [
                        COLUMN_EXPERIMENT,
                        COLUMN_ADSORBENT,
                        COLUMN_ADSORBATE,
                        COLUMN_TEMPERATURE_K,
                    ],
                    dropna=False,
                )
                for (
                    experiment,
                    adsorbent_name,
                    adsorbate_name,
                    temperature,
                ), frame in grouped:
                    adsorbent = self._ensure_adsorbent(session, str(adsorbent_name))
                    adsorbate = self._ensure_adsorbate(session, str(adsorbate_name))
                    experiment_name = f"uploaded:{dataset_name}:{experiment}"
                    isotherm = AdsorptionIsotherm(
                        dataset_id=dataset_entry.id,
                        source_record_id=str(experiment),
                        experiment_name=experiment_name,
                        adsorbent_id=adsorbent.id,
                        temperature_k=self.to_float(temperature) or 0.0,
                        pressure_units="pa",
                        adsorption_units="mol/g",
                        created_at=self.now_iso(),
                    )
                    session.add(isotherm)
                    session.flush()

                    component = self._ensure_isotherm_component(
                        session,
                        isotherm.id,
                        component_index=1,
                        adsorbate_id=adsorbate.id,
                    )
                    pressure_values = [
                        value
                        for value in frame[COLUMN_PRESSURE_PA]
                        .apply(self.to_float)
                        .tolist()
                        if value is not None
                    ]
                    uptake_values = [
                        value
                        for value in frame[COLUMN_UPTAKE_MOL_G]
                        .apply(self.to_float)
                        .tolist()
                        if value is not None
                    ]
                    self._insert_points_for_single_component(
                        session,
                        isotherm.id,
                        component.id,
                        pressure_values,
                        uptake_values,
                        original_pressure_values=pressure_values,
                        original_uptake_values=uptake_values,
                    )

            session.commit()

    # -------------------------------------------------------------------------
    def delete_raw_dataset(self, dataset_name: str) -> bool:
        normalized_name = self.normalize_text(dataset_name)
        if not normalized_name:
            return False

        with self.session_factory() as session:
            target = session.execute(
                select(Dataset).where(
                    and_(
                        Dataset.dataset_name == normalized_name,
                        Dataset.source == "uploaded",
                    )
                )
            ).scalar_one_or_none()
            if target is None:
                return False
            session.delete(target)
            session.commit()
            return True

    # -------------------------------------------------------------------------
    def _load_uploaded_raw(self) -> pd.DataFrame:
        with self.session_factory() as session:
            rows = session.execute(
                select(
                    Dataset.dataset_name,
                    AdsorptionIsotherm.source_record_id,
                    Adsorbent.name,
                    Adsorbate.name,
                    AdsorptionIsotherm.temperature_k,
                    AdsorptionPoint.point_index,
                    AdsorptionPointComponent.original_pressure,
                    AdsorptionPointComponent.original_uptake,
                    AdsorptionPointComponent.partial_pressure_pa,
                    AdsorptionPointComponent.uptake_mol_g,
                )
                .join(AdsorptionIsotherm, AdsorptionIsotherm.dataset_id == Dataset.id)
                .join(
                    AdsorptionIsothermComponent,
                    and_(
                        AdsorptionIsothermComponent.isotherm_id
                        == AdsorptionIsotherm.id,
                        AdsorptionIsothermComponent.component_index == 1,
                    ),
                )
                .join(
                    Adsorbate, Adsorbate.id == AdsorptionIsothermComponent.adsorbate_id
                )
                .join(Adsorbent, Adsorbent.id == AdsorptionIsotherm.adsorbent_id)
                .join(
                    AdsorptionPoint,
                    AdsorptionPoint.isotherm_id == AdsorptionIsotherm.id,
                )
                .join(
                    AdsorptionPointComponent,
                    and_(
                        AdsorptionPointComponent.point_id == AdsorptionPoint.id,
                        AdsorptionPointComponent.component_id
                        == AdsorptionIsothermComponent.id,
                    ),
                )
                .where(Dataset.source == "uploaded")
                .order_by(
                    Dataset.dataset_name,
                    AdsorptionIsotherm.source_record_id,
                    AdsorptionPoint.point_index,
                )
            ).all()

        if not rows:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        for row in rows:
            pressure = row.original_pressure
            uptake = row.original_uptake
            if pressure is None:
                pressure = row.partial_pressure_pa
            if uptake is None:
                uptake = row.uptake_mol_g
            records.append(
                {
                    self.raw_name_column: row.dataset_name,
                    COLUMN_EXPERIMENT: row.source_record_id,
                    COLUMN_ADSORBENT: row.name,
                    COLUMN_ADSORBATE: row.name_1,
                    COLUMN_TEMPERATURE_K: row.temperature_k,
                    COLUMN_PRESSURE_PA: pressure,
                    COLUMN_UPTAKE_MOL_G: uptake,
                }
            )
        return pd.DataFrame.from_records(records)

    # -------------------------------------------------------------------------
    def _load_processed_compat(self) -> pd.DataFrame:
        with self.session_factory() as session:
            rows = session.execute(
                select(
                    AdsorptionProcessedIsotherm.id,
                    AdsorptionIsotherm.experiment_name,
                    Adsorbent.name,
                    Adsorbate.name,
                    AdsorptionIsotherm.temperature_k,
                    AdsorptionProcessedIsotherm.pressure_pa_series,
                    AdsorptionProcessedIsotherm.uptake_mol_g_series,
                    AdsorptionProcessedIsotherm.measurement_count,
                    AdsorptionProcessedIsotherm.min_pressure,
                    AdsorptionProcessedIsotherm.max_pressure,
                    AdsorptionProcessedIsotherm.min_uptake,
                    AdsorptionProcessedIsotherm.max_uptake,
                    AdsorptionProcessedIsotherm.processed_key,
                )
                .join(
                    AdsorptionIsotherm,
                    AdsorptionIsotherm.id == AdsorptionProcessedIsotherm.isotherm_id,
                )
                .join(
                    AdsorptionIsothermComponent,
                    and_(
                        AdsorptionIsothermComponent.isotherm_id
                        == AdsorptionIsotherm.id,
                        AdsorptionIsothermComponent.component_index == 1,
                    ),
                )
                .join(
                    Adsorbate, Adsorbate.id == AdsorptionIsothermComponent.adsorbate_id
                )
                .join(Adsorbent, Adsorbent.id == AdsorptionIsotherm.adsorbent_id)
                .order_by(AdsorptionProcessedIsotherm.id)
            ).all()

        if not rows:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    COLUMN_ID: row.id,
                    self.fitting_name_column: row.experiment_name,
                    COLUMN_ADSORBENT: row.name,
                    COLUMN_ADSORBATE: row.name_1,
                    COLUMN_TEMPERATURE_K: row.temperature_k,
                    COLUMN_PRESSURE_PA: row.pressure_pa_series,
                    COLUMN_UPTAKE_MOL_G: row.uptake_mol_g_series,
                    COLUMN_MEASUREMENT_COUNT: row.measurement_count,
                    COLUMN_MIN_PRESSURE: row.min_pressure,
                    COLUMN_MAX_PRESSURE: row.max_pressure,
                    COLUMN_MIN_UPTAKE: row.min_uptake,
                    COLUMN_MAX_UPTAKE: row.max_uptake,
                    self.processed_key_column: row.processed_key,
                }
            )
        return pd.DataFrame.from_records(records)

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        normalized = self.normalize_table_name(table_name)

        if normalized == self.raw_table:
            frame = self._load_uploaded_raw()
            if frame.empty:
                return frame
            if offset:
                frame = frame.iloc[offset:]
            if limit is not None:
                frame = frame.head(limit)
            return frame.reset_index(drop=True)

        if normalized == self.processed_table:
            frame = self._load_processed_compat()
            if frame.empty:
                return frame
            if offset:
                frame = frame.iloc[offset:]
            if limit is not None:
                frame = frame.head(limit)
            return frame.reset_index(drop=True)

        if normalized == self.best_fit_table:
            return self.load_best_fit()

        return self.queries.load_table(normalized, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def save_processed_dataset(self, dataset: pd.DataFrame) -> None:
        self.save_fitting_results(dataset)

    # -------------------------------------------------------------------------
    def _resolve_dataset_column(
        self, prefix: str, suffix: str, columns: list[str] | pd.Index
    ) -> str | None:
        target = f"{prefix} {suffix}".lower()
        for column in columns:
            if str(column).lower() == target:
                return str(column)
        return None

    # -------------------------------------------------------------------------
    def save_fitting_results(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for fitting results.")

        prepared = self.normalize_material_columns(dataset.copy())
        with self.session_factory() as session:
            default_dataset_name = "fitting_runtime"
            dataset_name = self.normalize_text(
                prepared.get(
                    COLUMN_DATASET_NAME, pd.Series([default_dataset_name])
                ).iloc[0]
            )
            dataset_entry = self._ensure_dataset(
                session,
                dataset_name or default_dataset_name,
                "uploaded",
            )

            for _, row in prepared.iterrows():
                experiment_name = self.normalize_text(row.get(COLUMN_EXPERIMENT_NAME))
                if not experiment_name:
                    continue

                adsorbent_name = self.normalize_lower(row.get(COLUMN_ADSORBENT))
                adsorbate_name = self.normalize_lower(row.get(COLUMN_ADSORBATE))
                temperature_k = self.to_float(row.get(COLUMN_TEMPERATURE_K)) or 0.0
                pressure_series = self.to_float_list(row.get(COLUMN_PRESSURE_PA))
                uptake_series = self.to_float_list(row.get(COLUMN_UPTAKE_MOL_G))

                adsorbent = self._ensure_adsorbent(session, adsorbent_name)
                adsorbate = self._ensure_adsorbate(session, adsorbate_name)

                isotherm = session.execute(
                    select(AdsorptionIsotherm).where(
                        AdsorptionIsotherm.experiment_name == experiment_name
                    )
                ).scalar_one_or_none()
                if isotherm is None:
                    isotherm = AdsorptionIsotherm(
                        dataset_id=dataset_entry.id,
                        source_record_id=experiment_name,
                        experiment_name=experiment_name,
                        adsorbent_id=adsorbent.id,
                        temperature_k=temperature_k,
                        pressure_units="pa",
                        adsorption_units="mol/g",
                        created_at=self.now_iso(),
                    )
                    session.add(isotherm)
                    session.flush()
                else:
                    isotherm.adsorbent_id = adsorbent.id
                    isotherm.temperature_k = temperature_k

                self._ensure_isotherm_component(
                    session,
                    isotherm.id,
                    component_index=1,
                    adsorbate_id=adsorbate.id,
                )

                processed_key = self.build_processed_key_from_values(
                    adsorbent_name,
                    adsorbate_name,
                    temperature_k,
                    pressure_series,
                    uptake_series,
                )
                processed = session.execute(
                    select(AdsorptionProcessedIsotherm).where(
                        and_(
                            AdsorptionProcessedIsotherm.isotherm_id == isotherm.id,
                            AdsorptionProcessedIsotherm.processing_version
                            == self.processing_version,
                        )
                    )
                ).scalar_one_or_none()
                if processed is None:
                    processed = AdsorptionProcessedIsotherm(
                        isotherm_id=isotherm.id,
                        processing_version=self.processing_version,
                        processed_key=processed_key,
                        pressure_pa_series=pressure_series,
                        uptake_mol_g_series=uptake_series,
                        original_pressure_series=pressure_series,
                        original_uptake_series=uptake_series,
                        measurement_count=len(pressure_series),
                        min_pressure=min(pressure_series) if pressure_series else None,
                        max_pressure=max(pressure_series) if pressure_series else None,
                        min_uptake=min(uptake_series) if uptake_series else None,
                        max_uptake=max(uptake_series) if uptake_series else None,
                    )
                    session.add(processed)
                    session.flush()
                else:
                    processed.processed_key = processed_key
                    processed.pressure_pa_series = pressure_series
                    processed.uptake_mol_g_series = uptake_series
                    processed.original_pressure_series = pressure_series
                    processed.original_uptake_series = uptake_series
                    processed.measurement_count = len(pressure_series)
                    processed.min_pressure = (
                        min(pressure_series) if pressure_series else None
                    )
                    processed.max_pressure = (
                        max(pressure_series) if pressure_series else None
                    )
                    processed.min_uptake = min(uptake_series) if uptake_series else None
                    processed.max_uptake = max(uptake_series) if uptake_series else None

                fit_ids = (
                    session.execute(
                        select(AdsorptionFit.id).where(
                            AdsorptionFit.processed_id == processed.id
                        )
                    )
                    .scalars()
                    .all()
                )
                if fit_ids:
                    session.query(AdsorptionFitParam).filter(
                        AdsorptionFitParam.fit_id.in_(fit_ids)
                    ).delete(synchronize_session=False)
                session.query(AdsorptionFit).filter(
                    AdsorptionFit.processed_id == processed.id
                ).delete(synchronize_session=False)

                for schema in MODEL_SCHEMAS.values():
                    prefix = schema["prefix"]
                    fields = schema["fields"]
                    score_column = self._resolve_dataset_column(
                        prefix, fields["score"], prepared.columns
                    )
                    score_value = (
                        self.to_float(row.get(score_column)) if score_column else None
                    )
                    if score_value is None:
                        continue

                    optimization_column = self._resolve_dataset_column(
                        prefix,
                        fields["optimization_method"],
                        prepared.columns,
                    )
                    optimization_method = (
                        self.normalize_text(
                            row.get(optimization_column)
                            if optimization_column
                            else "LSS"
                        )
                        or "LSS"
                    )

                    aic_column = self._resolve_dataset_column(
                        prefix, fields["aic"], prepared.columns
                    )
                    aicc_column = self._resolve_dataset_column(
                        prefix, fields["aicc"], prepared.columns
                    )

                    model_key = self.normalize_model_key(prefix)
                    fit = AdsorptionFit(
                        processed_id=processed.id,
                        model_name=model_key,
                        optimization_method=optimization_method,
                        score=score_value,
                        aic=self.to_float(row.get(aic_column)) if aic_column else None,
                        aicc=self.to_float(row.get(aicc_column))
                        if aicc_column
                        else None,
                        created_at=self.now_iso(),
                    )
                    session.add(fit)
                    session.flush()

                    for field_name, db_name in fields.items():
                        if field_name in {
                            "optimization_method",
                            "score",
                            "aic",
                            "aicc",
                        }:
                            continue
                        value_column = self._resolve_dataset_column(
                            prefix, db_name, prepared.columns
                        )
                        param_value = (
                            self.to_float(row.get(value_column))
                            if value_column
                            else None
                        )
                        if param_value is None:
                            continue
                        error_column = self._resolve_dataset_column(
                            prefix,
                            f"{db_name} error",
                            prepared.columns,
                        )
                        param_error = (
                            self.to_float(row.get(error_column))
                            if error_column
                            else None
                        )
                        session.add(
                            AdsorptionFitParam(
                                fit_id=fit.id,
                                param_name=db_name,
                                param_value=param_value,
                                param_error=param_error,
                            )
                        )

            session.commit()

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        return self._load_processed_compat().rename(
            columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
        )

    # -------------------------------------------------------------------------
    def save_best_fit(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for best fit results.")

        with self.session_factory() as session:
            for _, row in dataset.iterrows():
                experiment_name = self.normalize_text(row.get(COLUMN_EXPERIMENT_NAME))
                if not experiment_name:
                    continue

                isotherm = session.execute(
                    select(AdsorptionIsotherm).where(
                        AdsorptionIsotherm.experiment_name == experiment_name
                    )
                ).scalar_one_or_none()
                if isotherm is None:
                    continue

                processed = session.execute(
                    select(AdsorptionProcessedIsotherm).where(
                        and_(
                            AdsorptionProcessedIsotherm.isotherm_id == isotherm.id,
                            AdsorptionProcessedIsotherm.processing_version
                            == self.processing_version,
                        )
                    )
                ).scalar_one_or_none()
                if processed is None:
                    continue

                best_model = self.normalize_model_key(row.get(COLUMN_BEST_MODEL))
                worst_model = self.normalize_model_key(row.get(COLUMN_WORST_MODEL))

                best_fit = session.execute(
                    select(AdsorptionFit).where(
                        and_(
                            AdsorptionFit.processed_id == processed.id,
                            AdsorptionFit.model_name == best_model,
                        )
                    )
                ).scalar_one_or_none()
                worst_fit = session.execute(
                    select(AdsorptionFit).where(
                        and_(
                            AdsorptionFit.processed_id == processed.id,
                            AdsorptionFit.model_name == worst_model,
                        )
                    )
                ).scalar_one_or_none()

                existing = session.execute(
                    select(AdsorptionBestFit).where(
                        AdsorptionBestFit.processed_id == processed.id
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        AdsorptionBestFit(
                            processed_id=processed.id,
                            best_fit_id=best_fit.id if best_fit else None,
                            worst_fit_id=worst_fit.id if worst_fit else None,
                            best_model=row.get(COLUMN_BEST_MODEL),
                            worst_model=row.get(COLUMN_WORST_MODEL),
                        )
                    )
                    continue

                existing.best_fit_id = best_fit.id if best_fit else None
                existing.worst_fit_id = worst_fit.id if worst_fit else None
                existing.best_model = row.get(COLUMN_BEST_MODEL)
                existing.worst_model = row.get(COLUMN_WORST_MODEL)

            session.commit()

    # -------------------------------------------------------------------------
    def load_best_fit(self) -> pd.DataFrame:
        processed = self._load_processed_compat()
        if processed.empty:
            return processed

        base = processed.rename(
            columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
        )
        best_rows = self.queries.load_table("adsorption_best_fit")
        if best_rows.empty:
            return base

        processed_ids = base[[COLUMN_ID, COLUMN_EXPERIMENT_NAME]].drop_duplicates()
        merged = processed_ids.merge(
            best_rows[["processed_id", "best_model", "worst_model"]],
            how="left",
            left_on=COLUMN_ID,
            right_on="processed_id",
        )
        merged = merged.rename(
            columns={"best_model": COLUMN_BEST_MODEL, "worst_model": COLUMN_WORST_MODEL}
        )
        merged = merged.drop(columns=["processed_id"], errors="ignore")

        return base.merge(merged, how="left", on=[COLUMN_ID, COLUMN_EXPERIMENT_NAME])


###############################################################################
