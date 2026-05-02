from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import hashlib

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from app.server.domain.models import MODEL_SCHEMAS
from app.server.common.constants import (
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
from app.server.repositories.database.backend import database
from app.server.repositories.queries.data import DataRepositoryQueries
from app.server.repositories.serialization.normalization import (
    build_processed_key_from_values,
    normalize_lower,
    normalize_model_key,
    normalize_text,
    to_float,
    to_float_list,
)
from app.server.repositories.schemas.models import (
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

    table_aliases: dict[str, str] = {}

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
    normalize_text = staticmethod(normalize_text)
    normalize_lower = staticmethod(normalize_lower)
    normalize_model_key = staticmethod(normalize_model_key)
    to_float = staticmethod(to_float)
    to_float_list = staticmethod(to_float_list)
    build_processed_key_from_values = staticmethod(build_processed_key_from_values)

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
        dataset = self.queries.get_dataset_by_name(session, normalized_name)
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
            key = self.material_key("inchi", normalized_name, normalized_inchi)
            adsorbate = self.queries.get_adsorbate_by_inchi(session, normalized_inchi)
        else:
            key = self.material_key("inchi", normalized_name, None)
            adsorbate = self.queries.get_adsorbate_by_key(session, key)
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
            adsorbent = self.queries.get_adsorbent_by_hash(session, normalized_hash)
        else:
            key = self.material_key("host", normalized_name, None)
            adsorbent = self.queries.get_adsorbent_by_key(session, key)
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
        component = self.queries.get_isotherm_component(
            session, isotherm_id, component_index
        )
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
    def _ensure_isotherm(
        self,
        session: Session,
        dataset_id: int,
        source_record_id: str,
        experiment_name: str,
        adsorbent_id: int,
        temperature_k: float,
        pressure_units: str = "pa",
        adsorption_units: str = "mol/g",
    ) -> AdsorptionIsotherm:
        isotherm = self.queries.get_isotherm_by_experiment_name(
            session, experiment_name
        )
        if isotherm is None:
            isotherm = AdsorptionIsotherm(
                dataset_id=dataset_id,
                source_record_id=source_record_id,
                experiment_name=experiment_name,
                adsorbent_id=adsorbent_id,
                temperature_k=temperature_k,
                pressure_units=pressure_units,
                adsorption_units=adsorption_units,
                created_at=self.now_iso(),
            )
            session.add(isotherm)
            session.flush()
            return isotherm

        isotherm.dataset_id = dataset_id
        isotherm.source_record_id = source_record_id
        isotherm.adsorbent_id = adsorbent_id
        isotherm.temperature_k = temperature_k
        isotherm.pressure_units = pressure_units
        isotherm.adsorption_units = adsorption_units
        return isotherm

    # -------------------------------------------------------------------------
    def _ensure_point(
        self,
        session: Session,
        isotherm_id: int,
        point_index: int,
    ) -> AdsorptionPoint:
        point = self.queries.get_point(session, isotherm_id, point_index)
        if point is None:
            point = AdsorptionPoint(isotherm_id=isotherm_id, point_index=point_index)
            session.add(point)
            session.flush()
        return point

    # -------------------------------------------------------------------------
    def _upsert_point_component(
        self,
        session: Session,
        point_id: int,
        component_id: int,
        partial_pressure_pa: float,
        uptake_mol_g: float,
        original_pressure: float | None,
        original_uptake: float | None,
    ) -> None:
        component = self.queries.get_point_component(session, point_id, component_id)
        if component is None:
            session.add(
                AdsorptionPointComponent(
                    point_id=point_id,
                    component_id=component_id,
                    partial_pressure_pa=partial_pressure_pa,
                    uptake_mol_g=uptake_mol_g,
                    original_pressure=original_pressure,
                    original_uptake=original_uptake,
                )
            )
            return

        component.partial_pressure_pa = partial_pressure_pa
        component.uptake_mol_g = uptake_mol_g
        component.original_pressure = original_pressure
        component.original_uptake = original_uptake

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
            point = self._ensure_point(
                session, isotherm_id=isotherm_id, point_index=idx
            )
            original_pressure = None
            original_uptake = None
            if original_pressure_values and idx < len(original_pressure_values):
                original_pressure = original_pressure_values[idx]
            if original_uptake_values and idx < len(original_uptake_values):
                original_uptake = original_uptake_values[idx]
            self._upsert_point_component(
                session,
                point_id=point.id,
                component_id=component_id,
                partial_pressure_pa=pressure_pa,
                uptake_mol_g=uptake_mol_g,
                original_pressure=original_pressure,
                original_uptake=original_uptake,
            )

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
                    isotherm = self._ensure_isotherm(
                        session,
                        dataset_id=dataset_entry.id,
                        source_record_id=str(experiment),
                        experiment_name=experiment_name,
                        adsorbent_id=adsorbent.id,
                        temperature_k=self.to_float(temperature) or 0.0,
                        pressure_units="pa",
                        adsorption_units="mol/g",
                    )

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
            target = self.queries.get_uploaded_dataset_by_name(session, normalized_name)
            if target is None:
                return False

            archived_name = f"archived::{normalized_name}::{self.now_iso()}"
            target.dataset_name = archived_name
            session.commit()
            return True

    # -------------------------------------------------------------------------
    def _load_uploaded_raw(self) -> pd.DataFrame:
        with self.session_factory() as session:
            rows = self.queries.load_uploaded_raw_rows(session)

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
    def _load_processed_isotherms(self) -> pd.DataFrame:
        with self.session_factory() as session:
            rows = self.queries.load_processed_rows(session)

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
            frame = self._load_processed_isotherms()
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

                isotherm = self.queries.get_isotherm_by_experiment_name(
                    session, experiment_name
                )
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
                processed = self.queries.get_processed_by_isotherm_and_version(
                    session,
                    isotherm.id,
                    self.processing_version,
                )
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
                    fit = self.queries.get_fit_by_processed_model_method(
                        session,
                        processed.id,
                        model_key,
                        optimization_method,
                    )
                    if fit is None:
                        fit = AdsorptionFit(
                            processed_id=processed.id,
                            model_name=model_key,
                            optimization_method=optimization_method,
                            score=score_value,
                            aic=self.to_float(row.get(aic_column))
                            if aic_column
                            else None,
                            aicc=self.to_float(row.get(aicc_column))
                            if aicc_column
                            else None,
                            created_at=self.now_iso(),
                        )
                        session.add(fit)
                        session.flush()
                    else:
                        fit.score = score_value
                        fit.aic = (
                            self.to_float(row.get(aic_column)) if aic_column else None
                        )
                        fit.aicc = (
                            self.to_float(row.get(aicc_column)) if aicc_column else None
                        )
                        fit.created_at = self.now_iso()

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
                        fit_param = self.queries.get_fit_param(
                            session,
                            fit.id,
                            db_name,
                        )
                        if fit_param is None:
                            session.add(
                                AdsorptionFitParam(
                                    fit_id=fit.id,
                                    param_name=db_name,
                                    param_value=param_value,
                                    param_error=param_error,
                                )
                            )
                        else:
                            fit_param.param_value = param_value
                            fit_param.param_error = param_error

            session.commit()

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        return self._load_processed_isotherms().rename(
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

                isotherm = self.queries.get_isotherm_by_experiment_name(
                    session, experiment_name
                )
                if isotherm is None:
                    continue

                processed = self.queries.get_processed_by_isotherm_and_version(
                    session,
                    isotherm.id,
                    self.processing_version,
                )
                if processed is None:
                    continue

                best_model = self.normalize_model_key(row.get(COLUMN_BEST_MODEL))
                worst_model = self.normalize_model_key(row.get(COLUMN_WORST_MODEL))

                best_fit = self.queries.get_fit_by_processed_and_model(
                    session,
                    processed.id,
                    best_model,
                )
                worst_fit = self.queries.get_fit_by_processed_and_model(
                    session,
                    processed.id,
                    worst_model,
                )

                existing = self.queries.get_best_fit_by_processed(
                    session,
                    processed.id,
                )
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
        processed = self._load_processed_isotherms()
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
