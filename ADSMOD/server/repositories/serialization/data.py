from __future__ import annotations

from typing import Any

import hashlib
import json

import pandas as pd

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
from ADSMOD.server.repositories.queries.data import DataRepositoryQueries


###############################################################################
class DataSerializer:
    raw_table = "adsorption_data"
    processed_table = "adsorption_processed_data"
    best_fit_table = "adsorption_best_fit"
    raw_name_column = "name"
    fitting_name_column = "name"
    processed_key_column = "processed_key"
    material_aliases = {
        COLUMN_ADSORBATE: [
            COLUMN_ADSORBATE,
            "adsorbates",
            "adsorbate_name",
            "adsorbate name",
        ],
        COLUMN_ADSORBENT: [
            COLUMN_ADSORBENT,
            "adsorbents",
            "adsorbent_name",
            "adsorbent name",
        ],
    }
    table_aliases = {
        "ADSORPTION_DATA": raw_table,
        "ADSORPTION_PROCESSED_DATA": processed_table,
        "ADSORPTION_BEST_FIT": best_fit_table,
        "ADSORPTION_LANGMUIR": "adsorption_langmuir",
        "ADSORPTION_SIPS": "adsorption_sips",
        "ADSORPTION_FREUNDLICH": "adsorption_freundlich",
        "ADSORPTION_TEMKIN": "adsorption_temkin",
        "ADSORPTION_TOTH": "adsorption_toth",
        "ADSORPTION_DUBININ_RADUSHKEVICH": "adsorption_dubinin_radushkevich",
        "ADSORPTION_DUAL_SITE_LANGMUIR": "adsorption_dual_site_langmuir",
        "ADSORPTION_REDLICH_PETERSON": "adsorption_redlich_peterson",
        "ADSORPTION_JOVANOVIC": "adsorption_jovanovic",
        "NIST_SINGLE_COMPONENT_ADSORPTION": "nist_single_component_adsorption",
        "NIST_BINARY_MIXTURE_ADSORPTION": "nist_binary_mixture_adsorption",
        "ADSORBATES": "adsorbates",
        "ADSORBENTS": "adsorbents",
        "TRAINING_DATASET": "training_dataset",
        "TRAINING_METADATA": "training_metadata",
    }
    experiment_columns = [
        fitting_name_column,
        COLUMN_ADSORBENT,
        COLUMN_ADSORBATE,
        COLUMN_TEMPERATURE_K,
        COLUMN_PRESSURE_PA,
        COLUMN_UPTAKE_MOL_G,
        COLUMN_MEASUREMENT_COUNT,
        COLUMN_MIN_PRESSURE,
        COLUMN_MAX_PRESSURE,
        COLUMN_MIN_UPTAKE,
        COLUMN_MAX_UPTAKE,
    ]

    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_table_name(cls, table_name: str) -> str:
        return cls.table_aliases.get(table_name, table_name)

    # -------------------------------------------------------------------------
    @classmethod
    def fitting_tables(cls) -> set[str]:
        return {cls.processed_table, cls.best_fit_table}.union(
            {schema["table"] for schema in MODEL_SCHEMAS.values()}
        )

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_material_columns(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        normalized = dataset.copy()
        stripped_names = {column: str(column).strip() for column in normalized.columns}
        if any(source != target for source, target in stripped_names.items()):
            normalized = normalized.rename(columns=stripped_names)
        rename_map: dict[str, str] = {}
        for target, aliases in cls.material_aliases.items():
            if target in normalized.columns:
                continue
            for alias in aliases:
                alias_normalized = alias.strip().lower()
                for column in normalized.columns:
                    value = str(column).strip().lower()
                    if value == alias_normalized:
                        rename_map[column] = target
                        break
                if target in rename_map.values():
                    break
        if rename_map:
            normalized = normalized.rename(columns=rename_map)
        return normalized

    # -------------------------------------------------------------------------
    @classmethod
    def build_processed_key(cls, row: pd.Series) -> str:
        payload = {
            COLUMN_ADSORBENT: row.get(COLUMN_ADSORBENT, ""),
            COLUMN_ADSORBATE: row.get(COLUMN_ADSORBATE, ""),
            COLUMN_TEMPERATURE_K: row.get(COLUMN_TEMPERATURE_K),
            COLUMN_PRESSURE_PA: row.get(COLUMN_PRESSURE_PA),
            COLUMN_UPTAKE_MOL_G: row.get(COLUMN_UPTAKE_MOL_G),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @classmethod
    def add_processed_keys(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        normalized = dataset.copy()
        if normalized.empty:
            normalized[cls.processed_key_column] = pd.Series(dtype="string")
            return normalized
        normalized[cls.processed_key_column] = normalized.apply(
            cls.build_processed_key, axis=1
        )
        return normalized

    # -------------------------------------------------------------------------
    @classmethod
    def prepare_for_storage(
        cls, dataset: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        normalized = cls.normalize_table_name(table_name)
        storage = dataset.copy()
        if normalized == cls.raw_table:
            storage = cls.normalize_material_columns(storage)
            if (
                COLUMN_DATASET_NAME in storage.columns
                and cls.raw_name_column in storage.columns
            ):
                storage = storage.drop(columns=[COLUMN_DATASET_NAME], errors="ignore")
            else:
                storage = storage.rename(
                    columns={COLUMN_DATASET_NAME: cls.raw_name_column}
                )
        elif normalized in cls.fitting_tables():
            if (
                COLUMN_EXPERIMENT_NAME in storage.columns
                and cls.fitting_name_column in storage.columns
            ):
                storage = storage.drop(
                    columns=[COLUMN_EXPERIMENT_NAME], errors="ignore"
                )
            else:
                storage = storage.rename(
                    columns={COLUMN_EXPERIMENT_NAME: cls.fitting_name_column}
                )
            if normalized == cls.processed_table:
                storage = cls.normalize_material_columns(storage)
                storage = storage.drop(columns=[COLUMN_EXPERIMENT], errors="ignore")
                storage = cls.add_processed_keys(storage)
        return storage

    # -------------------------------------------------------------------------
    @classmethod
    def restore_from_storage(
        cls, dataset: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        return dataset.copy()

    # -------------------------------------------------------------------------
    def save_raw_dataset(self, dataset: pd.DataFrame) -> None:
        table_name = self.raw_table
        storage_dataset = self.prepare_for_storage(dataset, table_name)
        self.queries.upsert_table(storage_dataset, table_name)

    # -------------------------------------------------------------------------
    def delete_raw_dataset(self, dataset_name: str) -> bool:
        dataset_name = str(dataset_name or "").strip()
        if not dataset_name:
            return False

        existing = self.queries.load_table(self.raw_table)
        if existing.empty or self.raw_name_column not in existing.columns:
            return False

        filtered = existing[existing[self.raw_name_column] != dataset_name].copy()
        if len(filtered) == len(existing):
            return False

        self.queries.save_table(filtered, self.raw_table)
        return True

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        normalized = self.normalize_table_name(table_name)
        loaded = self.queries.load_table(normalized, limit=limit, offset=offset)
        return self.restore_from_storage(loaded, normalized)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        normalized = self.normalize_table_name(table_name)
        storage_dataset = self.prepare_for_storage(dataset, normalized)
        self.queries.upsert_table(storage_dataset, normalized)

    # -------------------------------------------------------------------------
    def save_processed_dataset(self, dataset: pd.DataFrame) -> None:
        self.upsert_table(dataset, self.processed_table)

    # -------------------------------------------------------------------------
    def save_fitting_results(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        experiments = self.build_experiment_frame(dataset)
        self.upsert_table(experiments, self.processed_table)
        for schema in MODEL_SCHEMAS.values():
            model_frame = self.build_model_frame(dataset, schema)
            if model_frame is None:
                continue
            self.upsert_table(model_frame, schema["table"])

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        experiments = self.load_table(self.processed_table)
        if experiments.empty:
            return experiments
        combined = experiments.rename(
            columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
        )
        if COLUMN_EXPERIMENT_NAME not in combined.columns:
            return combined
        for schema in MODEL_SCHEMAS.values():
            model_frame = self.load_table(schema["table"])
            if model_frame.empty:
                continue
            model_frame = model_frame.rename(
                columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
            )
            if COLUMN_EXPERIMENT_NAME not in model_frame.columns:
                continue
            renamed = self.rename_model_columns(model_frame, schema)
            combined = combined.merge(renamed, how="left", on=COLUMN_EXPERIMENT_NAME)
        return combined

    # -------------------------------------------------------------------------
    def save_best_fit(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for best fit results.")
        best = pd.DataFrame()
        best[COLUMN_EXPERIMENT_NAME] = dataset.get(COLUMN_EXPERIMENT_NAME)
        best[COLUMN_BEST_MODEL] = dataset.get(COLUMN_BEST_MODEL)
        best[COLUMN_WORST_MODEL] = dataset.get(COLUMN_WORST_MODEL)
        self.upsert_table(best, self.best_fit_table)

    # -------------------------------------------------------------------------
    def load_best_fit(self) -> pd.DataFrame:
        best = self.load_table(self.best_fit_table)
        if best.empty:
            return best
        best = best.rename(columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME})
        if COLUMN_EXPERIMENT_NAME not in best.columns:
            return best
        experiments = self.load_table(self.processed_table)
        if experiments.empty:
            return pd.DataFrame()
        experiments = experiments.rename(
            columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
        )
        drop_columns = [COLUMN_ID]
        merged = experiments.merge(
            best.drop(columns=drop_columns, errors="ignore"),
            how="left",
            on=COLUMN_EXPERIMENT_NAME,
        )
        return merged

    # -------------------------------------------------------------------------
    def build_experiment_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for fitting results.")
        experiments = dataset.copy()
        experiments[self.fitting_name_column] = experiments[COLUMN_EXPERIMENT_NAME]
        for column in self.experiment_columns:
            if column not in experiments.columns:
                experiments[column] = pd.NA
        experiments = experiments.loc[:, self.experiment_columns].copy()
        return experiments

    # -------------------------------------------------------------------------
    def resolve_dataset_column(
        self, prefix: str, suffix: str, columns: list[str] | pd.Index
    ) -> str | None:
        target = f"{prefix} {suffix}".lower()
        for column in columns:
            if str(column).lower() == target:
                return column
        return None

    # -------------------------------------------------------------------------
    def build_model_frame(
        self,
        dataset: pd.DataFrame,
        schema: dict[str, Any],
    ) -> pd.DataFrame | None:
        resolved = {
            field: self.resolve_dataset_column(
                schema["prefix"], suffix, dataset.columns
            )
            for field, suffix in schema["fields"].items()
        }
        if all(column is None for column in resolved.values()):
            return None
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for fitting results.")
        model_frame = pd.DataFrame()
        model_frame[self.fitting_name_column] = dataset.get(COLUMN_EXPERIMENT_NAME)
        for field, column in resolved.items():
            target = schema["fields"][field]
            if column is None:
                model_frame[target] = pd.NA
            else:
                model_frame[target] = dataset[column]
        return model_frame

    # -------------------------------------------------------------------------
    def rename_model_columns(
        self, model_frame: pd.DataFrame, schema: dict[str, Any]
    ) -> pd.DataFrame:
        rename_map = {
            column_name: f"{schema['prefix']} {column_name}"
            for column_name in schema["fields"].values()
        }
        trimmed = model_frame.rename(columns=rename_map)
        drop_columns = [COLUMN_ID, "experiment_id"]
        return trimmed.drop(columns=drop_columns, errors="ignore")


###############################################################################


