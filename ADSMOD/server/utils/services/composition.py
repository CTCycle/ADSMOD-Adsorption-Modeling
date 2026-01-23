from __future__ import annotations

import asyncio
import hashlib
from difflib import get_close_matches
from typing import Any

import pandas as pd

from ADSMOD.server.database.database import database
from ADSMOD.server.utils.configurations import server_settings
from ADSMOD.server.utils.constants import COLUMN_DATASET_NAME
from ADSMOD.server.utils.repository.isodb import NISTDataSerializer
from ADSMOD.server.utils.repository.serializer import DataSerializer
from ADSMOD.server.utils.services.conversion import PQ_units_conversion
from ADSMOD.server.utils.services.nistads import PubChemClient


###############################################################################
class DatasetCompositionService:
    def __init__(self) -> None:
        self.serializer = DataSerializer()
        self.nist_serializer = NISTDataSerializer()
        self.upload_column_aliases = {
            "filename": ["filename", "experiment", "experiment name", "sample", "run"],
            "temperature": ["temperature", "temp", "temperature [k]", "temp [k]"],
            "pressure": ["pressure", "pressure [pa]", "pressure(pa)", "p"],
            "adsorbed_amount": [
                "uptake",
                "adsorbed_amount",
                "adsorbed amount",
                "adsorption",
                "q",
                "uptake [mol/g]",
            ],
            "adsorbate_name": [
                "adsorbate",
                "adsorbate name",
                "adsorbate_name",
                "guest",
                "gas",
            ],
            "adsorbent_name": [
                "adsorbent",
                "adsorbent name",
                "adsorbent_name",
                "host",
                "material",
            ],
        }
        self.required_columns = [
            "filename",
            "temperature",
            "pressure",
            "adsorbed_amount",
            "adsorbate_name",
            "adsorbent_name",
        ]

    # -------------------------------------------------------------------------
    def list_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        nist_rows = database.count_rows("NIST_SINGLE_COMPONENT_ADSORPTION")
        if nist_rows > 0:
            sources.append(
                {
                    "source": "nist",
                    "dataset_name": "NIST_SINGLE_COMPONENT_ADSORPTION",
                    "display_name": "NIST Single Component",
                    "row_count": nist_rows,
                }
            )

        adsorption = self.serializer.load_table("ADSORPTION_DATA")
        if not adsorption.empty and COLUMN_DATASET_NAME in adsorption.columns:
            name_series = (
                adsorption[COLUMN_DATASET_NAME]
                .fillna("")
                .astype("string")
                .str.strip()
            )
            counts = name_series.value_counts()
            for name, count in counts.items():
                if not name:
                    continue
                sources.append(
                    {
                        "source": "uploaded",
                        "dataset_name": str(name),
                        "display_name": str(name),
                        "row_count": int(count),
                    }
                )

        nist_entries = [entry for entry in sources if entry["source"] == "nist"]
        upload_entries = sorted(
            [entry for entry in sources if entry["source"] == "uploaded"],
            key=lambda entry: entry["display_name"].lower(),
        )
        return nist_entries + upload_entries

    # -------------------------------------------------------------------------
    def compose_datasets(
        self, selections: list[dict[str, Any]]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        if not selections:
            raise ValueError("No datasets were selected for processing.")

        frames: list[pd.DataFrame] = []
        labels: list[str] = []
        for selection in selections:
            source = str(selection.get("source", "")).strip().lower()
            dataset_name = str(selection.get("dataset_name", "")).strip()
            if not dataset_name:
                raise ValueError("Dataset selection is missing a dataset name.")

            if source == "nist":
                adsorption = self.nist_serializer.load_adsorption_datasets()[0]
                if adsorption.empty:
                    raise ValueError("NIST single-component dataset is not available.")
                dataset_frame = self.standardize_nist_dataset(
                    adsorption, dataset_name
                )
            elif source == "uploaded":
                dataset_frame = self.standardize_uploaded_dataset(dataset_name)
            else:
                raise ValueError(f"Unsupported dataset source: {source}")

            frames.append(dataset_frame)
            labels.append(dataset_name)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            raise ValueError("No adsorption data was available after composition.")

        combined = self.ensure_required_columns(combined)
        guest_data, host_data = self.ensure_materials(combined)
        dataset_label = self.compose_dataset_label(labels)

        return combined, guest_data, host_data, dataset_label

    # -------------------------------------------------------------------------
    def standardize_uploaded_dataset(self, dataset_name: str) -> pd.DataFrame:
        raw = self.serializer.load_table("ADSORPTION_DATA")
        if raw.empty or COLUMN_DATASET_NAME not in raw.columns:
            raise ValueError("No uploaded datasets are available.")

        filtered = raw[raw[COLUMN_DATASET_NAME] == dataset_name].copy()
        if filtered.empty:
            raise ValueError(f"Uploaded dataset '{dataset_name}' was not found.")

        resolved = self.resolve_uploaded_columns(filtered.columns)
        rename_map = {value: key for key, value in resolved.items() if value}
        standardized = filtered.rename(columns=rename_map).copy()

        if "filename" not in standardized.columns:
            standardized["filename"] = [
                f"{dataset_name}_{idx}" for idx in range(len(standardized))
            ]

        missing = [col for col in self.required_columns if col not in standardized.columns]
        if missing:
            raise ValueError(
                "Uploaded dataset is missing required columns: "
                + ", ".join(missing)
            )

        for column in ("adsorbate_name", "adsorbent_name"):
            if column in standardized.columns:
                standardized[column] = (
                    standardized[column].astype("string").str.strip().str.lower()
                )

        standardized = self.normalize_measurements(standardized)
        standardized["pressureUnits"] = "pa"
        standardized["adsorptionUnits"] = "mol/g"
        standardized[COLUMN_DATASET_NAME] = dataset_name
        standardized = self.ensure_filename_prefix(
            standardized, self.build_dataset_tag("uploaded", dataset_name)
        )
        return standardized

    # -------------------------------------------------------------------------
    def standardize_nist_dataset(
        self, adsorption: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        if adsorption.empty:
            return adsorption
        cleaned = adsorption.copy()
        if "adsorptionUnits" in cleaned.columns:
            units = cleaned["adsorptionUnits"].astype("string").str.strip().str.lower()
            mol_g_mask = units.isin({"mol/g", "mol per g"})
        else:
            mol_g_mask = pd.Series(True, index=cleaned.index)
        for column in ("adsorbate_name", "adsorbent_name", "filename"):
            if column in cleaned.columns:
                cleaned[column] = (
                    cleaned[column].astype("string").str.strip().str.lower()
                )

        cleaned = PQ_units_conversion(cleaned)
        if "adsorbed_amount" in cleaned.columns:
            cleaned.loc[~mol_g_mask, "adsorbed_amount"] = (
                cleaned.loc[~mol_g_mask, "adsorbed_amount"] / 1000.0
            )
        cleaned["pressureUnits"] = "pa"
        cleaned["adsorptionUnits"] = "mol/g"
        cleaned = self.normalize_measurements(cleaned)
        cleaned = self.ensure_filename_prefix(
            cleaned, self.build_dataset_tag("nist", dataset_name)
        )
        return cleaned

    # -------------------------------------------------------------------------
    def resolve_uploaded_columns(self, columns: list[str]) -> dict[str, str | None]:
        resolved: dict[str, str | None] = {}
        for key, aliases in self.upload_column_aliases.items():
            resolved[key] = self.match_column(columns, aliases)
        return resolved

    # -------------------------------------------------------------------------
    def match_column(self, columns: list[str], aliases: list[str]) -> str | None:
        normalized = {column: str(column).strip().lower() for column in columns}
        for alias in aliases:
            for column, value in normalized.items():
                if value == alias:
                    return column
        for alias in aliases:
            for column, value in normalized.items():
                if alias in value:
                    return column
        if aliases:
            matches = get_close_matches(aliases[0], list(normalized.values()), cutoff=0.78)
            if matches:
                match = matches[0]
                for column, value in normalized.items():
                    if value == match:
                        return column
        return None

    # -------------------------------------------------------------------------
    def normalize_measurements(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        cleaned = dataframe.copy()
        for column in ("temperature", "pressure", "adsorbed_amount"):
            if column in cleaned.columns:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        required = [col for col in self.required_columns if col in cleaned.columns]
        cleaned = cleaned.dropna(subset=required).reset_index(drop=True)
        if "temperature" in cleaned.columns:
            cleaned = cleaned[cleaned["temperature"] > 0]
        if "pressure" in cleaned.columns:
            cleaned = cleaned[cleaned["pressure"] >= 0]
        if "adsorbed_amount" in cleaned.columns:
            cleaned = cleaned[cleaned["adsorbed_amount"] >= 0]
        return cleaned.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def ensure_required_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.required_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(
                "Composed dataset is missing required columns: " + ", ".join(missing)
            )
        return dataframe

    # -------------------------------------------------------------------------
    def ensure_filename_prefix(self, dataframe: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if "filename" not in dataframe.columns:
            dataframe["filename"] = [
                f"{prefix}_{idx}" for idx in range(len(dataframe))
            ]
            return dataframe

        dataframe["filename"] = dataframe["filename"].astype("string").str.strip()
        missing_mask = dataframe["filename"].isna() | (dataframe["filename"] == "")
        if missing_mask.any():
            dataframe.loc[missing_mask, "filename"] = [
                f"{prefix}_{idx}" for idx in dataframe.index[missing_mask].tolist()
            ]
        dataframe["filename"] = prefix + "_" + dataframe["filename"].astype(str)
        return dataframe

    # -------------------------------------------------------------------------
    def build_dataset_tag(self, source: str, dataset_name: str) -> str:
        cleaned_name = dataset_name.strip().lower().replace(" ", "_")
        cleaned_source = source.strip().lower().replace(" ", "_")
        label = f"{cleaned_source}_{cleaned_name}" if cleaned_name else cleaned_source
        return label or "dataset"

    # -------------------------------------------------------------------------
    def compose_dataset_label(self, labels: list[str]) -> str:
        cleaned = [label.strip() for label in labels if label.strip()]
        combined = "+".join(cleaned)
        return combined[:120] if combined else "composed"

    # -------------------------------------------------------------------------
    def ensure_materials(
        self, adsorption_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _, guest_data, host_data = self.nist_serializer.load_adsorption_datasets()
        adsorbate_names = self.collect_names(adsorption_data, "adsorbate_name")
        adsorbent_names = self.collect_names(adsorption_data, "adsorbent_name")

        guest_data = self.enrich_materials(
            guest_data,
            adsorbate_names,
            target="adsorbate",
            key_column="InChIKey",
            weight_column="adsorbate_molecular_weight",
            formula_column="adsorbate_molecular_formula",
            smile_column="adsorbate_SMILE",
        )
        host_data = self.enrich_materials(
            host_data,
            adsorbent_names,
            target="adsorbent",
            key_column="hashkey",
            weight_column="adsorbent_molecular_weight",
            formula_column="adsorbent_molecular_formula",
            smile_column="adsorbent_SMILE",
        )

        self.nist_serializer.save_materials_datasets(guest_data, host_data)
        return guest_data, host_data

    # -------------------------------------------------------------------------
    def collect_names(self, dataframe: pd.DataFrame, column: str) -> list[str]:
        if column not in dataframe.columns:
            return []
        names = (
            dataframe[column]
            .dropna()
            .astype("string")
            .str.strip()
            .str.lower()
            .loc[lambda series: series != ""]
            .unique()
            .tolist()
        )
        return sorted(names)

    # -------------------------------------------------------------------------
    def enrich_materials(
        self,
        existing: pd.DataFrame,
        names: list[str],
        target: str,
        key_column: str,
        weight_column: str,
        formula_column: str,
        smile_column: str,
    ) -> pd.DataFrame:
        if not names:
            return existing if isinstance(existing, pd.DataFrame) else pd.DataFrame()

        data = existing.copy() if isinstance(existing, pd.DataFrame) else pd.DataFrame()
        if data.empty:
            data = pd.DataFrame(columns=[key_column, "name"])
        if "name" not in data.columns:
            data["name"] = pd.NA

        data["name"] = data["name"].astype("string").str.strip().str.lower()
        for column in (weight_column, formula_column, smile_column):
            if column not in data.columns:
                data[column] = pd.NA

        missing_names = [name for name in names if name not in data["name"].tolist()]
        incomplete_mask = data["name"].isin(names) & (
            data[weight_column].isna() | data[smile_column].isna()
        )
        incomplete_names = data.loc[incomplete_mask, "name"].tolist()
        names_to_fetch = sorted(set(missing_names + incomplete_names))

        if names_to_fetch:
            properties = self.fetch_pubchem_properties(names_to_fetch)
            properties_frame = pd.DataFrame(properties)
            if not properties_frame.empty:
                properties_frame["name"] = (
                    properties_frame["name"]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                )
                properties_frame = properties_frame.rename(
                    columns={
                        "molecular_weight": weight_column,
                        "molecular_formula": formula_column,
                        "smile": smile_column,
                    }
                )
            else:
                properties_frame = pd.DataFrame(columns=["name", weight_column, formula_column, smile_column])
            data = self.merge_material_properties(
                data,
                properties_frame,
                names_to_fetch,
                key_column,
                weight_column,
                formula_column,
                smile_column,
                target,
            )
        return data

    # -------------------------------------------------------------------------
    def fetch_pubchem_properties(self, names: list[str]) -> list[dict[str, Any]]:
        pubchem = PubChemClient(server_settings.nist.pubchem_parallel_tasks)
        return asyncio.run(self.fetch_properties_async(pubchem, names))

    # -------------------------------------------------------------------------
    async def fetch_properties_async(
        self, pubchem: PubChemClient, names: list[str]
    ) -> list[dict[str, Any]]:
        import httpx

        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await pubchem.fetch_properties_for_names(client, names)

    # -------------------------------------------------------------------------
    def merge_material_properties(
        self,
        data: pd.DataFrame,
        properties: pd.DataFrame,
        names: list[str],
        key_column: str,
        weight_column: str,
        formula_column: str,
        smile_column: str,
        target: str,
    ) -> pd.DataFrame:
        if key_column not in data.columns:
            data[key_column] = pd.NA

        new_rows: list[dict[str, Any]] = []
        for name in names:
            existing_row = data[data["name"] == name]
            if not existing_row.empty:
                continue
            new_rows.append(
                {
                    key_column: self.build_material_key(target, name),
                    "name": name,
                }
            )

        if new_rows:
            data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)

        data = data.set_index("name")
        properties = properties.set_index("name")
        data.update(properties)
        updated = data.reset_index()

        for column in (weight_column,):
            if column in updated.columns:
                updated[column] = pd.to_numeric(updated[column], errors="coerce")
        for column in (formula_column, smile_column):
            if column in updated.columns:
                updated[column] = updated[column].astype("string")

        return updated

    # -------------------------------------------------------------------------
    def build_material_key(self, target: str, name: str) -> str:
        key_base = f"{target}:{name}"
        digest = hashlib.sha1(key_base.encode("utf-8")).hexdigest()
        return f"local_{digest[:24]}"
