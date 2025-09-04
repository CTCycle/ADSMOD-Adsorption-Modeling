from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

from NISTADS.app.client.workers import check_thread_status, update_progress_callback
from NISTADS.app.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.molecular_identifier = "InChIKey"
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def remove_duplicates_without_identifiers(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.molecular_identifier in data.columns:
            data = (
                data.assign(_has_id=data["InChIKey"].notna())
                .sort_values(["name", "_has_id"], ascending=[True, False])
                .drop_duplicates("name")
                .drop("_has_id", axis=1)
            )
        else:
            data = data.drop_duplicates("name")
        return data

    # -------------------------------------------------------------------------
    def extract_fetched_properties(
        self, data: pd.DataFrame, properties: dict
    ) -> None | pd.DataFrame:
        if not properties:
            return

        properties["name"] = [x.lower() for x in properties["name"]]
        data["name"] = data["name"].str.lower()
        indexed_properties = pd.DataFrame(properties).set_index("name")
        indexed_data = data.set_index("name")
        indexed_data.update(indexed_properties)
        dataset = indexed_data.reset_index()

        return dataset

    # -------------------------------------------------------------------------
    def fetch_guest_properties(
        self, experiments: pd.DataFrame, data: pd.DataFrame, **kwargs
    ) -> None | pd.DataFrame:
        compound_properties = CompoundProperties(
            self.configuration, compound_type="adsorbate"
        )
        guest_names = (
            pd.concat([experiments["adsorbate_name"].dropna(), data["name"].dropna()])
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        properties = compound_properties.get_properties_for_multiple_compounds(
            guest_names,
            worker=kwargs.get("worker", None),
            progress_callback=kwargs.get("progress_callback", None),
        )

        dataset = self.extract_fetched_properties(data, properties)

        return dataset

    # -------------------------------------------------------------------------
    def fetch_host_properties(self, experiments, data, **kwargs) -> None | pd.DataFrame:
        compound_properties = CompoundProperties(
            self.configuration, compound_type="adsorbent"
        )
        host_names = (
            pd.concat([experiments["adsorbent_name"].dropna(), data["name"].dropna()])
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        properties = compound_properties.get_properties_for_multiple_compounds(
            host_names,
            worker=kwargs.get("worker", None),
            progress_callback=kwargs.get("progress_callback", None),
        )

        dataset = self.extract_fetched_properties(data, properties)
        return dataset


###############################################################################
class CompoundProperties:
    def __init__(self, configuration: dict, compound_type: str = "adsorbate") -> None:
        self.configuration = configuration
        self.compound_type = compound_type.lower()
        prefix = self.compound_type
        self.properties = {
            "name": [],
            f"{prefix}_molecular_weight": [],
            f"{prefix}_molecular_formula": [],
            f"{prefix}_SMILE": [],
        }

    # -------------------------------------------------------------------------
    def is_chemical_formula(self, string: str) -> bool:
        formula_pattern = r"^[A-Za-z0-9\[\](){}·.,+\-_/]+$"
        return bool(re.match(formula_pattern, string))

    # -------------------------------------------------------------------------
    def get_molecular_properties(
        self, identifier: str | np.ndarray, namespace: str
    ) -> pd.Series | pcp.Compound | None:
        try:
            compounds = pcp.get_compounds(
                identifier, namespace=namespace, list_return="flat"
            )            
            return compounds[0]
        except Exception:
            logger.error(
                f"Cannot fetch molecules properties for {identifier}]"
            )
            return

    # -------------------------------------------------------------------------
    def get_properties_for_multiple_compounds(
        self, names: list[str] | np.ndarray, **kwargs
    ) -> dict[str, list]:
        for i, name in enumerate(tqdm(names, total=len(names))):
            # Optionally check for chemical formula
            # is_formula = self.is_chemical_formula(name)
            compound = self.get_molecular_properties(name, namespace="name")
            if compound is not None:
                self.extract_properties(name, compound)

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(names), kwargs.get("progress_callback", None)
            )

        return self.properties

    # -------------------------------------------------------------------------
    def extract_properties(
        self, name: str | np.ndarray, compound: pd.Series | pcp.Compound | None
    ) -> None:
        molecular_weight = float(getattr(compound, "molecular_weight", np.nan))
        molecular_formula = getattr(compound, "molecular_formula", np.nan)
        SMILE = np.nan

        if not getattr(compound, "isomeric_smiles", None):
            records = getattr(compound, "record", None)
            # SMILES are now fetched from record/props within the compound object
            # props is a list of dictionary with structure:
            # {urns : {properties : val}, value : {ival : val}}
            internal_properties = records.get("props", []) if records else []
            formatted_properties = {}
            for p in internal_properties:
                urns = p.get("urn", {})
                label = urns.get("label", np.nan)
                prop_name = urns.get("name", np.nan)
                value = next(iter(p.get("value", {}).values()), np.nan)
                formatted_properties[f"{label}_{prop_name}"] = value

            SMILE = formatted_properties.get("SMILES_Absolute", np.nan)

        self.properties["name"].append(name)
        self.properties[f"{self.compound_type}_molecular_weight"].append(
            molecular_weight
        )
        self.properties[f"{self.compound_type}_molecular_formula"].append(
            molecular_formula
        )
        self.properties[f"{self.compound_type}_SMILE"].append(SMILE)
