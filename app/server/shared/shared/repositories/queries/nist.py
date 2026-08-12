from __future__ import annotations

import pandas as pd

from shared.common.settings import get_server_settings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Adsorbate, Adsorbent, Dataset, Isotherm, IsothermComponent, Observation
from sqlalchemy import func, select

NIST_DATASET_NAME = "NIST ISODB"

###############################################################################
class NISTDataSerializer:
    """Read the canonical NIST collection for training/inference consumers."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager | None = None) -> None:
        self.database = database or DatabaseManager(get_server_settings().database)

    # -------------------------------------------------------------------------
    def count_nist_rows(self) -> dict[str, int]:
        with self.database.session_factory() as session:
            experiments_count = session.scalar(
                select(func.count(Isotherm.id))
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            single_component_rows = session.scalar(
                select(func.count(Observation.id))
                .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            guest_rows = session.scalar(
                select(func.count(func.distinct(Adsorbate.id)))
                .join(IsothermComponent, IsothermComponent.adsorbate_id == Adsorbate.id)
                .join(Observation, Observation.component_id == IsothermComponent.id)
                .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            host_rows = session.scalar(
                select(func.count(func.distinct(Adsorbent.id)))
                .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
        return {
            "experiments_count": int(experiments_count or 0),
            "single_component_rows": int(single_component_rows or 0),
            "binary_mixture_rows": 0,
            "guest_rows": int(guest_rows or 0),
            "host_rows": int(host_rows or 0),
        }

    # -------------------------------------------------------------------------
    def load_adsorption_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(Isotherm.external_key, Isotherm.temperature_k, Adsorbent.name.label("adsorbent"), Adsorbate.name.label("adsorbate"), Observation.pressure_canonical.label("pressure"), Observation.uptake_mol_kg.label("adsorbed_amount"))
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .join(Observation, Observation.isotherm_id == Isotherm.id)
                .join(IsothermComponent, IsothermComponent.id == Observation.component_id)
                .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                .where(Dataset.source == "nist")
                .order_by(Isotherm.id, Observation.sequence_index)
            ).mappings()
            guests = session.scalars(select(Adsorbate)).all()
            hosts = session.scalars(select(Adsorbent)).all()
        adsorption = pd.DataFrame([dict(row) for row in rows])
        guest = pd.DataFrame([{"name": item.name, "InChIKey": item.inchi_key, "molecular_weight": item.molar_mass_g_mol, "molecular_formula": item.formula, "smile_code": item.smiles} for item in guests])
        host = pd.DataFrame([{"name": item.name, "hashkey": item.external_identifier, "molecular_weight": item.molar_mass_g_mol, "molecular_formula": item.formula, "smile_code": item.smiles} for item in hosts])
        return adsorption, guest, host
