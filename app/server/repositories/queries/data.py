from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.server.repositories.database.backend import ADSMODDatabase, database
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
class DataRepositoryQueries:
    def __init__(self, db: ADSMODDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return self.database.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_dataset_by_name(session: Session, dataset_name: str) -> Dataset | None:
        return session.execute(
            select(Dataset).where(Dataset.dataset_name == dataset_name)
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_adsorbate_by_inchi(
        session: Session, normalized_inchi: str
    ) -> Adsorbate | None:
        return session.execute(
            select(Adsorbate).where(Adsorbate.InChIKey == normalized_inchi)
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_adsorbate_by_key(session: Session, adsorbate_key: str) -> Adsorbate | None:
        return session.execute(
            select(Adsorbate).where(Adsorbate.adsorbate_key == adsorbate_key)
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_adsorbent_by_hash(
        session: Session, normalized_hash: str
    ) -> Adsorbent | None:
        return session.execute(
            select(Adsorbent).where(Adsorbent.hashkey == normalized_hash)
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_adsorbent_by_key(session: Session, adsorbent_key: str) -> Adsorbent | None:
        return session.execute(
            select(Adsorbent).where(Adsorbent.adsorbent_key == adsorbent_key)
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_isotherm_component(
        session: Session,
        isotherm_id: int,
        component_index: int,
    ) -> AdsorptionIsothermComponent | None:
        return session.execute(
            select(AdsorptionIsothermComponent).where(
                and_(
                    AdsorptionIsothermComponent.isotherm_id == isotherm_id,
                    AdsorptionIsothermComponent.component_index == component_index,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_isotherm_by_experiment_name(
        session: Session, experiment_name: str
    ) -> AdsorptionIsotherm | None:
        return session.execute(
            select(AdsorptionIsotherm).where(
                AdsorptionIsotherm.experiment_name == experiment_name
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_point(
        session: Session,
        isotherm_id: int,
        point_index: int,
    ) -> AdsorptionPoint | None:
        return session.execute(
            select(AdsorptionPoint).where(
                and_(
                    AdsorptionPoint.isotherm_id == isotherm_id,
                    AdsorptionPoint.point_index == point_index,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_point_component(
        session: Session,
        point_id: int,
        component_id: int,
    ) -> AdsorptionPointComponent | None:
        return session.execute(
            select(AdsorptionPointComponent).where(
                and_(
                    AdsorptionPointComponent.point_id == point_id,
                    AdsorptionPointComponent.component_id == component_id,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_uploaded_dataset_by_name(
        session: Session,
        dataset_name: str,
    ) -> Dataset | None:
        return session.execute(
            select(Dataset).where(
                and_(
                    Dataset.dataset_name == dataset_name,
                    Dataset.source == "uploaded",
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def load_uploaded_raw_rows(session: Session) -> list[Any]:
        return session.execute(
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
                    AdsorptionIsothermComponent.isotherm_id == AdsorptionIsotherm.id,
                    AdsorptionIsothermComponent.component_index == 1,
                ),
            )
            .join(Adsorbate, Adsorbate.id == AdsorptionIsothermComponent.adsorbate_id)
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
            .where(
                and_(
                    Dataset.source == "uploaded",
                    ~Dataset.dataset_name.like("archived::%"),
                )
            )
            .order_by(
                Dataset.dataset_name,
                AdsorptionIsotherm.source_record_id,
                AdsorptionPoint.point_index,
            )
        ).all()

    # -------------------------------------------------------------------------
    @staticmethod
    def load_processed_rows(session: Session) -> list[Any]:
        return session.execute(
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
                    AdsorptionIsothermComponent.isotherm_id == AdsorptionIsotherm.id,
                    AdsorptionIsothermComponent.component_index == 1,
                ),
            )
            .join(Adsorbate, Adsorbate.id == AdsorptionIsothermComponent.adsorbate_id)
            .join(Adsorbent, Adsorbent.id == AdsorptionIsotherm.adsorbent_id)
            .order_by(AdsorptionProcessedIsotherm.id)
        ).all()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_processed_by_isotherm_and_version(
        session: Session,
        isotherm_id: int,
        processing_version: str,
    ) -> AdsorptionProcessedIsotherm | None:
        return session.execute(
            select(AdsorptionProcessedIsotherm).where(
                and_(
                    AdsorptionProcessedIsotherm.isotherm_id == isotherm_id,
                    AdsorptionProcessedIsotherm.processing_version == processing_version,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_fit_by_processed_model_method(
        session: Session,
        processed_id: int,
        model_name: str,
        optimization_method: str,
    ) -> AdsorptionFit | None:
        return session.execute(
            select(AdsorptionFit).where(
                and_(
                    AdsorptionFit.processed_id == processed_id,
                    AdsorptionFit.model_name == model_name,
                    AdsorptionFit.optimization_method == optimization_method,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_fit_param(
        session: Session,
        fit_id: int,
        param_name: str,
    ) -> AdsorptionFitParam | None:
        return session.execute(
            select(AdsorptionFitParam).where(
                and_(
                    AdsorptionFitParam.fit_id == fit_id,
                    AdsorptionFitParam.param_name == param_name,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_fit_by_processed_and_model(
        session: Session,
        processed_id: int,
        model_name: str,
    ) -> AdsorptionFit | None:
        return session.execute(
            select(AdsorptionFit).where(
                and_(
                    AdsorptionFit.processed_id == processed_id,
                    AdsorptionFit.model_name == model_name,
                )
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_best_fit_by_processed(
        session: Session,
        processed_id: int,
    ) -> AdsorptionBestFit | None:
        return session.execute(
            select(AdsorptionBestFit).where(
                AdsorptionBestFit.processed_id == processed_id
            )
        ).scalar_one_or_none()
