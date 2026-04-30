from __future__ import annotations

from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.server.repositories.schemas.types import JSONSequence


###############################################################################
class Base(DeclarativeBase):
    pass


###############################################################################
class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("dataset_name"),
        CheckConstraint("source IN ('uploaded', 'nist')"),
        Index("ix_datasets_source", "source"),
    )


###############################################################################
class Adsorbate(Base):
    __tablename__ = "adsorbates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adsorbate_key: Mapped[str] = mapped_column(String, nullable=False)
    InChIKey: Mapped[str | None] = mapped_column(String)
    name: Mapped[str | None] = mapped_column(String)
    InChICode: Mapped[str | None] = mapped_column(String)
    formula: Mapped[str | None] = mapped_column(String)
    molecular_weight: Mapped[float | None] = mapped_column(Float)
    molecular_formula: Mapped[str | None] = mapped_column(String)
    smile_code: Mapped[str | None] = mapped_column(String)
    __table_args__ = (
        UniqueConstraint("adsorbate_key"),
        UniqueConstraint("InChIKey"),
        Index("ix_adsorbates_name", "name"),
    )


###############################################################################
class Adsorbent(Base):
    __tablename__ = "adsorbents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adsorbent_key: Mapped[str] = mapped_column(String, nullable=False)
    hashkey: Mapped[str | None] = mapped_column(String)
    name: Mapped[str | None] = mapped_column(String)
    formula: Mapped[str | None] = mapped_column(String)
    molecular_weight: Mapped[float | None] = mapped_column(Float)
    molecular_formula: Mapped[str | None] = mapped_column(String)
    smile_code: Mapped[str | None] = mapped_column(String)
    __table_args__ = (
        UniqueConstraint("adsorbent_key"),
        UniqueConstraint("hashkey"),
        Index("ix_adsorbents_name", "name"),
    )


###############################################################################
class AdsorptionIsotherm(Base):
    __tablename__ = "adsorption_isotherms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    source_record_id: Mapped[str | None] = mapped_column(String)
    experiment_name: Mapped[str] = mapped_column(String, nullable=False)
    adsorbent_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("adsorbents.id", ondelete="RESTRICT"), nullable=False
    )
    temperature_k: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_units: Mapped[str | None] = mapped_column(String)
    adsorption_units: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("experiment_name"),
        UniqueConstraint(
            "dataset_id", "source_record_id", "adsorbent_id", "temperature_k"
        ),
        Index("ix_adsorption_isotherms_dataset_id", "dataset_id"),
        Index("ix_adsorption_isotherms_adsorbent_id", "adsorbent_id"),
    )


###############################################################################
class AdsorptionIsothermComponent(Base):
    __tablename__ = "adsorption_isotherm_components"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isotherm_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_isotherms.id", ondelete="CASCADE"),
        nullable=False,
    )
    component_index: Mapped[int] = mapped_column(Integer, nullable=False)
    adsorbate_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("adsorbates.id", ondelete="RESTRICT"), nullable=False
    )
    mole_fraction: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        UniqueConstraint("isotherm_id", "component_index"),
        UniqueConstraint("isotherm_id", "adsorbate_id"),
        CheckConstraint("component_index IN (1, 2)"),
        Index("ix_adsorption_isotherm_components_isotherm_id", "isotherm_id"),
        Index("ix_adsorption_isotherm_components_adsorbate_id", "adsorbate_id"),
    )


###############################################################################
class AdsorptionPoint(Base):
    __tablename__ = "adsorption_points"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isotherm_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_isotherms.id", ondelete="CASCADE"),
        nullable=False,
    )
    point_index: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        UniqueConstraint("isotherm_id", "point_index"),
        Index("ix_adsorption_points_isotherm_id", "isotherm_id"),
    )


###############################################################################
class AdsorptionPointComponent(Base):
    __tablename__ = "adsorption_point_components"
    point_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_points.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    component_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_isotherm_components.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    partial_pressure_pa: Mapped[float] = mapped_column(Float, nullable=False)
    uptake_mol_g: Mapped[float] = mapped_column(Float, nullable=False)
    original_pressure: Mapped[float | None] = mapped_column(Float)
    original_uptake: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        Index("ix_adsorption_point_components_point_id", "point_id"),
        Index("ix_adsorption_point_components_component_id", "component_id"),
    )


###############################################################################
class AdsorptionProcessedIsotherm(Base):
    __tablename__ = "adsorption_processed_isotherms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isotherm_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_isotherms.id", ondelete="CASCADE"),
        nullable=False,
    )
    processing_version: Mapped[str] = mapped_column(String, nullable=False, default="v1")
    processed_key: Mapped[str] = mapped_column(String, nullable=False)
    pressure_pa_series: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    uptake_mol_g_series: Mapped[Any] = mapped_column(JSONSequence, nullable=False)
    original_pressure_series: Mapped[Any | None] = mapped_column(JSONSequence)
    original_uptake_series: Mapped[Any | None] = mapped_column(JSONSequence)
    measurement_count: Mapped[int | None] = mapped_column(Integer)
    min_pressure: Mapped[float | None] = mapped_column(Float)
    max_pressure: Mapped[float | None] = mapped_column(Float)
    min_uptake: Mapped[float | None] = mapped_column(Float)
    max_uptake: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        UniqueConstraint("processed_key"),
        UniqueConstraint("isotherm_id", "processing_version"),
        Index("ix_adsorption_processed_isotherms_isotherm_id", "isotherm_id"),
    )


###############################################################################
class AdsorptionFit(Base):
    __tablename__ = "adsorption_fits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processed_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_processed_isotherms.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    optimization_method: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    aic: Mapped[float | None] = mapped_column(Float)
    aicc: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("processed_id", "model_name", "optimization_method"),
        Index("ix_adsorption_fits_processed_id", "processed_id"),
        Index("ix_adsorption_fits_model_name", "model_name"),
    )


###############################################################################
class AdsorptionFitParam(Base):
    __tablename__ = "adsorption_fit_params"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("adsorption_fits.id", ondelete="CASCADE"), nullable=False
    )
    param_name: Mapped[str] = mapped_column(String, nullable=False)
    param_value: Mapped[float] = mapped_column(Float, nullable=False)
    param_error: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        UniqueConstraint("fit_id", "param_name"),
        Index("ix_adsorption_fit_params_fit_id", "fit_id"),
    )


###############################################################################
class AdsorptionBestFit(Base):
    __tablename__ = "adsorption_best_fit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processed_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("adsorption_processed_isotherms.id", ondelete="CASCADE"),
        nullable=False,
    )
    best_fit_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("adsorption_fits.id", ondelete="SET NULL")
    )
    worst_fit_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("adsorption_fits.id", ondelete="SET NULL")
    )
    best_model: Mapped[str | None] = mapped_column(String)
    worst_model: Mapped[str | None] = mapped_column(String)
    __table_args__ = (
        UniqueConstraint("processed_id"),
        Index("ix_adsorption_best_fit_best_fit_id", "best_fit_id"),
        Index("ix_adsorption_best_fit_worst_fit_id", "worst_fit_id"),
    )


###############################################################################
class TrainingMetadata(Base):
    __tablename__ = "training_metadata"
    hashcode: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_label: Mapped[str] = mapped_column(String, nullable=False, default="default")
    created_at: Mapped[str | None] = mapped_column(String)
    sample_size: Mapped[float | None] = mapped_column(Float)
    validation_size: Mapped[float | None] = mapped_column(Float)
    min_measurements: Mapped[int | None] = mapped_column(Integer)
    max_measurements: Mapped[int | None] = mapped_column(Integer)
    smile_sequence_size: Mapped[int | None] = mapped_column(Integer)
    max_pressure: Mapped[float | None] = mapped_column(Float)
    max_uptake: Mapped[float | None] = mapped_column(Float)
    total_samples: Mapped[int | None] = mapped_column(Integer)
    train_samples: Mapped[int | None] = mapped_column(Integer)
    validation_samples: Mapped[int | None] = mapped_column(Integer)
    smile_vocabulary: Mapped[Any | None] = mapped_column(JSONSequence)
    adsorbent_vocabulary: Mapped[Any | None] = mapped_column(JSONSequence)
    normalization_stats: Mapped[Any | None] = mapped_column(JSONSequence)
    __table_args__ = (
        UniqueConstraint("hashcode"),
        Index("ix_training_metadata_dataset_label", "dataset_label"),
    )


###############################################################################
class TrainingDataset(Base):
    __tablename__ = "training_dataset"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, default="default")
    source_dataset: Mapped[str | None] = mapped_column(String)
    split: Mapped[str | None] = mapped_column(String)
    temperature: Mapped[float | None] = mapped_column(Float)
    pressure: Mapped[Any | None] = mapped_column(JSONSequence)
    adsorbed_amount: Mapped[Any | None] = mapped_column(JSONSequence)
    encoded_adsorbent: Mapped[int | None] = mapped_column(Integer)
    adsorbate_molecular_weight: Mapped[float | None] = mapped_column(Float)
    adsorbate_encoded_smile: Mapped[Any | None] = mapped_column(JSONSequence)
    training_hashcode: Mapped[str | None] = mapped_column(
        String, ForeignKey("training_metadata.hashcode", ondelete="SET NULL")
    )
    sample_key: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("sample_key"),
        Index("ix_training_dataset_name", "name"),
        Index("ix_training_dataset_training_hashcode", "training_hashcode"),
    )
