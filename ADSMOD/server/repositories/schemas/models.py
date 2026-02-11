from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

from ADSMOD.server.repositories.schemas.types import JSONSequence

Base = declarative_base()


###############################################################################
class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String, nullable=False)
    source = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("dataset_name"),
        CheckConstraint("source IN ('uploaded', 'nist')"),
        Index("ix_datasets_source", "source"),
    )


###############################################################################
class Adsorbate(Base):
    __tablename__ = "adsorbates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    adsorbate_key = Column(String, nullable=False)
    InChIKey = Column(String)
    name = Column(String)
    InChICode = Column(String)
    formula = Column(String)
    molecular_weight = Column(Float)
    molecular_formula = Column(String)
    smile_code = Column(String)
    __table_args__ = (
        UniqueConstraint("adsorbate_key"),
        UniqueConstraint("InChIKey"),
        Index("ix_adsorbates_name", "name"),
    )


###############################################################################
class Adsorbent(Base):
    __tablename__ = "adsorbents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    adsorbent_key = Column(String, nullable=False)
    hashkey = Column(String)
    name = Column(String)
    formula = Column(String)
    molecular_weight = Column(Float)
    molecular_formula = Column(String)
    smile_code = Column(String)
    __table_args__ = (
        UniqueConstraint("adsorbent_key"),
        UniqueConstraint("hashkey"),
        Index("ix_adsorbents_name", "name"),
    )


###############################################################################
class AdsorptionIsotherm(Base):
    __tablename__ = "adsorption_isotherms"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    source_record_id = Column(String)
    experiment_name = Column(String, nullable=False)
    adsorbent_id = Column(Integer, ForeignKey("adsorbents.id", ondelete="RESTRICT"), nullable=False)
    temperature_k = Column(Float, nullable=False)
    pressure_units = Column(String)
    adsorption_units = Column(String)
    created_at = Column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("experiment_name"),
        UniqueConstraint("dataset_id", "source_record_id", "adsorbent_id", "temperature_k"),
        Index("ix_adsorption_isotherms_dataset_id", "dataset_id"),
        Index("ix_adsorption_isotherms_adsorbent_id", "adsorbent_id"),
    )


###############################################################################
class AdsorptionIsothermComponent(Base):
    __tablename__ = "adsorption_isotherm_components"
    id = Column(Integer, primary_key=True, autoincrement=True)
    isotherm_id = Column(Integer, ForeignKey("adsorption_isotherms.id", ondelete="CASCADE"), nullable=False)
    component_index = Column(Integer, nullable=False)
    adsorbate_id = Column(Integer, ForeignKey("adsorbates.id", ondelete="RESTRICT"), nullable=False)
    mole_fraction = Column(Float)
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    isotherm_id = Column(Integer, ForeignKey("adsorption_isotherms.id", ondelete="CASCADE"), nullable=False)
    point_index = Column(Integer, nullable=False)
    __table_args__ = (
        UniqueConstraint("isotherm_id", "point_index"),
        Index("ix_adsorption_points_isotherm_id", "isotherm_id"),
    )


###############################################################################
class AdsorptionPointComponent(Base):
    __tablename__ = "adsorption_point_components"
    id = Column(Integer, primary_key=True, autoincrement=True)
    point_id = Column(Integer, ForeignKey("adsorption_points.id", ondelete="CASCADE"), nullable=False)
    component_id = Column(Integer, ForeignKey("adsorption_isotherm_components.id", ondelete="CASCADE"), nullable=False)
    partial_pressure_pa = Column(Float, nullable=False)
    uptake_mol_g = Column(Float, nullable=False)
    original_pressure = Column(Float)
    original_uptake = Column(Float)
    __table_args__ = (
        UniqueConstraint("point_id", "component_id"),
        Index("ix_adsorption_point_components_point_id", "point_id"),
        Index("ix_adsorption_point_components_component_id", "component_id"),
    )


###############################################################################
class AdsorptionProcessedIsotherm(Base):
    __tablename__ = "adsorption_processed_isotherms"
    id = Column(Integer, primary_key=True, autoincrement=True)
    isotherm_id = Column(Integer, ForeignKey("adsorption_isotherms.id", ondelete="CASCADE"), nullable=False)
    processing_version = Column(String, nullable=False, default="v1")
    processed_key = Column(String, nullable=False)
    pressure_pa_series = Column(JSONSequence, nullable=False)
    uptake_mol_g_series = Column(JSONSequence, nullable=False)
    original_pressure_series = Column(JSONSequence)
    original_uptake_series = Column(JSONSequence)
    measurement_count = Column(Integer)
    min_pressure = Column(Float)
    max_pressure = Column(Float)
    min_uptake = Column(Float)
    max_uptake = Column(Float)
    __table_args__ = (
        UniqueConstraint("processed_key"),
        UniqueConstraint("isotherm_id", "processing_version"),
        Index("ix_adsorption_processed_isotherms_isotherm_id", "isotherm_id"),
    )


###############################################################################
class AdsorptionFit(Base):
    __tablename__ = "adsorption_fits"
    id = Column(Integer, primary_key=True, autoincrement=True)
    processed_id = Column(Integer, ForeignKey("adsorption_processed_isotherms.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String, nullable=False)
    optimization_method = Column(String, nullable=False)
    score = Column(Float)
    aic = Column(Float)
    aicc = Column(Float)
    created_at = Column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("processed_id", "model_name", "optimization_method"),
        Index("ix_adsorption_fits_processed_id", "processed_id"),
        Index("ix_adsorption_fits_model_name", "model_name"),
    )


###############################################################################
class AdsorptionFitParam(Base):
    __tablename__ = "adsorption_fit_params"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fit_id = Column(Integer, ForeignKey("adsorption_fits.id", ondelete="CASCADE"), nullable=False)
    param_name = Column(String, nullable=False)
    param_value = Column(Float, nullable=False)
    param_error = Column(Float)
    __table_args__ = (
        UniqueConstraint("fit_id", "param_name"),
        Index("ix_adsorption_fit_params_fit_id", "fit_id"),
    )


###############################################################################
class AdsorptionBestFit(Base):
    __tablename__ = "adsorption_best_fit"
    id = Column(Integer, primary_key=True, autoincrement=True)
    processed_id = Column(Integer, ForeignKey("adsorption_processed_isotherms.id", ondelete="CASCADE"), nullable=False)
    best_fit_id = Column(Integer, ForeignKey("adsorption_fits.id", ondelete="SET NULL"))
    worst_fit_id = Column(Integer, ForeignKey("adsorption_fits.id", ondelete="SET NULL"))
    best_model = Column(String)
    worst_model = Column(String)
    __table_args__ = (
        UniqueConstraint("processed_id"),
        Index("ix_adsorption_best_fit_best_fit_id", "best_fit_id"),
        Index("ix_adsorption_best_fit_worst_fit_id", "worst_fit_id"),
    )


###############################################################################
class TrainingMetadata(Base):
    __tablename__ = "training_metadata"
    hashcode = Column(String, primary_key=True)
    dataset_label = Column(String, nullable=False, default="default")
    created_at = Column(String)
    sample_size = Column(Float)
    validation_size = Column(Float)
    min_measurements = Column(Integer)
    max_measurements = Column(Integer)
    smile_sequence_size = Column(Integer)
    max_pressure = Column(Float)
    max_uptake = Column(Float)
    total_samples = Column(Integer)
    train_samples = Column(Integer)
    validation_samples = Column(Integer)
    smile_vocabulary = Column(JSONSequence)
    adsorbent_vocabulary = Column(JSONSequence)
    normalization_stats = Column(JSONSequence)
    __table_args__ = (
        UniqueConstraint("hashcode"),
        Index("ix_training_metadata_dataset_label", "dataset_label"),
    )


###############################################################################
class TrainingDataset(Base):
    __tablename__ = "training_dataset"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, default="default")
    source_dataset = Column(String)
    split = Column(String)
    temperature = Column(Float)
    pressure = Column(JSONSequence)
    adsorbed_amount = Column(JSONSequence)
    encoded_adsorbent = Column(Integer)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_encoded_smile = Column(JSONSequence)
    training_hashcode = Column(String, ForeignKey("training_metadata.hashcode", ondelete="SET NULL"))
    sample_key = Column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("sample_key"),
        Index("ix_training_dataset_name", "name"),
        Index("ix_training_dataset_training_hashcode", "training_hashcode"),
    )
