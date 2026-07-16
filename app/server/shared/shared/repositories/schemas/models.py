from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import CheckConstraint, Column, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, MetaData, String, Table, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, foreign, mapped_column, relationship, synonym

from shared.repositories.schemas.types import JSONList, JSONMapping, UTCDateTime, normalize_identity


###############################################################################
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


###############################################################################
class Base(DeclarativeBase):
    pass


###############################################################################
class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column("dataset_name", String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(255), nullable=False,
        default=lambda context: normalize_identity(
            context.get_current_parameters().get("dataset_name", "")
        ),
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    tags: Mapped[list[Any]] = mapped_column(JSONList, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False)
    isotherms: Mapped[list[Isotherm]] = relationship(back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_datasets_normalized_name"), CheckConstraint("source IN ('uploaded', 'nist')", name="ck_datasets_source"), CheckConstraint("length(normalized_name) > 0", name="ck_datasets_name"), Index("ix_datasets_source_created_at", "source", "created_at"))

    dataset_name = synonym("name")

    # -------------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        if "name" not in kwargs and "dataset_name" in kwargs:
            kwargs["name"] = kwargs.pop("dataset_name")
        if "normalized_name" not in kwargs and "name" in kwargs:
            kwargs["normalized_name"] = normalize_identity(kwargs["name"])
        super().__init__(**kwargs)


###############################################################################
class Adsorbate(Base):
    __tablename__ = "adsorbates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    adsorbate_key: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(255), nullable=False,
        default=lambda context: normalize_identity(
            context.get_current_parameters().get("name", "")
        ),
    )
    inchi_key: Mapped[str | None] = mapped_column("InChIKey", String(27), key="InChIKey", unique=True)
    inchi: Mapped[str | None] = mapped_column(String)
    formula: Mapped[str | None] = mapped_column(String(255))
    molar_mass_g_mol: Mapped[float | None] = mapped_column(Float)
    smiles: Mapped[str | None] = mapped_column(String)
    molecular_formula: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False)
    components: Mapped[list[IsothermComponent]] = relationship(back_populates="adsorbate", lazy="raise")
    __table_args__ = (UniqueConstraint("adsorbate_key", name="uq_adsorbates_key"), Index("ix_adsorbates_normalized_name", "normalized_name"))

    def __init__(self, **kwargs: Any) -> None:
        if "normalized_name" not in kwargs and "name" in kwargs:
            kwargs["normalized_name"] = normalize_identity(kwargs["name"])
        super().__init__(**kwargs)


###############################################################################
class Adsorbent(Base):
    __tablename__ = "adsorbents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    adsorbent_key: Mapped[str] = mapped_column(String(255), nullable=False)
    external_hash: Mapped[str | None] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(255), nullable=False,
        default=lambda context: normalize_identity(
            context.get_current_parameters().get("name", "")
        ),
    )
    formula: Mapped[str | None] = mapped_column(String(255))
    molar_mass_g_mol: Mapped[float | None] = mapped_column(Float)
    smiles: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False)
    isotherms: Mapped[list[Isotherm]] = relationship(back_populates="adsorbent", lazy="raise")
    __table_args__ = (UniqueConstraint("adsorbent_key", name="uq_adsorbents_key"), Index("ix_adsorbents_normalized_name", "normalized_name"))


###############################################################################
class Isotherm(Base):
    __tablename__ = "isotherms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    isotherm_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    source_record_key: Mapped[str] = mapped_column(String(255), nullable=False)
    source_record_label: Mapped[str | None] = mapped_column(String(255))
    adsorbent_id: Mapped[int] = mapped_column(ForeignKey("adsorbents.id", ondelete="RESTRICT"), nullable=False)
    temperature_k: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_unit: Mapped[str | None] = mapped_column(String(32))
    uptake_unit: Mapped[str | None] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    tags: Mapped[list[Any]] = mapped_column(JSONList, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False)
    dataset: Mapped[Dataset] = relationship(back_populates="isotherms", lazy="raise")
    adsorbent: Mapped[Adsorbent] = relationship(back_populates="isotherms", lazy="raise")
    components: Mapped[list[IsothermComponent]] = relationship(back_populates="isotherm", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    measurements: Mapped[list[IsothermMeasurement]] = relationship(back_populates="isotherm", primaryjoin=lambda: Isotherm.id == foreign(IsothermMeasurement.isotherm_id), cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    processed: Mapped[list[ProcessedIsotherm]] = relationship(back_populates="isotherm", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    __table_args__ = (Index("ix_isotherms_dataset_source", "dataset_id", "source_record_key"), Index("ix_isotherms_adsorbent_temperature", "adsorbent_id", "temperature_k"), CheckConstraint("temperature_k > 0", name="ck_isotherms_temperature"))

    source_record_id = synonym("source_record_key")
    experiment_name = synonym("source_record_key")
    pressure_units = synonym("pressure_unit")
    adsorption_units = synonym("uptake_unit")

    def __init__(self, **kwargs: Any) -> None:
        if "source_record_key" not in kwargs:
            kwargs["source_record_key"] = kwargs.get("source_record_id") or kwargs.get("experiment_name") or "record"
        if "isotherm_key" not in kwargs:
            kwargs["isotherm_key"] = normalize_identity(kwargs["source_record_key"])
        super().__init__(**kwargs)


###############################################################################
class IsothermComponent(Base):
    __tablename__ = "isotherm_components"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    isotherm_id: Mapped[int] = mapped_column(ForeignKey("isotherms.id", ondelete="CASCADE"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    adsorbate_id: Mapped[int] = mapped_column(ForeignKey("adsorbates.id", ondelete="RESTRICT"), nullable=False)
    mole_fraction: Mapped[float | None] = mapped_column(Float)
    isotherm: Mapped[Isotherm] = relationship(back_populates="components", lazy="raise")
    adsorbate: Mapped[Adsorbate] = relationship(back_populates="components", lazy="raise")
    measurements: Mapped[list[IsothermMeasurement]] = relationship(back_populates="component", overlaps="isotherm,measurements", lazy="raise")
    __table_args__ = (UniqueConstraint("isotherm_id", "position", name="uq_components_position"), UniqueConstraint("isotherm_id", "adsorbate_id", name="uq_components_adsorbate"), UniqueConstraint("id", "isotherm_id", name="uq_components_ownership"), CheckConstraint("position >= 1", name="ck_components_position"), CheckConstraint("mole_fraction IS NULL OR (mole_fraction >= 0 AND mole_fraction <= 1)", name="ck_components_fraction"))

    component_index = synonym("position")


###############################################################################
class IsothermMeasurement(Base):
    __tablename__ = "isotherm_measurements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    isotherm_id: Mapped[int | None] = mapped_column(Integer)
    point_id_value: Mapped[int | None] = mapped_column(
        "point_id", Integer, ForeignKey("isotherm_measurements.id", ondelete="CASCADE")
    )
    component_id: Mapped[int | None] = mapped_column(Integer)
    point_index: Mapped[int | None] = mapped_column(Integer)
    partial_pressure_pa: Mapped[float | None] = mapped_column(Float)
    uptake_mol_g: Mapped[float | None] = mapped_column(Float)
    original_pressure: Mapped[float | None] = mapped_column(Float)
    original_uptake: Mapped[float | None] = mapped_column(Float)
    isotherm: Mapped[Isotherm] = relationship(back_populates="measurements", primaryjoin=lambda: Isotherm.id == foreign(IsothermMeasurement.isotherm_id), overlaps="component,measurements", lazy="raise")
    component: Mapped[IsothermComponent] = relationship(back_populates="measurements", overlaps="isotherm,measurements", lazy="raise")
    __table_args__ = (ForeignKeyConstraint(["component_id", "isotherm_id"], ["isotherm_components.id", "isotherm_components.isotherm_id"], ondelete="CASCADE"), UniqueConstraint("point_id", "component_id", name="uq_measurements_identity"), Index("ix_measurements_isotherm_point", "isotherm_id", "point_index"), Index("ix_measurements_component", "component_id"), CheckConstraint("partial_pressure_pa >= 0", name="ck_measurements_pressure"))

    point_id = synonym("point_id_value")


###############################################################################
class ProcessedIsotherm(Base):
    __tablename__ = "processed_isotherms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    isotherm_id: Mapped[int] = mapped_column(ForeignKey("isotherms.id", ondelete="CASCADE"), nullable=False)
    processing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    pressure_pa_values: Mapped[list[Any]] = mapped_column(JSONList, nullable=False)
    uptake_mol_g_values: Mapped[list[Any]] = mapped_column(JSONList, nullable=False)
    measurement_count: Mapped[int] = mapped_column(Integer, nullable=False)
    min_pressure_pa: Mapped[float | None] = mapped_column(Float)
    max_pressure_pa: Mapped[float | None] = mapped_column(Float)
    min_uptake_mol_g: Mapped[float | None] = mapped_column(Float)
    max_uptake_mol_g: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False)
    isotherm: Mapped[Isotherm] = relationship(back_populates="processed", lazy="raise")
    fits: Mapped[list[Fit]] = relationship(back_populates="processed_isotherm", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    __table_args__ = (UniqueConstraint("isotherm_id", "processing_version", name="uq_processed_identity"),)


###############################################################################
class Fit(Base):
    __tablename__ = "fits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processed_isotherm_id: Mapped[int] = mapped_column("processed_id", ForeignKey("processed_isotherms.id", ondelete="CASCADE"), nullable=False, )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    optimization_method: Mapped[str] = mapped_column(String(128), nullable=False)
    objective_score: Mapped[float | None] = mapped_column(Float)
    aic: Mapped[float | None] = mapped_column(Float)
    aicc: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    processed_isotherm: Mapped[ProcessedIsotherm] = relationship(back_populates="fits", lazy="raise")
    parameters: Mapped[list[FitParameter]] = relationship(back_populates="fit", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    __table_args__ = (UniqueConstraint("processed_id", "model_name", "optimization_method", name="uq_fits_identity"), Index("ix_fits_processed_aicc", "processed_id", "aicc"))


###############################################################################
class FitParameter(Base):
    __tablename__ = "fit_parameters"
    fit_id: Mapped[int] = mapped_column(ForeignKey("fits.id", ondelete="CASCADE"), primary_key=True)
    parameter_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    parameter_value: Mapped[float] = mapped_column(Float, nullable=False)
    standard_error: Mapped[float | None] = mapped_column(Float)
    fit: Mapped[Fit] = relationship(back_populates="parameters", lazy="raise")


###############################################################################
class TrainingDataset(Base):
    __tablename__ = "training_datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONMapping, nullable=False, default=dict)
    sample_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    validation_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    min_measurements: Mapped[int | None] = mapped_column(Integer)
    max_measurements: Mapped[int | None] = mapped_column(Integer)
    total_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vocabularies: Mapped[dict[str, Any]] = mapped_column(JSONMapping, nullable=False, default=dict)
    normalization_stats: Mapped[dict[str, Any]] = mapped_column(JSONMapping, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    samples: Mapped[list[TrainingSample]] = relationship(back_populates="training_dataset", cascade="all, delete-orphan", passive_deletes=True, lazy="raise")
    __table_args__ = (CheckConstraint("sample_fraction >= 0 AND sample_fraction <= 1", name="ck_training_sample_fraction"), CheckConstraint("validation_fraction >= 0 AND validation_fraction <= 1", name="ck_training_validation_fraction"), CheckConstraint("min_measurements IS NULL OR max_measurements IS NULL OR min_measurements <= max_measurements", name="ck_training_measurement_bounds"))


###############################################################################
class TrainingSample(Base):
    __tablename__ = "training_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_dataset_id: Mapped[int] = mapped_column(ForeignKey("training_datasets.id", ondelete="CASCADE"), nullable=False)
    sample_key: Mapped[str] = mapped_column(String(255), nullable=False)
    source_isotherm_id: Mapped[int | None] = mapped_column(ForeignKey("isotherms.id", ondelete="SET NULL"))
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    temperature_k: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_values: Mapped[list[Any]] = mapped_column(JSONList, nullable=False)
    uptake_values: Mapped[list[Any]] = mapped_column(JSONList, nullable=False)
    encoded_adsorbent: Mapped[int | None] = mapped_column(Integer)
    adsorbate_molar_mass: Mapped[float | None] = mapped_column(Float)
    encoded_smiles: Mapped[list[Any] | None] = mapped_column(JSONList)
    training_dataset: Mapped[TrainingDataset] = relationship(back_populates="samples", lazy="raise")
    __table_args__ = (UniqueConstraint("training_dataset_id", "sample_key", name="uq_training_samples_key"), CheckConstraint("split IN ('train', 'validation', 'test')", name="ck_training_samples_split"), Index("ix_training_samples_dataset_split", "training_dataset_id", "split"))


# Import-only aliases let the consumer migration land in separate, reviewable slices.
AdsorptionIsotherm = Isotherm
AdsorptionIsothermComponent = IsothermComponent
AdsorptionPoint = IsothermMeasurement
_compatibility_metadata = MetaData()
_compatibility_point_component_table = Table(
    "isotherm_measurements",
    _compatibility_metadata,
    Column("point_id", Integer, primary_key=True),
    Column("component_id", Integer, primary_key=True),
    Column("partial_pressure_pa", Float, nullable=False),
    Column("uptake_mol_g", Float, nullable=False),
    Column("original_pressure", Float),
    Column("original_uptake", Float),
)


class _CompatibilityBase(DeclarativeBase):
    pass


class AdsorptionPointComponent(_CompatibilityBase):
    __table__ = _compatibility_point_component_table
AdsorptionProcessedIsotherm = ProcessedIsotherm
AdsorptionFit = Fit
AdsorptionFitParam = FitParameter
AdsorptionBestFit = Fit
TrainingMetadata = TrainingDataset

__all__ = ["Base", "Dataset", "Adsorbate", "Adsorbent", "Isotherm", "IsothermComponent", "IsothermMeasurement", "ProcessedIsotherm", "Fit", "FitParameter", "TrainingDataset", "TrainingSample", "AdsorptionIsotherm", "AdsorptionIsothermComponent", "AdsorptionPoint", "AdsorptionPointComponent", "AdsorptionProcessedIsotherm", "AdsorptionFit", "AdsorptionFitParam", "AdsorptionBestFit", "TrainingMetadata"]
