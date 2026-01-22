from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


###############################################################################
class AdsorptionData(Base):
    __tablename__ = "ADSORPTION_DATA"
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String)
    experiment = Column(String)
    temperature_K = Column("temperature [K]", Integer)
    pressure_Pa = Column("pressure [Pa]", Float)
    uptake_mol_g = Column("uptake [mol/g]", Float)
    __table_args__ = (
        UniqueConstraint(
            "dataset_name",
            "experiment",
            "temperature [K]",
            "pressure [Pa]",
        ),
    )


###############################################################################
class AdsorptionProcessedData(Base):
    __tablename__ = "ADSORPTION_PROCESSED_DATA"
    id = Column(Integer, primary_key=True)
    experiment = Column(String)
    temperature_K = Column("temperature [K]", Integer)
    pressure_Pa = Column("pressure [Pa]", String)
    uptake_mol_g = Column("uptake [mol/g]", String)
    measurement_count = Column(Integer)
    min_pressure = Column(Float)
    max_pressure = Column(Float)
    min_uptake = Column(Float)
    max_uptake = Column(Float)
    __table_args__ = (UniqueConstraint("id"),)


###############################################################################
class AdsorptionLangmuirResults(Base):
    __tablename__ = "ADSORPTION_LANGMUIR"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionSipsResults(Base):
    __tablename__ = "ADSORPTION_SIPS"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionFreundlichResults(Base):
    __tablename__ = "ADSORPTION_FREUNDLICH"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionTemkinResults(Base):
    __tablename__ = "ADSORPTION_TEMKIN"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionTothResults(Base):
    __tablename__ = "ADSORPTION_TOTH"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionDubininRadushkevichResults(Base):
    __tablename__ = "ADSORPTION_DUBININ_RADUSHKEVICH"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionDualSiteLangmuirResults(Base):
    __tablename__ = "ADSORPTION_DUAL_SITE_LANGMUIR"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k1 = Column("k1", Float)
    k1_error = Column("k1 error", Float)
    qsat1 = Column("qsat1", Float)
    qsat1_error = Column("qsat1 error", Float)
    k2 = Column("k2", Float)
    k2_error = Column("k2 error", Float)
    qsat2 = Column("qsat2", Float)
    qsat2_error = Column("qsat2 error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionRedlichPetersonResults(Base):
    __tablename__ = "ADSORPTION_REDLICH_PETERSON"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    a = Column("a", Float)
    a_error = Column("a error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionJovanovicResults(Base):
    __tablename__ = "ADSORPTION_JOVANOVIC"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    optimization_method = Column("optimization method", String)
    score = Column("score", Float)
    aic = Column("AIC", Float)
    aicc = Column("AICc", Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class AdsorptionBestFit(Base):
    __tablename__ = "ADSORPTION_BEST_FIT"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_PROCESSED_DATA.id"), nullable=False
    )
    best_model = Column("best model", String)
    worst_model = Column("worst model", String)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )


###############################################################################
class NistSingleComponentAdsorption(Base):
    __tablename__ = "NIST_SINGLE_COMPONENT_ADSORPTION"
    filename = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    adsorptionUnits = Column(String)
    pressureUnits = Column(String)
    adsorbent_name = Column(String, primary_key=True)
    adsorbate_name = Column(String, primary_key=True)
    pressure = Column(Float, primary_key=True)
    adsorbed_amount = Column(Float)
    composition = Column(Float)
    __table_args__ = (
        UniqueConstraint(
            "filename",
            "temperature",
            "pressure",
            "adsorbent_name",
            "adsorbate_name",
        ),
    )


###############################################################################
class NistBinaryMixtureAdsorption(Base):
    __tablename__ = "NIST_BINARY_MIXTURE_ADSORPTION"
    filename = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    adsorptionUnits = Column(String)
    pressureUnits = Column(String)
    adsorbent_name = Column(String, primary_key=True)
    compound_1 = Column(String, primary_key=True)
    compound_2 = Column(String, primary_key=True)
    compound_1_composition = Column(Float)
    compound_2_composition = Column(Float)
    compound_1_pressure = Column(Float, primary_key=True)
    compound_2_pressure = Column(Float, primary_key=True)
    compound_1_adsorption = Column(Float)
    compound_2_adsorption = Column(Float)
    __table_args__ = (
        UniqueConstraint(
            "filename",
            "temperature",
            "adsorbent_name",
            "compound_1",
            "compound_2",
            "compound_1_pressure",
            "compound_2_pressure",
        ),
    )


###############################################################################
class Adsorbate(Base):
    __tablename__ = "ADSORBATES"
    InChIKey = Column(String, primary_key=True)
    name = Column(String)
    InChICode = Column(String)
    formula = Column(String)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_molecular_formula = Column(String)
    adsorbate_SMILE = Column(String)
    __table_args__ = (UniqueConstraint("InChIKey"),)


###############################################################################
class Adsorbent(Base):
    __tablename__ = "ADSORBENTS"
    name = Column(String)
    hashkey = Column(String, primary_key=True)
    formula = Column(String)
    adsorbent_molecular_weight = Column(Float)
    adsorbent_molecular_formula = Column(String)
    adsorbent_SMILE = Column(String)
    __table_args__ = (UniqueConstraint("hashkey"),)


###############################################################################
class TrainingDataset(Base):
    __tablename__ = "TRAINING_DATASET"
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String)
    split = Column(String)
    temperature = Column(Float)
    pressure = Column(String)
    adsorbed_amount = Column(String)
    encoded_adsorbent = Column(Integer)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_encoded_SMILE = Column(String)
    __table_args__ = (UniqueConstraint("id"),)


###############################################################################
class TrainingMetadata(Base):
    __tablename__ = "TRAINING_METADATA"
    id = Column(Integer, primary_key=True)
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
    smile_vocabulary = Column(String)
    adsorbent_vocabulary = Column(String)
    normalization_stats = Column(String)
    __table_args__ = (UniqueConstraint("id"),)
