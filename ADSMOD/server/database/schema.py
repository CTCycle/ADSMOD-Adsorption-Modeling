from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

from ADSMOD.server.utils.constants import (
    COLUMN_AIC,
    COLUMN_AICC,
    COLUMN_BEST_MODEL,
    COLUMN_DATASET_NAME,
    COLUMN_EXPERIMENT,
    COLUMN_EXPERIMENT_NAME,
    COLUMN_MAX_PRESSURE,
    COLUMN_MAX_UPTAKE,
    COLUMN_MEASUREMENT_COUNT,
    COLUMN_MIN_PRESSURE,
    COLUMN_MIN_UPTAKE,
    COLUMN_OPTIMIZATION_METHOD,
    COLUMN_PRESSURE_PA,
    COLUMN_SCORE,
    COLUMN_TEMPERATURE_K,
    COLUMN_UPTAKE_MOL_G,
    COLUMN_WORST_MODEL,
)

Base = declarative_base()


###############################################################################
class AdsorptionData(Base):
    __tablename__ = "ADSORPTION_DATA"
    id = Column(Integer, primary_key=True)
    dataset_name = Column(COLUMN_DATASET_NAME, String)
    experiment = Column(COLUMN_EXPERIMENT, String)
    temperature_K = Column(COLUMN_TEMPERATURE_K, Integer)
    pressure_Pa = Column(COLUMN_PRESSURE_PA, Float)
    uptake_mol_g = Column(COLUMN_UPTAKE_MOL_G, Float)
    __table_args__ = (
        UniqueConstraint(
            COLUMN_DATASET_NAME,
            COLUMN_EXPERIMENT,
            COLUMN_TEMPERATURE_K,
            COLUMN_PRESSURE_PA,
        ),
    )


###############################################################################
class AdsorptionProcessedData(Base):
    __tablename__ = "ADSORPTION_PROCESSED_DATA"
    id = Column(Integer, primary_key=True)
    experiment = Column(COLUMN_EXPERIMENT, String)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    temperature_K = Column(COLUMN_TEMPERATURE_K, Integer)
    pressure_Pa = Column(COLUMN_PRESSURE_PA, String)
    uptake_mol_g = Column(COLUMN_UPTAKE_MOL_G, String)
    measurement_count = Column(COLUMN_MEASUREMENT_COUNT, Integer)
    min_pressure = Column(COLUMN_MIN_PRESSURE, Float)
    max_pressure = Column(COLUMN_MAX_PRESSURE, Float)
    min_uptake = Column(COLUMN_MIN_UPTAKE, Float)
    max_uptake = Column(COLUMN_MAX_UPTAKE, Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionLangmuirResults(Base):
    __tablename__ = "ADSORPTION_LANGMUIR"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionSipsResults(Base):
    __tablename__ = "ADSORPTION_SIPS"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionFreundlichResults(Base):
    __tablename__ = "ADSORPTION_FREUNDLICH"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionTemkinResults(Base):
    __tablename__ = "ADSORPTION_TEMKIN"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionTothResults(Base):
    __tablename__ = "ADSORPTION_TOTH"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionDubininRadushkevichResults(Base):
    __tablename__ = "ADSORPTION_DUBININ_RADUSHKEVICH"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionDualSiteLangmuirResults(Base):
    __tablename__ = "ADSORPTION_DUAL_SITE_LANGMUIR"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k1 = Column("k1", Float)
    k1_error = Column("k1 error", Float)
    qsat1 = Column("qsat1", Float)
    qsat1_error = Column("qsat1 error", Float)
    k2 = Column("k2", Float)
    k2_error = Column("k2 error", Float)
    qsat2 = Column("qsat2", Float)
    qsat2_error = Column("qsat2 error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionRedlichPetersonResults(Base):
    __tablename__ = "ADSORPTION_REDLICH_PETERSON"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    a = Column("a", Float)
    a_error = Column("a error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionJovanovicResults(Base):
    __tablename__ = "ADSORPTION_JOVANOVIC"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


###############################################################################
class AdsorptionBestFit(Base):
    __tablename__ = "ADSORPTION_BEST_FIT"
    id = Column(Integer, primary_key=True)
    experiment_name = Column(COLUMN_EXPERIMENT_NAME, String)
    best_model = Column(COLUMN_BEST_MODEL, String)
    worst_model = Column(COLUMN_WORST_MODEL, String)
    __table_args__ = (UniqueConstraint(COLUMN_EXPERIMENT_NAME),)


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
