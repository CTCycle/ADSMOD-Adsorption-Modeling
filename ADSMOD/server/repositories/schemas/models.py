from __future__ import annotations

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

from ADSMOD.server.common.constants import (
    COLUMN_AIC,
    COLUMN_AICC,
    COLUMN_BEST_MODEL,
    COLUMN_EXPERIMENT,
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
from ADSMOD.server.repositories.schemas.types import JSONSequence

Base = declarative_base()


###############################################################################
class AdsorptionData(Base):
    __tablename__ = "adsorption_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    experiment = Column(COLUMN_EXPERIMENT, String)
    temperature_K = Column(COLUMN_TEMPERATURE_K, Integer)
    pressure_Pa = Column(COLUMN_PRESSURE_PA, Float)
    uptake_mol_g = Column(COLUMN_UPTAKE_MOL_G, Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionProcessedData(Base):
    __tablename__ = "adsorption_processed_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    temperature_K = Column(COLUMN_TEMPERATURE_K, Integer)
    pressure_Pa = Column(COLUMN_PRESSURE_PA, JSONSequence)
    uptake_mol_g = Column(COLUMN_UPTAKE_MOL_G, JSONSequence)
    measurement_count = Column(COLUMN_MEASUREMENT_COUNT, Integer)
    min_pressure = Column(COLUMN_MIN_PRESSURE, Float)
    max_pressure = Column(COLUMN_MAX_PRESSURE, Float)
    min_uptake = Column(COLUMN_MIN_UPTAKE, Float)
    max_uptake = Column(COLUMN_MAX_UPTAKE, Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionLangmuirResults(Base):
    __tablename__ = "adsorption_langmuir"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionSipsResults(Base):
    __tablename__ = "adsorption_sips"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
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
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionFreundlichResults(Base):
    __tablename__ = "adsorption_freundlich"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    exponent = Column("exponent", Float)
    exponent_error = Column("exponent error", Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionTemkinResults(Base):
    __tablename__ = "adsorption_temkin"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionTothResults(Base):
    __tablename__ = "adsorption_toth"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
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
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionDubininRadushkevichResults(Base):
    __tablename__ = "adsorption_dubinin_radushkevich"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    beta = Column("beta", Float)
    beta_error = Column("beta error", Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionDualSiteLangmuirResults(Base):
    __tablename__ = "adsorption_dual_site_langmuir"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
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
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionRedlichPetersonResults(Base):
    __tablename__ = "adsorption_redlich_peterson"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
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
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionJovanovicResults(Base):
    __tablename__ = "adsorption_jovanovic"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    optimization_method = Column(COLUMN_OPTIMIZATION_METHOD, String)
    score = Column(COLUMN_SCORE, Float)
    aic = Column(COLUMN_AIC, Float)
    aicc = Column(COLUMN_AICC, Float)
    k = Column("k", Float)
    k_error = Column("k error", Float)
    qsat = Column("qsat", Float)
    qsat_error = Column("qsat error", Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class AdsorptionBestFit(Base):
    __tablename__ = "adsorption_best_fit"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    best_model = Column(COLUMN_BEST_MODEL, String)
    worst_model = Column(COLUMN_WORST_MODEL, String)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class NistSingleComponentAdsorption(Base):
    __tablename__ = "nist_single_component_adsorption"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    temperature = Column(Float)
    adsorption_units = Column("adsorption_units", String)
    pressure_units = Column("pressure_units", String)
    adsorbent_name = Column(String)
    adsorbate_name = Column(String)
    pressure = Column(Float)
    adsorbed_amount = Column(Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class NistBinaryMixtureAdsorption(Base):
    __tablename__ = "nist_binary_mixture_adsorption"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    temperature = Column(Float)
    adsorption_units = Column("adsorption_units", String)
    pressure_units = Column("pressure_units", String)
    adsorbent_name = Column(String)
    compound_1 = Column(String)
    compound_2 = Column(String)
    compound_1_composition = Column(Float)
    compound_2_composition = Column(Float)
    compound_1_pressure = Column(Float)
    compound_2_pressure = Column(Float)
    compound_1_adsorption = Column(Float)
    compound_2_adsorption = Column(Float)
    __table_args__ = (UniqueConstraint("id", "name"),)


###############################################################################
class Adsorbate(Base):
    __tablename__ = "adsorbates"
    InChIKey = Column(String, primary_key=True)
    name = Column(String)
    InChICode = Column(String)
    formula = Column(String)
    molecular_weight = Column(Float)
    molecular_formula = Column(String)
    smile_code = Column(String)
    __table_args__ = (UniqueConstraint("InChIKey"),)


###############################################################################
class Adsorbent(Base):
    __tablename__ = "adsorbents"
    name = Column(String)
    hashkey = Column(String, primary_key=True)
    formula = Column(String)
    molecular_weight = Column(Float)
    molecular_formula = Column(String)
    smile_code = Column(String)
    __table_args__ = (UniqueConstraint("hashkey"),)


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
    __table_args__ = (UniqueConstraint("id"),)


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
    __table_args__ = (UniqueConstraint("hashcode"),)
