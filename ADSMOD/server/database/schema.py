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
    experiment = Column(String)
    temperature_K = Column("temperature [K]", BigInteger)
    pressure_Pa = Column("pressure [Pa]", Float)
    uptake_mol_g = Column("uptake [mol/g]", Float)
    __table_args__ = (UniqueConstraint("id"),)


###############################################################################
class AdsorptionProcessedData(Base):
    __tablename__ = "ADSORPTION_PROCESSED_DATA"
    id = Column(Integer, primary_key=True)
    experiment = Column(String)
    temperature_K = Column("temperature [K]", BigInteger)
    pressure_Pa = Column("pressure [Pa]", String)
    uptake_mol_g = Column("uptake [mol/g]", String)
    measurement_count = Column(BigInteger)
    min_pressure = Column(Float)
    max_pressure = Column(Float)
    min_uptake = Column(Float)
    max_uptake = Column(Float)
    __table_args__ = (UniqueConstraint("id"),)


###############################################################################
class AdsorptionExperiment(Base):
    __tablename__ = "ADSORPTION_EXPERIMENT"
    id = Column(Integer, primary_key=True)
    experiment = Column(String)
    temperature_K = Column("temperature [K]", BigInteger)
    pressure_Pa = Column("pressure [Pa]", String)
    uptake_mol_g = Column("uptake [mol/g]", String)
    measurement_count = Column(BigInteger)
    min_pressure = Column(Float)
    max_pressure = Column(Float)
    min_uptake = Column(Float)
    max_uptake = Column(Float)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment"),
    )


###############################################################################
class AdsorptionLangmuirResults(Base):
    __tablename__ = "ADSORPTION_LANGMUIR"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
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
        Integer, ForeignKey("ADSORPTION_EXPERIMENT.id"), nullable=False
    )
    best_model = Column("best model", String)
    worst_model = Column("worst model", String)
    __table_args__ = (
        UniqueConstraint("id"),
        UniqueConstraint("experiment_id"),
    )
