# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.domain.public_data import EntityReference, Pagination

DatasetSource = Literal["uploaded", "nist"]
PressureBasis = Literal["absolute", "partial", "relative"]
DistributionVariable = Literal[
    "temperature_k", "pressure", "uptake_mol_kg", "points_per_isotherm"
]
HistogramScale = Literal["linear", "log"]
UptakeMethod = Literal["linear", "nearest"]
ValueMethod = Literal["measured", "nearest", "interpolated"]
ValueStatus = Literal["ok", "out_of_range", "no_point_in_tolerance"]

MAX_FILTER_IDS = 500
MAX_SERIES = 200
MAX_SERIES_POINTS = 50_000

###############################################################################
class AnalyticsFilter(BaseModel):
    """Isotherm selection shared by every analytics query; identity is by id."""

    model_config = ConfigDict(extra="forbid")

    dataset_sources: list[DatasetSource] = Field(default_factory=list)
    dataset_ids: list[int] = Field(default_factory=list, max_length=MAX_FILTER_IDS)
    adsorbent_ids: list[int] = Field(default_factory=list, max_length=MAX_FILTER_IDS)
    adsorbate_ids: list[int] = Field(default_factory=list, max_length=MAX_FILTER_IDS)
    isotherm_ids: list[int] = Field(default_factory=list, max_length=MAX_SERIES)
    temperature_min_k: float | None = Field(default=None, gt=0)
    temperature_max_k: float | None = Field(default=None, gt=0)
    pressure_basis: PressureBasis | None = None
    include_mixtures: bool = False

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_temperature_range(self) -> AnalyticsFilter:
        if (
            self.temperature_min_k is not None
            and self.temperature_max_k is not None
            and self.temperature_min_k > self.temperature_max_k
        ):
            raise ValueError("temperature_min_k must not exceed temperature_max_k.")
        return self

###############################################################################
class FacetCount(BaseModel):
    id: int
    name: str
    count: int = Field(ge=0)

###############################################################################
class SelectionFacets(BaseModel):
    materials: list[FacetCount]
    adsorbates: list[FacetCount]
    datasets: list[FacetCount]
    sources: dict[str, int]
    pressure_bases: dict[str, int]
    temperature_min_k: float | None = None
    temperature_max_k: float | None = None

###############################################################################
class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter: AnalyticsFilter = Field(default_factory=AnalyticsFilter)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
    material_search: str | None = Field(default=None, max_length=120)
    adsorbate_search: str | None = Field(default=None, max_length=120)

###############################################################################
class IsothermSummaryView(BaseModel):
    id: int
    name: str
    dataset_id: int
    dataset: str
    dataset_source: DatasetSource
    material_id: int
    material: str
    adsorbates: list[EntityReference]
    temperature_k: float
    pressure_basis: PressureBasis
    point_count: int = Field(ge=0)

###############################################################################
class SelectionResponse(BaseModel):
    items: list[IsothermSummaryView]
    pagination: Pagination
    facets: SelectionFacets

###############################################################################
class SeriesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter: AnalyticsFilter = Field(default_factory=AnalyticsFilter)
    pressure_unit: str = Field(default="bar", max_length=16)
    uptake_unit: str = Field(default="mol/kg", max_length=24)

###############################################################################
class SeriesMetadata(BaseModel):
    isotherm_id: int
    component_id: int
    isotherm: str
    dataset_id: int
    dataset: str
    dataset_source: DatasetSource
    material_id: int
    material: str
    adsorbate_id: int
    adsorbate: str
    mole_fraction: float | None = None
    temperature_k: float
    uptake_unit: str

###############################################################################
class SeriesPoint(BaseModel):
    observation_id: int
    pressure: float
    uptake: float
    uptake_stddev: float | None = None

###############################################################################
class IsothermSeries(SeriesMetadata):
    points: list[SeriesPoint]

###############################################################################
class SeriesResponse(BaseModel):
    pressure_basis: PressureBasis | None = None
    pressure_unit: str
    series: list[IsothermSeries]
    total_series: int = Field(ge=0)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)

###############################################################################
class UptakeAtPressureRequest(SeriesRequest):
    pressure: float = Field(gt=0)
    method: UptakeMethod = "linear"
    tolerance: float = Field(default=0.05, gt=0, le=1)

###############################################################################
class UptakeAtPressureRow(SeriesMetadata):
    value: float | None = None
    method: ValueMethod | None = None
    status: ValueStatus
    sources: list[SeriesPoint] = Field(default_factory=list)

###############################################################################
class UptakeAtPressureResponse(BaseModel):
    pressure: float
    pressure_unit: str
    pressure_basis: PressureBasis | None = None
    method: UptakeMethod
    rows: list[UptakeAtPressureRow]
    total_series: int = Field(ge=0)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)

###############################################################################
class DistributionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter: AnalyticsFilter = Field(default_factory=AnalyticsFilter)
    variable: DistributionVariable
    bins: int = Field(default=20, ge=1, le=100)
    scale: HistogramScale = "linear"
    pressure_unit: str = Field(default="bar", max_length=16)

###############################################################################
class HistogramBin(BaseModel):
    lower: float
    upper: float
    count: int = Field(ge=0)

###############################################################################
class DistributionStats(BaseModel):
    count: int = Field(ge=0)
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None

###############################################################################
class DistributionResponse(BaseModel):
    variable: DistributionVariable
    unit: str
    scale: HistogramScale
    bins: list[HistogramBin]
    stats: DistributionStats
    excluded: int = Field(default=0, ge=0)

###############################################################################
class BasisSpan(BaseModel):
    pressure_basis: PressureBasis
    pressure_unit: Literal["Pa", "1"]
    isotherms: int = Field(ge=0)
    observations: int = Field(ge=0)
    pressure_min: float | None = None
    pressure_max: float | None = None
    uptake_min_mol_kg: float | None = None
    uptake_max_mol_kg: float | None = None
    temperature_min_k: float | None = None
    temperature_max_k: float | None = None

###############################################################################
class AnalyticsOverviewResponse(BaseModel):
    datasets: int = Field(ge=0)
    isotherms: int = Field(ge=0)
    observations: int = Field(ge=0)
    materials: int = Field(ge=0)
    adsorbates: int = Field(ge=0)
    mixture_isotherms: int = Field(ge=0)
    isotherms_by_source: dict[str, int]
    pressure_bases: list[BasisSpan]
    top_materials: list[FacetCount]
    top_adsorbates: list[FacetCount]
    temperature_histogram: list[HistogramBin]
