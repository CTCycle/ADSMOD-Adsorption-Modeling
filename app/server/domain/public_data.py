# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


SourceStatus = Literal["available", "degraded", "unavailable", "unknown"]
ProviderCapability = Literal[
    "adsorption",
    "materials",
    "chemicals",
    "structures",
    "references",
]

###############################################################################
class PublicSourceSummary(BaseModel):
    key: str
    name: str
    description: str
    capabilities: list[ProviderCapability]
    homepage_url: str
    license_name: str | None = None
    license_url: str | None = None
    terms_url: str | None = None
    record_count: int = 0
    health_status: SourceStatus = "unknown"
    health_detail: str | None = None
    last_health_checked_at: datetime | None = None
    health_fresh: bool = False
    last_successful_sync_at: datetime | None = None

###############################################################################
class PublicSourceListResponse(BaseModel):
    sources: list[PublicSourceSummary]

###############################################################################
class PublicDataOverviewResponse(BaseModel):
    adsorption_records: int = Field(ge=0)
    materials: int = Field(ge=0)
    chemicals: int = Field(ge=0)
    structures: int = Field(ge=0)
    active_sync_runs: int = Field(default=0, ge=0)
    failed_sync_runs: int = Field(default=0, ge=0)
    latest_successful_sync_at: datetime | None = None

###############################################################################
SyncOperation = Literal[
    "health_check",
    "nist_index",
    "nist_category",
    "pubchem_resolve",
    "pubchem_enrich",
    "cod_import",
]
NISTCategory = Literal["experiments", "guest", "host"]
SyncRunStatus = Literal[
    "queued",
    "running",
    "completed",
    "partially_completed",
    "failed",
    "cancelled",
    "interrupted",
]

###############################################################################
class PublicDataSyncCreateRequest(BaseModel):
    source_key: Literal["nist", "pubchem", "cod"]
    operation: Literal[
        "nist_index", "nist_category", "pubchem_resolve", "pubchem_enrich", "cod_import"
    ]
    category: NISTCategory | None = None
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    query: str | None = Field(default=None, min_length=1, max_length=512)
    adsorbate_id: int | None = Field(default=None, ge=1)
    cod_id: str | None = Field(default=None, pattern=r"^\d{4,12}$")
    cod_ids: list[str] | None = Field(default=None, min_length=1, max_length=50)
    adsorbent_id: int | None = Field(default=None, ge=1)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_operation_parameters(self) -> "PublicDataSyncCreateRequest":
        required = {
            "nist_index": ("nist", self.category is not None),
            "nist_category": ("nist", self.category is not None),
            "pubchem_resolve": ("pubchem", bool(self.query and self.query.strip())),
            "pubchem_enrich": ("pubchem", True),
            "cod_import": (
                "cod",
                (self.cod_id is not None) != (self.cod_ids is not None),
            ),
        }
        source, has_value = required[self.operation]
        if source != self.source_key or not has_value:
            raise ValueError(
                f"{self.operation} requires source_key={source!r} and its matching parameters."
            )
        if self.query is not None:
            self.query = self.query.strip()
        if self.cod_ids is not None:
            if len(set(self.cod_ids)) != len(self.cod_ids):
                raise ValueError("cod_ids must not contain duplicate identifiers.")
            if any(
                not cod_id.isdigit() or not 4 <= len(cod_id) <= 12
                for cod_id in self.cod_ids
            ):
                raise ValueError("Each COD identifier must contain 4 to 12 digits.")
        return self

    # -------------------------------------------------------------------------
    def parameters(self) -> dict[str, object]:
        """Return only the parameters the selected operation uses."""
        fields = {
            "nist_index": ("category",),
            "nist_category": ("category", "fraction"),
            "pubchem_resolve": ("query", "adsorbate_id"),
            "pubchem_enrich": (),
            "cod_import": ("cod_id", "cod_ids", "adsorbent_id"),
        }[self.operation]
        values = {name: getattr(self, name) for name in fields}
        if self.operation == "nist_category" and values["fraction"] is None:
            values["fraction"] = 1.0
        return {key: value for key, value in values.items() if value is not None}

###############################################################################
class PublicDataSyncRunView(BaseModel):
    id: str
    source_key: str
    operation: SyncOperation
    parameters: dict[str, object]
    status: SyncRunStatus
    progress: float = Field(ge=0.0, le=100.0)
    processed_count: int = Field(ge=0)
    imported_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    checkpoint: dict[str, object]
    cancel_requested: bool
    error_detail: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

###############################################################################
class PublicDataSyncRunListResponse(BaseModel):
    runs: list[PublicDataSyncRunView]
    total: int = Field(ge=0)

###############################################################################
class PublicDataSyncFailureView(BaseModel):
    id: int
    source_key: str
    external_id: str | None = None
    reason: str
    retry_eligible_at: datetime | None = None
    created_at: datetime

###############################################################################
class PublicDataSyncFailureListResponse(BaseModel):
    failures: list[PublicDataSyncFailureView]
    total: int = Field(ge=0)

###############################################################################
class Pagination(BaseModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)

###############################################################################
class ExternalIdentifierView(BaseModel):
    source: str
    external_id: str
    source_url: str | None = None
    retrieved_at: datetime | None = None
    source_version: str | None = None

###############################################################################
class EntityReference(BaseModel):
    id: int
    name: str

###############################################################################
class AdsorptionRecordView(BaseModel):
    id: int
    external_id: str
    source: str
    source_url: str | None = None
    material_id: int
    material: str
    adsorbates: list[EntityReference]
    temperature_k: float
    pressure_min_pa: float | None = None
    pressure_max_pa: float | None = None
    uptake_min_mol_kg: float | None = None
    uptake_max_mol_kg: float | None = None
    point_count: int = 0
    doi: str | None = None
    reference_title: str | None = None
    retrieved_at: datetime | None = None

###############################################################################
class AdsorptionPageResponse(BaseModel):
    items: list[AdsorptionRecordView]
    pagination: Pagination

###############################################################################
class MeasurementView(BaseModel):
    sequence_index: int
    adsorbate: str
    pressure_original: float
    pressure_original_unit: str
    pressure_pa: float
    uptake_original: float
    uptake_original_unit: str
    uptake_mol_kg: float

###############################################################################
class AdsorptionDetailResponse(AdsorptionRecordView):
    pressure_basis: str
    conditions: dict[str, object]
    provenance: dict[str, object]
    measurements: list[MeasurementView]
    external_identifiers: list[ExternalIdentifierView]

###############################################################################
class MaterialRecordView(BaseModel):
    id: int
    name: str
    formula: str | None = None
    molar_mass_g_mol: float | None = None
    structure_count: int = 0
    isotherm_count: int = 0
    external_identifiers: list[ExternalIdentifierView]

###############################################################################
class MaterialPageResponse(BaseModel):
    items: list[MaterialRecordView]
    pagination: Pagination

###############################################################################
class ChemicalPropertyView(BaseModel):
    key: str
    value_number: float | None = None
    value_text: str | None = None
    unit: str | None = None
    source: str

###############################################################################
class ChemicalRecordView(BaseModel):
    id: int
    name: str
    preferred_name: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    connectivity_smiles: str | None = None
    smiles: str | None = None
    pubchem_cid: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    properties: list[ChemicalPropertyView] = Field(default_factory=list)
    external_identifiers: list[ExternalIdentifierView] = Field(default_factory=list)
    structure_2d_url: str | None = None
    conformer_3d_url: str | None = None
    isotherm_count: int = 0
    retrieved_at: datetime | None = None

###############################################################################
class ChemicalPageResponse(BaseModel):
    items: list[ChemicalRecordView]
    pagination: Pagination

###############################################################################
class CODSearchResult(BaseModel):
    cod_id: str
    name: str | None = None
    formula: str | None = None
    space_group: str | None = None
    space_group_number: int | None = None
    cell_a_angstrom: float | None = None
    cell_b_angstrom: float | None = None
    cell_c_angstrom: float | None = None
    cell_alpha_deg: float | None = None
    cell_beta_deg: float | None = None
    cell_gamma_deg: float | None = None
    cell_volume_angstrom3: float | None = None
    doi: str | None = None
    year: int | None = None
    has_coordinates: bool = False
    source_url: str
    cif_url: str

###############################################################################
class CODSearchResponse(BaseModel):
    items: list[CODSearchResult]

###############################################################################
class StructureMaterialLinkRequest(BaseModel):
    adsorbent_id: int | None = Field(default=None, ge=1)

###############################################################################
class NISTCollectionView(BaseModel):
    category: NISTCategory
    local_count: int = Field(ge=0)
    available_count: int | None = Field(default=None, ge=0)
    indexed_at: datetime | None = None

###############################################################################
class NISTCollectionListResponse(BaseModel):
    collections: list[NISTCollectionView]

###############################################################################
class StructureAtomView(BaseModel):
    sequence_index: int
    label: str
    element: str
    fractional_x: float
    fractional_y: float
    fractional_z: float
    occupancy: float | None = None

###############################################################################
class StructureRecordView(BaseModel):
    id: int
    source: str
    external_id: str
    source_url: str | None = None
    material_id: int | None = None
    material_name: str | None = None
    name: str | None = None
    formula: str | None = None
    format: str
    content_sha256: str
    space_group: str | None = None
    space_group_number: int | None = None
    cell_a_angstrom: float | None = None
    cell_b_angstrom: float | None = None
    cell_c_angstrom: float | None = None
    cell_alpha_deg: float | None = None
    cell_beta_deg: float | None = None
    cell_gamma_deg: float | None = None
    cell_volume_angstrom3: float | None = None
    has_coordinates: bool
    atom_count: int
    doi: str | None = None
    reference_title: str | None = None
    retrieved_at: datetime | None = None
    atoms: list[StructureAtomView] = Field(default_factory=list)

###############################################################################
class StructurePageResponse(BaseModel):
    items: list[StructureRecordView]
    pagination: Pagination

###############################################################################
class MaterialDetailResponse(MaterialRecordView):
    structures: list[StructureRecordView] = Field(default_factory=list)


__all__ = [
    "AdsorptionDetailResponse",
    "AdsorptionPageResponse",
    "AdsorptionRecordView",
    "ChemicalPageResponse",
    "ChemicalPropertyView",
    "ChemicalRecordView",
    "CODSearchResponse",
    "CODSearchResult",
    "EntityReference",
    "ExternalIdentifierView",
    "MaterialDetailResponse",
    "MaterialPageResponse",
    "MaterialRecordView",
    "MeasurementView",
    "NISTCategory",
    "NISTCollectionListResponse",
    "NISTCollectionView",
    "Pagination",
    "ProviderCapability",
    "PublicDataSyncCreateRequest",
    "PublicDataSyncFailureListResponse",
    "PublicDataSyncFailureView",
    "PublicDataSyncRunListResponse",
    "PublicDataSyncRunView",
    "PublicSourceListResponse",
    "PublicSourceSummary",
    "SourceStatus",
    "SyncOperation",
    "SyncRunStatus",
    "StructureAtomView",
    "StructureMaterialLinkRequest",
    "StructurePageResponse",
    "StructureRecordView",
]
