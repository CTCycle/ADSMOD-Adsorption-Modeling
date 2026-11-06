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
    # Local coverage for providers that publish no reliable catalogue total:
    # PubChem identities over all stored species, COD structures over all
    # stored adsorbent materials.
    species_total: int = Field(default=0, ge=0)
    species_enriched: int = Field(default=0, ge=0)
    materials_total: int = Field(default=0, ge=0)
    materials_with_structure: int = Field(default=0, ge=0)
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
    overwrite_local_edits: bool = False
    # pubchem_enrich only: download missing 3D conformers instead of resolving identities.
    conformers: bool = False
    identifiers: list[str] | None = Field(default=None, min_length=1, max_length=200)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_operation_parameters(self) -> "PublicDataSyncCreateRequest":
        if self.conformers and self.operation != "pubchem_enrich":
            raise ValueError("conformers applies only to pubchem_enrich runs.")
        # COD imports either name explicit identifiers or run in bulk over the
        # local materials that have no structure yet (selected by `fraction`).
        cod_explicit = (self.cod_id is not None) != (self.cod_ids is not None)
        cod_bulk = (
            self.cod_id is None and self.cod_ids is None and self.fraction is not None
        )
        required = {
            "nist_index": ("nist", self.category is not None),
            "nist_category": ("nist", self.category is not None),
            "pubchem_resolve": ("pubchem", bool(self.query and self.query.strip())),
            "pubchem_enrich": ("pubchem", True),
            "cod_import": ("cod", cod_explicit or cod_bulk),
        }
        source, has_value = required[self.operation]
        if source != self.source_key or not has_value:
            raise ValueError(
                f"{self.operation} requires source_key={source!r} and its matching parameters."
            )
        if self.query is not None:
            self.query = self.query.strip()
        if self.identifiers is not None:
            if self.operation != "nist_category":
                raise ValueError("identifiers apply only to nist_category runs.")
            self.identifiers = [item.strip() for item in self.identifiers]
            if any(not item for item in self.identifiers) or len(
                {item.lower() for item in self.identifiers}
            ) != len(self.identifiers):
                raise ValueError("identifiers must be unique and not empty.")
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
            "nist_category": (
                "category",
                "fraction",
                "identifiers",
                "overwrite_local_edits",
            ),
            "pubchem_resolve": ("query", "adsorbate_id"),
            "pubchem_enrich": ("fraction", "conformers"),
            "cod_import": ("cod_id", "cod_ids", "adsorbent_id", "fraction"),
        }[self.operation]
        values = {name: getattr(self, name) for name in fields}
        if self.operation == "nist_category":
            if values["identifiers"] is not None:
                values["fraction"] = None
            elif values["fraction"] is None:
                values["fraction"] = 1.0
        if self.operation == "cod_import" and (
            self.cod_id is not None or self.cod_ids is not None
        ):
            values["fraction"] = None
        for flag in ("overwrite_local_edits", "conformers"):
            if flag in values and not values[flag]:
                del values[flag]
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
class NISTSearchItem(BaseModel):
    filename: str
    doi: str | None = None
    adsorbent_name: str | None = None
    adsorbent_hashkey: str | None = None
    adsorbate_names: list[str] = Field(default_factory=list)
    adsorbate_inchi_keys: list[str] = Field(default_factory=list)
    temperature_k: float | None = None
    local: bool = False

###############################################################################
class NISTSearchResponse(BaseModel):
    items: list[NISTSearchItem]
    total: int = Field(ge=0)

###############################################################################
class ChemicalLookupItem(BaseModel):
    cid: str
    name: str
    formula: str | None = None
    molecular_weight: float | None = None
    inchi_key: str | None = None
    smiles: str | None = None
    source_url: str
    structure_2d_url: str | None = None
    local_adsorbate_id: int | None = None

###############################################################################
class ChemicalLookupResponse(BaseModel):
    items: list[ChemicalLookupItem]

###############################################################################
class NISTEditedExperimentsResponse(BaseModel):
    count: int = Field(ge=0)

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


__all__ = [
    "ChemicalLookupItem",
    "ChemicalLookupResponse",
    "NISTSearchItem",
    "NISTSearchResponse",
    "CODSearchResponse",
    "CODSearchResult",
    "EntityReference",
    "ExternalIdentifierView",
    "NISTCategory",
    "NISTCollectionListResponse",
    "NISTCollectionView",
    "NISTEditedExperimentsResponse",
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
