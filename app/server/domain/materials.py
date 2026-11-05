# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from server.domain.public_data import (
    ExternalIdentifierView,
    Pagination,
    StructureRecordView,
)


###############################################################################
class DatasetUsage(BaseModel):
    id: int
    name: str
    experiment_count: int = Field(ge=0)


###############################################################################
class MaterialRecordView(BaseModel):
    id: int
    name: str
    formula: str | None = None
    molar_mass_g_mol: float | None = None
    smiles: str | None = None
    structure_count: int = 0
    isotherm_count: int = 0
    external_identifiers: list[ExternalIdentifierView] = Field(default_factory=list)


###############################################################################
class MaterialPageResponse(BaseModel):
    items: list[MaterialRecordView]
    pagination: Pagination


###############################################################################
class MaterialDetailResponse(MaterialRecordView):
    structures: list[StructureRecordView] = Field(default_factory=list)
    datasets: list[DatasetUsage] = Field(default_factory=list)


###############################################################################
class SpeciesPropertyView(BaseModel):
    key: str
    value_number: float | None = None
    value_text: str | None = None
    unit: str | None = None
    source: str


###############################################################################
class SpeciesRecordView(BaseModel):
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
    properties: list[SpeciesPropertyView] = Field(default_factory=list)
    external_identifiers: list[ExternalIdentifierView] = Field(default_factory=list)
    structure_2d_url: str | None = None
    conformer_3d_url: str | None = None
    isotherm_count: int = 0
    retrieved_at: datetime | None = None


###############################################################################
class SpeciesPageResponse(BaseModel):
    items: list[SpeciesRecordView]
    pagination: Pagination


###############################################################################
class SpeciesDetailResponse(SpeciesRecordView):
    datasets: list[DatasetUsage] = Field(default_factory=list)
