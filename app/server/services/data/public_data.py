# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from server.domain.public_data import (
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    CODSearchResponse,
    CODStructureImportRequest,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialPageResponse,
    Pagination,
    PublicDataOverviewResponse,
    PublicSourceListResponse,
    PublicSourceSummary,
    StructurePageResponse,
    StructureRecordView,
)
from server.services.providers.cod import CODProvider
from server.services.providers.pubchem import PubChemProvider
from server.services.providers.public_data import PublicDataProvider
from server.repositories.public_data import PublicDataRepository

###############################################################################
class PublicDataService:
    """Coordinate normalized public-data queries and provider adapters."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        repository: PublicDataRepository,
        providers: list[PublicDataProvider],
        health_cache_ttl_seconds: int = 900,
    ) -> None:
        self.repository = repository
        self.providers = {provider.key: provider for provider in providers}
        self.health_cache_ttl = timedelta(seconds=max(30, health_cache_ttl_seconds))

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await asyncio.gather(
            *(provider.close() for provider in self.providers.values()),
            return_exceptions=True,
        )

    # -------------------------------------------------------------------------
    def list_sources(self) -> PublicSourceListResponse:
        persisted = {row["key"]: row for row in self.repository.source_rows()}
        items: list[PublicSourceSummary] = []
        for key, provider in sorted(self.providers.items(), key=lambda item: item[1].name):
            row = persisted.get(key, {})
            items.append(
                PublicSourceSummary(
                    key=key,
                    name=provider.name,
                    description=provider.description,
                    capabilities=[capability.value for capability in provider.capabilities],
                    homepage_url=provider.homepage_url,
                    license_name=provider.license_name,
                    license_url=provider.license_url,
                    terms_url=provider.terms_url,
                    record_count=int(row.get("record_count", 0)),
                    health_status=row.get("health_status", "unknown"),
                    health_detail=row.get("health_detail"),
                    last_health_checked_at=row.get("last_health_checked_at"),
                    health_fresh=self._health_is_fresh(
                        row.get("last_health_checked_at")
                    ),
                    last_successful_sync_at=row.get("last_successful_sync_at"),
                )
            )
        return PublicSourceListResponse(sources=items)

    # -------------------------------------------------------------------------
    def _health_is_fresh(self, checked_at: datetime | None) -> bool:
        if checked_at is None:
            return False
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - checked_at <= self.health_cache_ttl

    # -------------------------------------------------------------------------
    def get_overview(self) -> PublicDataOverviewResponse:
        return PublicDataOverviewResponse.model_validate(self.repository.overview_counts())

    # -------------------------------------------------------------------------
    def list_adsorption(self, **filters: Any) -> AdsorptionPageResponse:
        rows, total = self.repository.list_adsorption(**filters)
        return AdsorptionPageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def get_adsorption(self, isotherm_id: int) -> AdsorptionDetailResponse:
        return AdsorptionDetailResponse.model_validate(
            self.repository.get_adsorption(isotherm_id)
        )

    # -------------------------------------------------------------------------
    def list_materials(self, **filters: Any) -> MaterialPageResponse:
        rows, total = self.repository.list_materials(**filters)
        return MaterialPageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def list_chemicals(self, **filters: Any) -> ChemicalPageResponse:
        rows, total = self.repository.list_chemicals(**filters)
        return ChemicalPageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def get_chemical(self, adsorbate_id: int) -> ChemicalRecordView:
        return ChemicalRecordView.model_validate(
            self.repository.get_chemical(adsorbate_id)
        )

    # -------------------------------------------------------------------------
    async def resolve_pubchem(self, query: str) -> ChemicalRecordView:
        local_id = await asyncio.to_thread(
            self.repository.find_local_pubchem_adsorbate, query
        )
        if local_id is not None:
            return await asyncio.to_thread(self.get_chemical, local_id)
        provider = self.providers.get("pubchem")
        if not isinstance(provider, PubChemProvider):
            raise RuntimeError("PubChem provider is not configured.")
        payload = await provider.resolve(query)
        adsorbate_id = await asyncio.to_thread(
            self.repository.upsert_pubchem_compound, payload
        )
        return self.get_chemical(adsorbate_id)

    # -------------------------------------------------------------------------
    async def search_cod(
        self,
        *,
        text: str | None,
        formula: str | None,
        cod_id: str | None,
    ) -> CODSearchResponse:
        provider = self.providers.get("cod")
        if not isinstance(provider, CODProvider):
            raise RuntimeError("COD provider is not configured.")
        rows = await provider.search(text=text, formula=formula, cod_id=cod_id)
        return CODSearchResponse(items=rows)

    # -------------------------------------------------------------------------
    async def import_cod(
        self, request: CODStructureImportRequest
    ) -> StructureRecordView:
        structure_id = await asyncio.to_thread(
            self.repository.find_cod_structure_id, request.cod_id
        )
        if structure_id is not None:
            if request.adsorbent_id is not None:
                structure_id = await asyncio.to_thread(
                    self.repository.link_cod_structure_to_material,
                    request.cod_id,
                    request.adsorbent_id,
                )
            payload = await asyncio.to_thread(
                self.repository.get_structure, structure_id
            )
            return StructureRecordView.model_validate(payload)
        provider = self.providers.get("cod")
        if not isinstance(provider, CODProvider):
            raise RuntimeError("COD provider is not configured.")
        metadata, cif_text = await provider.fetch_record(request.cod_id)
        atoms = provider.parse_atoms(cif_text)
        provider.validate_cif(metadata, cif_text, atoms)
        structure_id = await asyncio.to_thread(
            self.repository.upsert_cod_structure,
            metadata=metadata,
            cif_text=cif_text,
            atoms=atoms,
            adsorbent_id=request.adsorbent_id,
        )
        return StructureRecordView.model_validate(
            self.repository.get_structure(structure_id)
        )

    # -------------------------------------------------------------------------
    def list_structures(self, **filters: Any) -> StructurePageResponse:
        rows, total = self.repository.list_structures(**filters)
        return StructurePageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def get_structure(self, structure_id: int) -> StructureRecordView:
        return StructureRecordView.model_validate(
            self.repository.get_structure(structure_id)
        )


__all__ = ["PublicDataService"]
