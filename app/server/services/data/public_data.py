# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from server.domain.public_data import (
    ChemicalLookupResponse,
    CODSearchResponse,
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
from server.repositories.schemas.types import normalize_identity

# A separate word-only qualifier such as "(synthetic)", which COD text search
# treats as a required term. Chemical groups like "Fe(OH)" are not preceded by a
# space, and formula fragments contain digits.
_TRAILING_NAME_QUALIFIER = re.compile(r"\s+\([A-Za-z ]+\)\s*$")
_COD_NAME_FIELDS = ("chemname", "commonname", "mineral")


###############################################################################
def _cod_search_name(name: str) -> str:
    cleaned = _TRAILING_NAME_QUALIFIER.sub("", name).strip()
    return cleaned if len(cleaned) >= 3 else name.strip()


###############################################################################
def _cod_row_matches(row: dict[str, Any], key: str) -> bool:
    metadata = row.get("raw_metadata") or {}
    return any(
        normalize_identity(str(metadata[field])) == key
        for field in _COD_NAME_FIELDS
        if str(metadata.get(field) or "").strip()
    )


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
    def ensure_sources(self) -> None:
        self.repository.ensure_sources(
            [provider.source_definition() for provider in self.providers.values()]
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
    async def resolve_pubchem(
        self, query: str, adsorbate_id: int | None = None
    ) -> tuple[int, bool]:
        """Resolve one compound locally first, then through PubChem.

        Returns the chemical ID and whether a remote record was imported.
        """
        if adsorbate_id is None:
            local_id = await asyncio.to_thread(
                self.repository.find_local_pubchem_adsorbate, query
            )
            if local_id is not None:
                return int(local_id), False
        provider = self.providers.get("pubchem")
        if not isinstance(provider, PubChemProvider):
            raise RuntimeError("PubChem provider is not configured.")
        payload = await provider.resolve(query)
        resolved_id = await asyncio.to_thread(
            self.repository.upsert_pubchem_compound, payload, adsorbate_id
        )
        return int(resolved_id), True

    # -------------------------------------------------------------------------
    async def ensure_conformer(self, adsorbate_id: int) -> tuple[str, bool]:
        """Return the 3D SDF of a PubChem-linked chemical, downloading it once.

        Returns the SDF text and whether it was downloaded now. A cached
        conformer never triggers a provider request.
        """
        state = await asyncio.to_thread(self.repository.conformer_state, adsorbate_id)
        if state is None:
            raise LookupError(
                f"Chemical {adsorbate_id} has no PubChem record; resolve it in PubChem first."
            )
        if state["sdf"]:
            return str(state["sdf"]), False
        provider = self.providers.get("pubchem")
        if not isinstance(provider, PubChemProvider):
            raise RuntimeError("PubChem provider is not configured.")
        sdf = await provider.fetch_conformer(state["cid"])
        await asyncio.to_thread(self.repository.store_conformer, adsorbate_id, sdf)
        return sdf, True

    # -------------------------------------------------------------------------
    async def lookup_pubchem(self, query: str) -> ChemicalLookupResponse:
        """Preview PubChem matches, flagging those already stored locally."""
        provider = self.providers.get("pubchem")
        if not isinstance(provider, PubChemProvider):
            raise RuntimeError("PubChem provider is not configured.")
        items = await provider.candidates(query)
        for item in items:
            item["local_adsorbate_id"] = await asyncio.to_thread(
                self.repository.find_local_pubchem_adsorbate, item["cid"]
            )
        return ChemicalLookupResponse(items=items)

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
        self,
        cod_id: str,
        adsorbent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, bool]:
        """Import one COD record unless it is stored already.

        Returns the structure ID and whether a remote record was imported.
        """
        structure_id = await asyncio.to_thread(
            self.repository.find_cod_structure_id, cod_id
        )
        if structure_id is not None:
            if adsorbent_id is not None:
                structure_id = await asyncio.to_thread(
                    self.repository.link_cod_structure_to_material,
                    cod_id,
                    adsorbent_id,
                )
            return int(structure_id), False
        provider = self.providers.get("cod")
        if not isinstance(provider, CODProvider):
            raise RuntimeError("COD provider is not configured.")
        metadata, cif_text = await provider.fetch_record(cod_id, metadata)
        atoms = provider.parse_atoms(cif_text)
        provider.validate_cif(metadata, cif_text, atoms)
        structure_id = await asyncio.to_thread(
            self.repository.upsert_cod_structure,
            metadata=metadata,
            cif_text=cif_text,
            atoms=atoms,
            adsorbent_id=adsorbent_id,
        )
        return int(structure_id), True

    # -------------------------------------------------------------------------
    async def import_cod_for_material(
        self, adsorbent_id: int, name: str
    ) -> tuple[int | None, bool, str | None]:
        """Import a COD structure for a stored material only on an exact name match.

        Several entries can carry the same name, so the pick is deterministic:
        entries with coordinates first, then the lowest COD ID. Returns
        (structure id, imported, reason). Without an exact match no structure is
        linked and a reason is returned.
        """
        provider = self.providers.get("cod")
        if not isinstance(provider, CODProvider):
            raise RuntimeError("COD provider is not configured.")
        if len(name.strip()) < 3:
            return None, False, "Material name is too short for a COD search."
        search_name = _cod_search_name(name)
        try:
            rows = await provider.search(text=search_name)
        except ValueError as exc:
            return None, False, f"Ambiguous: {exc}"
        if not rows:
            return None, False, "No COD structure found."
        # COD rows often have no chemical name (only a formula and a title).
        key = normalize_identity(search_name)
        exact = [row for row in rows if _cod_row_matches(row, key)]
        if not exact:
            candidates = ", ".join(str(row["cod_id"]) for row in rows[:5])
            return None, False, (
                f"Ambiguous: {len(rows)} COD candidates, no exact name match "
                f"(e.g. {candidates}); choose one manually."
            )
        selected = min(
            exact,
            key=lambda row: (not row.get("has_coordinates"), int(row["cod_id"])),
        )
        # The search row already carries the metadata, so the import need not
        # look the entry up again.
        structure_id, imported = await self.import_cod(
            str(selected["cod_id"]), adsorbent_id, metadata=selected
        )
        return structure_id, imported, None

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

    # -------------------------------------------------------------------------
    def set_structure_material(
        self, structure_id: int, adsorbent_id: int | None
    ) -> StructureRecordView:
        self.repository.set_structure_material(structure_id, adsorbent_id)
        return self.get_structure(structure_id)

    # -------------------------------------------------------------------------
    def get_structure_content(self, structure_id: int) -> tuple[str, str]:
        return self.repository.get_structure_content(structure_id)


__all__ = ["PublicDataService"]
