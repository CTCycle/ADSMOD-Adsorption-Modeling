# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from server.domain.materials import (
    IdentityAuditResponse,
    IdentityReviewPageResponse,
    MaterialDetailResponse,
    MaterialPageResponse,
    SpeciesDetailResponse,
    SpeciesPageResponse,
)
from server.domain.public_data import Pagination
from server.repositories.identity import IdentityRepository
from server.repositories.materials import MaterialRepository
from server.repositories.public_data import PublicDataRepository

MATERIAL_STRUCTURE_LIMIT = 50


###############################################################################
class MaterialService:
    """Locally managed adsorbent materials and adsorbate species."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: MaterialRepository,
        public_data: PublicDataRepository,
        identity: IdentityRepository,
    ) -> None:
        self.repository = repository
        self.public_data = public_data
        self.identity = identity

    # -------------------------------------------------------------------------
    def list_identity_reviews(self, **filters: Any) -> IdentityReviewPageResponse:
        rows, total = self.identity.list_reviews(**filters)
        return IdentityReviewPageResponse(
            items=rows, pagination=self._pagination(filters, total)
        )

    # -------------------------------------------------------------------------
    def merge_identity_review(self, review_id: int, keep_id: int) -> None:
        self.identity.merge(review_id, keep_id)

    # -------------------------------------------------------------------------
    def reject_identity_review(self, review_id: int) -> None:
        self.identity.reject(review_id)

    # -------------------------------------------------------------------------
    def audit_identities(self) -> IdentityAuditResponse:
        return IdentityAuditResponse(**self.identity.audit())

    # -------------------------------------------------------------------------
    @staticmethod
    def _pagination(filters: dict[str, Any], total: int) -> Pagination:
        return Pagination(
            page=int(filters["page"]), page_size=int(filters["page_size"]), total=total
        )

    # -------------------------------------------------------------------------
    def list_materials(self, **filters: Any) -> MaterialPageResponse:
        rows, total = self.repository.list_materials(**filters)
        return MaterialPageResponse(
            items=rows, pagination=self._pagination(filters, total)
        )

    # -------------------------------------------------------------------------
    def get_material(self, adsorbent_id: int) -> MaterialDetailResponse:
        view = self.repository.get_material(adsorbent_id)
        structures, _ = self.public_data.list_structures(
            page=1,
            page_size=MATERIAL_STRUCTURE_LIMIT,
            material_id=adsorbent_id,
        )
        return MaterialDetailResponse.model_validate({**view, "structures": structures})

    # -------------------------------------------------------------------------
    def list_species(self, **filters: Any) -> SpeciesPageResponse:
        rows, total = self.repository.list_species(**filters)
        return SpeciesPageResponse(items=rows, pagination=self._pagination(filters, total))

    # -------------------------------------------------------------------------
    def get_species(self, adsorbate_id: int) -> SpeciesDetailResponse:
        return SpeciesDetailResponse.model_validate(
            self.repository.get_species(adsorbate_id)
        )

    # -------------------------------------------------------------------------
    def delete_material(self, adsorbent_id: int) -> None:
        self.repository.delete_material(adsorbent_id)

    # -------------------------------------------------------------------------
    def delete_species(self, adsorbate_id: int) -> None:
        self.repository.delete_species(adsorbate_id)
