# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, status

from server.domain.materials import (
    MaterialDetailResponse,
    MaterialPageResponse,
    SpeciesDetailResponse,
    SpeciesPageResponse,
)
from server.repositories.materials import EntityInUseError
from server.services.container import CoreServiceContainer
from server.services.data.materials import MaterialService


###############################################################################
class MaterialEndpoint:
    """Locally managed adsorbent materials and chemical species."""

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: MaterialService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def not_found(exc: LookupError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def conflict(exc: EntityInUseError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    # -------------------------------------------------------------------------
    def list_materials(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str | None = Query(default=None, max_length=512),
        formula: str | None = Query(default=None, max_length=255),
        source: str | None = Query(default=None, max_length=64),
        has_structure: bool | None = None,
    ) -> MaterialPageResponse:
        return self.service.list_materials(
            page=page,
            page_size=page_size,
            query_text=q,
            formula=formula,
            source=source,
            has_structure=has_structure,
        )

    # -------------------------------------------------------------------------
    def get_material(
        self, adsorbent_id: int = Path(..., ge=1)
    ) -> MaterialDetailResponse:
        try:
            return self.service.get_material(adsorbent_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def delete_material(self, adsorbent_id: int = Path(..., ge=1)) -> None:
        try:
            self.service.delete_material(adsorbent_id)
        except EntityInUseError as exc:
            raise self.conflict(exc) from exc
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def list_species(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str | None = Query(default=None, max_length=512),
        formula: str | None = Query(default=None, max_length=255),
        source: str | None = Query(default=None, max_length=64),
        molecular_weight_min: float | None = Query(default=None, ge=0),
        molecular_weight_max: float | None = Query(default=None, ge=0),
    ) -> SpeciesPageResponse:
        if (
            molecular_weight_min is not None
            and molecular_weight_max is not None
            and molecular_weight_min > molecular_weight_max
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="molecular_weight_min cannot exceed molecular_weight_max.",
            )
        return self.service.list_species(
            page=page,
            page_size=page_size,
            query_text=q,
            formula=formula,
            source=source,
            molecular_weight_min=molecular_weight_min,
            molecular_weight_max=molecular_weight_max,
        )

    # -------------------------------------------------------------------------
    def get_species(self, adsorbate_id: int = Path(..., ge=1)) -> SpeciesDetailResponse:
        try:
            return self.service.get_species(adsorbate_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def delete_species(self, adsorbate_id: int = Path(..., ge=1)) -> None:
        try:
            self.service.delete_species(adsorbate_id)
        except EntityInUseError as exc:
            raise self.conflict(exc) from exc
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/materials",
            self.list_materials,
            methods=["GET"],
            response_model=MaterialPageResponse,
        )
        self.router.add_api_route(
            "/materials/{adsorbent_id}",
            self.get_material,
            methods=["GET"],
            response_model=MaterialDetailResponse,
        )
        self.router.add_api_route(
            "/materials/{adsorbent_id}",
            self.delete_material,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
        )
        self.router.add_api_route(
            "/species",
            self.list_species,
            methods=["GET"],
            response_model=SpeciesPageResponse,
        )
        self.router.add_api_route(
            "/species/{adsorbate_id}",
            self.get_species,
            methods=["GET"],
            response_model=SpeciesDetailResponse,
        )
        self.router.add_api_route(
            "/species/{adsorbate_id}",
            self.delete_species,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
        )


###############################################################################
def create_material_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(tags=["materials"])
    MaterialEndpoint(router, container.material_service).add_routes()
    return router
