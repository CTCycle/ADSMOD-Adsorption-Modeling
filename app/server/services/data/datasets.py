# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import hashlib
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from server.configurations.settings import DEFAULT_DATASET_ALLOWED_EXTENSIONS
from server.domain.datasets import (
    DatasetDeleteImpact,
    DatasetDetailResponse,
    DatasetImportResponse,
    DatasetImportInspectionResponse,
    DatasetListResponse,
    DatasetMutationResponse,
    DatasetSummary,
    DatasetUpdateRequest,
    CustomUnitRequest,
    CustomUnitResponse,
    ExperimentDetailResponse,
    ExperimentPage,
    ImportMapping,
    ImportPreviewResponse,
    ImportValidationResponse,
    ObservationPage,
    SupportedUnitsResponse,
    UnitOption,
)
from server.services.data.dataset_editing import DatasetEditService
from server.services.data.importer import AdsorptionImportEngine, PARSER_VERSION
from server.common.units import BUILT_IN_UNIT_DETAILS, UnitRegistry
from server.repositories.datasets import DatasetRepository
from server.repositories.units import UnitRepository

###############################################################################
class DatasetService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: DatasetRepository,
        units: UnitRepository,
        editor: DatasetEditService,
        importer: AdsorptionImportEngine | None = None,
        allowed_extensions: Collection[str] = DEFAULT_DATASET_ALLOWED_EXTENSIONS,
        column_detection_cutoff: float = 0.6,
        preview_row_limit: int = 20,
    ) -> None:
        self.repository = repository
        self.units = units
        self.editor = editor
        self.importer = importer or AdsorptionImportEngine(
            allowed_extensions,
            column_detection_cutoff=column_detection_cutoff,
            preview_row_limit=preview_row_limit,
        )
        self.allowed_extensions = list(
            getattr(self.importer, "allowed_extensions", allowed_extensions)
        )

    # -------------------------------------------------------------------------
    def apply_settings(
        self,
        *,
        allowed_extensions: Collection[str],
        column_detection_cutoff: float,
        preview_row_limit: int,
    ) -> None:
        self.allowed_extensions = list(allowed_extensions)
        self.importer.allowed_extensions = tuple(allowed_extensions)
        self.importer.column_detection_cutoff = float(column_detection_cutoff)
        self.importer.preview_row_limit = int(preview_row_limit)

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_mapping(mapping_json: str) -> ImportMapping:
        try:
            decoded = json.loads(mapping_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Import mapping must be valid JSON.") from exc
        try:
            return ImportMapping.model_validate(decoded)
        except ValidationError as exc:
            first = exc.errors()[0]
            raise ValueError(str(first.get("msg", "Invalid import mapping."))) from exc

    # -------------------------------------------------------------------------
    def preview(self, payload: bytes, filename: str | None) -> ImportPreviewResponse:
        return self.importer.preview(payload, filename)

    # -------------------------------------------------------------------------
    def validate(
        self, payload: bytes, filename: str | None, mapping: ImportMapping
    ) -> ImportValidationResponse:
        return self.importer.validate(
            payload, filename, mapping, self.units.custom_units()
        ).response

    # -------------------------------------------------------------------------
    def commit(
        self, payload: bytes, filename: str | None, mapping: ImportMapping
    ) -> DatasetImportResponse:
        bundle = self.importer.validate(
            payload, filename, mapping, self.units.custom_units()
        )
        if bundle.response.status != "valid":
            raise ValueError(
                "Dataset validation must succeed and all required confirmations "
                "must be acknowledged before saving."
            )
        manifest = {
            "original_filename": Path(filename or "dataset").name,
            "source_sha256": bundle.response.source_sha256,
            "source_structure": mapping.structure,
            "parser_version": PARSER_VERSION,
            "column_mapping": mapping.model_dump(mode="json"),
            "validation_result": bundle.response.model_dump(mode="json"),
            "warnings": [
                issue.model_dump(mode="json")
                for issue in bundle.response.issues
                if issue.severity == "warning"
            ],
        }
        mapping_sha256 = hashlib.sha256(
            json.dumps(
                mapping.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        manifest["mapping_sha256"] = mapping_sha256
        dataset_id = self.repository.persist_canonical(
            name=mapping.dataset_name,
            source="uploaded",
            provenance={
                "origin": "user_upload",
                "original_filename": Path(filename or "dataset").name,
                "source_sha256": bundle.response.source_sha256,
                "mapping_sha256": mapping_sha256,
            },
            experiments=bundle.experiments,
            import_manifest=manifest,
        )
        return DatasetImportResponse(
            dataset=DatasetSummary(**self.repository.summary(dataset_id)),
            validation=bundle.response,
        )

    # -------------------------------------------------------------------------
    def list_datasets(self) -> DatasetListResponse:
        return DatasetListResponse(
            datasets=[
                DatasetSummary(**record) for record in self.repository.list_summaries()
            ]
        )

    # -------------------------------------------------------------------------
    def import_inspection(
        self, dataset_id: int
    ) -> DatasetImportInspectionResponse:
        return DatasetImportInspectionResponse(
            **self.repository.import_inspection(dataset_id)
        )

    # -------------------------------------------------------------------------
    def supported_units(self) -> SupportedUnitsResponse:
        options: dict[str, list[UnitOption]] = {
            "pressure": [],
            "uptake": [],
            "temperature": [],
        }
        for detail in BUILT_IN_UNIT_DETAILS:
            aliases = [
                alias
                for alias, symbol in self.alias_table(detail.quantity).items()
                if symbol == detail.symbol and alias != detail.label
            ]
            options[detail.quantity].append(
                UnitOption(
                    quantity=detail.quantity,  # type: ignore[arg-type]
                    symbol=detail.symbol,
                    label=detail.label,
                    name=detail.name,
                    group=detail.group,
                    note=detail.note,
                    aliases=aliases,
                )
            )
        for item in self.units.list_units():
            options[item["quantity"]].append(self.custom_option(item))
        return SupportedUnitsResponse(
            **options, allowed_extensions=self.allowed_extensions
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def alias_table(quantity: str) -> dict[str, str]:
        return {
            "pressure": UnitRegistry.PRESSURE_ALIASES,
            "uptake": UnitRegistry.UPTAKE_ALIASES,
            "temperature": UnitRegistry.TEMPERATURE_ALIASES,
        }[quantity]

    # -------------------------------------------------------------------------
    @staticmethod
    def custom_option(item: dict[str, Any]) -> UnitOption:
        factor = item["factor_to_canonical"]
        return UnitOption(
            quantity=item["quantity"],
            symbol=item["symbol"],
            label=item["symbol"],
            name="Custom unit",
            group="Saved custom units",
            note=(
                "Saved label without a conversion definition."
                if factor is None
                else f"User-defined factor to canonical unit: {factor:g}"
            ),
            built_in=False,
            has_conversion=factor is not None,
            factor_to_canonical=factor,
            id=item["id"],
        )

    # -------------------------------------------------------------------------
    def save_custom_unit(self, request: CustomUnitRequest) -> CustomUnitResponse:
        item = self.units.add(
            request.quantity, request.symbol, request.factor_to_canonical
        )
        return CustomUnitResponse(unit=self.custom_option(item))

    # -------------------------------------------------------------------------
    def delete_custom_unit(self, unit_id: int) -> None:
        self.units.delete(unit_id)

    # -------------------------------------------------------------------------
    def get_dataset(self, dataset_id: int) -> DatasetDetailResponse:
        return DatasetDetailResponse(**self.repository.detail(dataset_id))

    # -------------------------------------------------------------------------
    def delete_impact(self, dataset_id: int) -> DatasetDeleteImpact:
        return DatasetDeleteImpact(**self.repository.delete_impact(dataset_id))

    # -------------------------------------------------------------------------
    def list_experiments(
        self, dataset_id: int, *, offset: int, limit: int, **filters: Any
    ) -> ExperimentPage:
        items, total = self.repository.experiments(
            dataset_id, offset=offset, limit=limit, **filters
        )
        return ExperimentPage(
            dataset_id=dataset_id,
            offset=offset,
            limit=limit,
            total=total,
            experiments=items,
        )

    # -------------------------------------------------------------------------
    def experiment_detail(
        self, dataset_id: int, experiment_id: int
    ) -> ExperimentDetailResponse:
        return ExperimentDetailResponse(
            **self.repository.experiment_detail(dataset_id, experiment_id)
        )

    # -------------------------------------------------------------------------
    def get_observations(
        self, dataset_id: int, isotherm_id: int, offset: int, limit: int, order: str
    ) -> ObservationPage:
        rows, total = self.repository.observations(
            dataset_id, isotherm_id, offset, limit, order
        )
        return ObservationPage(
            dataset_id=dataset_id,
            isotherm_id=isotherm_id,
            offset=offset,
            limit=limit,
            total=total,
            rows=rows,
        )

    # -------------------------------------------------------------------------
    def update(
        self, dataset_id: int, request: DatasetUpdateRequest
    ) -> DatasetMutationResponse:
        self.editor.apply(dataset_id, request)
        return DatasetMutationResponse(
            dataset=DatasetSummary(**self.repository.summary(dataset_id))
        )

    # -------------------------------------------------------------------------
    def delete(self, dataset_id: int) -> None:
        self.repository.delete(dataset_id)
