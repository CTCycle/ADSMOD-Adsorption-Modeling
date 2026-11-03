# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DatasetStructure = Literal["atomic", "aggregated", "mixed", "ambiguous"]
PressureBasis = Literal["absolute", "partial", "relative"]
UnitQuantity = Literal["pressure", "uptake", "temperature"]
ColumnRole = Literal[
    "experiment_id",
    "experiment_name",
    "pressure",
    "uptake",
    "adsorbate",
    "adsorbate_smiles",
    "adsorbent",
    "temperature",
    "pressure_unit",
    "uptake_unit",
    "temperature_unit",
    "uptake_stddev",
    "saturation_pressure",
    "metadata",
    "ignore",
]

###############################################################################
class ImportIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["error", "warning", "confirmation"]
    message: str
    column: str | None = None
    source_row: int | None = None
    experiment: str | None = None
    remediation: str | None = None

###############################################################################
class ColumnDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    inferred_type: str
    sample_values: list[Any] = Field(default_factory=list)
    proposed_role: ColumnRole = "ignore"
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    detected_unit: str | None = None
    array_like: bool = False

###############################################################################
class WidePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure_column: str
    uptake_column: str

###############################################################################
class CustomUnitDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: UnitQuantity
    symbol: str = Field(min_length=1, max_length=64)
    factor_to_canonical: float | None = Field(default=None, gt=0, allow_inf_nan=False)

###############################################################################
class ImportMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1, max_length=255)
    structure: Literal["atomic", "aggregated", "mixed"]
    column_roles: dict[str, ColumnRole]
    grouping_columns: list[str] = Field(default_factory=list)
    whole_file_grouping: bool = False
    constants: dict[str, str | float] = Field(default_factory=dict)
    unit_overrides: dict[str, str] = Field(default_factory=dict)
    unit_definitions: list[CustomUnitDefinition] = Field(default_factory=list)
    pressure_basis: PressureBasis
    decimal_separator: Literal["auto", ".", ","] = "auto"
    field_delimiter: str | None = None
    series_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    wide_pairs: list[WidePair] = Field(default_factory=list)
    worksheet: str | int | None = None
    header_row: int = Field(default=0, ge=0)
    thousands_separator: str | None = None
    encoding: str = "utf-8"
    duplicate_policy: Literal["keep", "average", "reject"] = "reject"
    confirmed_issue_codes: list[str] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_unique_roles(self) -> ImportMapping:
        single_roles = {
            "pressure",
            "uptake",
            "temperature",
            "adsorbate_smiles",
            "pressure_unit",
            "uptake_unit",
            "temperature_unit",
            "uptake_stddev",
            "saturation_pressure",
        }
        used: dict[str, str] = {}
        for column, role in self.column_roles.items():
            if role in single_roles and role in used:
                raise ValueError(
                    f"Columns '{used[role]}' and '{column}' are both mapped to '{role}'."
                )
            if role in single_roles:
                used[role] = column
        return self

###############################################################################
class ImportPreviewResponse(BaseModel):
    status: Literal["success"] = "success"
    filename: str
    source_sha256: str
    row_count: int
    column_count: int
    detected_structure: DatasetStructure
    structure_confidence: float = Field(ge=0, le=1)
    columns: list[ColumnDetection]
    preview_rows: list[dict[str, Any]]
    proposed_grouping_columns: list[str] = Field(default_factory=list)
    proposed_pressure_basis: PressureBasis | None = None
    issues: list[ImportIssue] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)

###############################################################################
class NormalizedObservationPreview(BaseModel):
    source_row: int | None
    sequence_index: int
    pressure_original: float
    pressure_original_unit: str
    pressure_canonical: float
    pressure_canonical_unit: str
    uptake_original: float
    uptake_original_unit: str
    uptake_mol_kg: float

###############################################################################
class NormalizedExperimentPreview(BaseModel):
    external_key: str
    name: str
    adsorbent: str
    adsorbate: str
    adsorbate_smiles: str | None = None
    temperature_k: float
    temperature_original: float
    temperature_original_unit: str
    pressure_basis: PressureBasis
    observation_count: int
    observations: list[NormalizedObservationPreview]

###############################################################################
class ImportValidationResponse(BaseModel):
    status: Literal["valid", "invalid", "confirmation_required"]
    source_sha256: str
    structure: Literal["atomic", "aggregated", "mixed"]
    experiment_count: int
    observation_count: int
    experiments: list[NormalizedExperimentPreview]
    issues: list[ImportIssue]

###############################################################################
class DatasetSummary(BaseModel):
    id: int
    name: str
    source: Literal["uploaded", "nist"]
    created_at: str
    experiment_count: int
    observation_count: int
    tags: list[str] = Field(default_factory=list)
    description: str = ""

###############################################################################
class DatasetListResponse(BaseModel):
    status: Literal["success"] = "success"
    datasets: list[DatasetSummary] = Field(default_factory=list)

###############################################################################
class UnitOption(BaseModel):
    quantity: UnitQuantity
    symbol: str
    label: str
    name: str
    group: str
    note: str = ""
    aliases: list[str] = Field(default_factory=list)
    built_in: bool = True
    has_conversion: bool = True
    factor_to_canonical: float | None = None
    id: int | None = None

###############################################################################
class SupportedUnitsResponse(BaseModel):
    pressure: list[UnitOption]
    uptake: list[UnitOption]
    temperature: list[UnitOption]
    allowed_extensions: list[str]

###############################################################################
class CustomUnitRequest(CustomUnitDefinition):
    pass

###############################################################################
class CustomUnitResponse(BaseModel):
    status: Literal["success"] = "success"
    unit: UnitOption

###############################################################################
class DatasetImportResponse(BaseModel):
    status: Literal["success"] = "success"
    dataset: DatasetSummary
    validation: ImportValidationResponse

###############################################################################
class DatasetImportInspectionResponse(BaseModel):
    status: Literal["success"] = "success"
    dataset_id: int
    original_filename: str
    source_sha256: str
    source_structure: Literal["atomic", "aggregated", "mixed"]
    parser_version: str
    mapping: ImportMapping
    validation: ImportValidationResponse
    warnings: list[ImportIssue] = Field(default_factory=list)

###############################################################################
class ExperimentSummary(BaseModel):
    id: int
    dataset_id: int
    external_key: str
    name: str
    adsorbent: str
    adsorbates: list[str]
    temperature_k: float
    temperature_original: float | None = None
    temperature_original_unit: str | None = None
    pressure_basis: PressureBasis
    observation_count: int
    pressure_min: float | None = None
    pressure_max: float | None = None
    pressure_canonical_unit: str | None = None
    fitting_eligible: bool
    ineligibility_reason: str | None = None

###############################################################################
class ExperimentListResponse(BaseModel):
    status: Literal["success"] = "success"
    experiments: list[ExperimentSummary] = Field(default_factory=list)

###############################################################################
class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(default_factory=list, max_length=32)
    description: str = Field(default="", max_length=2000)

###############################################################################
class DatasetRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_name: str = Field(min_length=1, max_length=255)

###############################################################################
class DatasetMutationResponse(BaseModel):
    status: Literal["success"] = "success"
    dataset: DatasetSummary

###############################################################################
class ObservationRow(BaseModel):
    id: int
    sequence_index: int
    source_row: int | None = None
    pressure_original: float
    pressure_original_unit: str
    pressure_canonical: float
    pressure_canonical_unit: str
    uptake_original: float
    uptake_original_unit: str
    uptake_mol_kg: float
    uptake_stddev_mol_kg: float | None = None

###############################################################################
class ObservationPage(BaseModel):
    status: Literal["success"] = "success"
    dataset_id: int
    isotherm_id: int
    offset: int
    limit: int
    total: int
    rows: list[ObservationRow]
