# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from server.configurations.settings import DEFAULT_DATASET_ALLOWED_EXTENSIONS
from server.domain.canonical import CanonicalizationError, normalize_experiment
from server.domain.datasets import (
    ColumnDetection,
    ImportIssue,
    ImportMapping,
    ImportPreviewResponse,
    ImportValidationResponse,
    NormalizedExperimentPreview,
    NormalizedObservationPreview,
)
from server.services.data.import_parser import (
    detect_structure,
    infer_column,
    infer_pressure_basis,
    parse_series_cell,
    propose_species_columns,
    read_tabular,
    safe_cell,
    source_hash,
)
from server.common.units import (
    KNOWN_MOLAR_MASSES_G_MOL,
    CustomUnit,
    UnitCatalog,
    UnitConversionError,
    detect_header_unit,
    parse_number,
)

PARSER_VERSION = "2.0"
DEFAULT_COLUMN_DETECTION_CUTOFF = 0.6
DEFAULT_PREVIEW_ROW_LIMIT = 20
EXPERIMENT_PREVIEW_POINTS = 25
UNIT_QUANTITIES = ("pressure", "uptake", "temperature")
SPECIES_UPTAKE_COLUMN = "__species_uptake__"
SPECIES_ADSORBATE_COLUMN = "__species_adsorbate__"
SPECIES_FRACTION_COLUMN = "__species_gas_mole_fraction__"

###############################################################################
@dataclass
class ValidationBundle:
    response: ImportValidationResponse
    experiments: list[dict[str, Any]]

###############################################################################
class AdsorptionImportEngine:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        allowed_extensions: Collection[str] = DEFAULT_DATASET_ALLOWED_EXTENSIONS,
        *,
        column_detection_cutoff: float = DEFAULT_COLUMN_DETECTION_CUTOFF,
        preview_row_limit: int = DEFAULT_PREVIEW_ROW_LIMIT,
    ) -> None:
        self.allowed_extensions = tuple(
            extension.strip().casefold()
            if extension.strip().startswith(".")
            else f".{extension.strip().casefold()}"
            for extension in allowed_extensions
            if extension.strip()
        )
        if not self.allowed_extensions:
            raise ValueError("At least one dataset file extension is required.")
        self.column_detection_cutoff = float(column_detection_cutoff)
        self.preview_row_limit = max(1, int(preview_row_limit))

    # -------------------------------------------------------------------------
    def preview(self, payload: bytes, filename: str | None) -> ImportPreviewResponse:
        frame = read_tabular(
            payload,
            filename,
            allowed_extensions=self.allowed_extensions,
        )
        columns = [infer_column(name, frame[name]) for name in frame.columns]
        structure, confidence = detect_structure(columns)
        species_columns = propose_species_columns(columns)
        if species_columns:
            structure, confidence = "atomic", 0.9
        grouping = [
            column.name
            for column in columns
            if column.proposed_role == "experiment_id"
            and column.confidence >= self.column_detection_cutoff
        ][:1]
        issues: list[ImportIssue] = []
        if structure == "ambiguous":
            issues.append(
                ImportIssue(
                    code="ambiguous_structure",
                    severity="confirmation",
                    message="The file structure could not be classified safely.",
                    remediation="Choose atomic or aggregated layout and map the measurement columns.",
                )
            )
        for role in ("pressure", "uptake", "temperature", "adsorbate", "adsorbent"):
            if species_columns and role in {"uptake", "adsorbate"}:
                continue
            matches = [
                column
                for column in columns
                if column.proposed_role == role
                and column.confidence >= self.column_detection_cutoff
            ]
            if not matches:
                issues.append(
                    ImportIssue(
                        code=f"unmapped_{role}",
                        severity="confirmation",
                        message=f"No reliable {role.replace('_', ' ')} column was detected.",
                        remediation="Map a source column or provide a dataset-level constant.",
                    )
                )
        return ImportPreviewResponse(
            filename=filename or "dataset",
            source_sha256=source_hash(payload),
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            detected_structure=structure,  # type: ignore[arg-type]
            structure_confidence=confidence,
            columns=columns,
            preview_rows=[
                {key: safe_cell(value) for key, value in row.items()}
                for row in frame.head(self.preview_row_limit).to_dict(orient="records")
            ],
            proposed_grouping_columns=grouping,
            proposed_pressure_basis=infer_pressure_basis(columns),  # type: ignore[arg-type]
            proposed_species_columns=species_columns,
            issues=issues,
            guidance=[
                "Atomic one-observation-per-row data is the recommended format.",
                "Files with one uptake column per species are converted to one row per species and measurement.",
                "Aggregated formats are accepted only when pressure and uptake series can be paired unambiguously.",
                "Convert the source to atomic format if validation cannot establish a safe interpretation.",
            ],
        )

    # -------------------------------------------------------------------------
    def validate(
        self,
        payload: bytes,
        filename: str | None,
        mapping: ImportMapping,
        custom_units: Sequence[CustomUnit] = (),
    ) -> ValidationBundle:
        catalog = UnitCatalog(
            [
                *custom_units,
                *(
                    CustomUnit(item.quantity, item.symbol, item.factor_to_canonical)
                    for item in mapping.unit_definitions
                ),
            ]
        )
        frame = read_tabular(
            payload,
            filename,
            allowed_extensions=self.allowed_extensions,
            header_row=mapping.header_row,
            field_delimiter=mapping.field_delimiter,
            encoding=mapping.encoding,
            worksheet=mapping.worksheet,
        )
        roles = {
            role: column
            for column, role in mapping.column_roles.items()
            if role not in {"ignore", "metadata"}
        }
        species_unit: str | None = None
        issues: list[ImportIssue] = []
        if mapping.species_columns:
            missing = [
                column
                for item in mapping.species_columns
                for column in (item.uptake_column, item.mole_fraction_column)
                if column and column not in frame.columns
            ]
            if missing:
                return self._bundle(
                    mapping,
                    payload,
                    [],
                    [
                        ImportIssue(
                            code="unknown_mapping_column",
                            severity="error",
                            column=column,
                            message=f"Species column '{column}' does not exist in the uploaded file.",
                        )
                        for column in missing
                    ],
                )
            frame, species_unit, issues = self._melt_species_columns(
                frame, mapping, roles
            )
            roles = {
                **roles,
                "uptake": SPECIES_UPTAKE_COLUMN,
                "adsorbate": SPECIES_ADSORBATE_COLUMN,
            }
            if any(item.mole_fraction_column for item in mapping.species_columns):
                roles["gas_mole_fraction"] = SPECIES_FRACTION_COLUMN
        issues.extend(self._mapping_issues(mapping, frame, roles))
        issues.extend(self._unit_selection_issues(mapping, roles, catalog))
        if issues:
            return self._bundle(mapping, payload, [], issues)

        detected = {
            item.name: item
            for item in [infer_column(name, frame[name]) for name in frame]
        }
        if species_unit:
            detected[SPECIES_UPTAKE_COLUMN] = detected[
                SPECIES_UPTAKE_COLUMN
            ].model_copy(update={"detected_unit": species_unit})
        issues.extend(self._unit_override_warnings(mapping, roles, detected, catalog))
        grouped_records, metadata_by_group = self._group_rows(
            frame, roles, mapping, detected, catalog, issues
        )
        experiments = self._build_experiments(
            grouped_records, metadata_by_group, mapping, issues
        )
        return self._bundle(mapping, payload, experiments, issues)

    # -------------------------------------------------------------------------
    @staticmethod
    def _melt_species_columns(
        frame: pd.DataFrame, mapping: ImportMapping, roles: dict[str, str]
    ) -> tuple[pd.DataFrame, str | None, list[ImportIssue]]:
        """Convert one-uptake-column-per-species data to the canonical long form.

        Each source row yields one row per species with a value; the original
        row index is kept so reported source rows still match the file.
        """
        issues: list[ImportIssue] = []
        unit: str | None = None
        unit_selected = (
            mapping.unit_overrides.get("uptake")
            or "uptake_unit" in roles
            or mapping.constants.get("uptake_unit")
        )
        if not unit_selected:
            units = {
                detect_header_unit(item.uptake_column, "uptake")
                for item in mapping.species_columns
            }
            if len(units) > 1:
                issues.append(
                    ImportIssue(
                        code="species_unit_mismatch",
                        severity="error",
                        message="The species uptake columns declare different units.",
                        remediation="Select one uptake unit for all species or convert the source columns.",
                    )
                )
            else:
                unit = units.pop()
        parts = []
        for item in mapping.species_columns:
            cells = frame[item.uptake_column]
            present = cells.notna() & cells.astype(str).str.strip().ne("")
            parts.append(
                frame[present].assign(
                    **{
                        SPECIES_UPTAKE_COLUMN: cells[present],
                        SPECIES_ADSORBATE_COLUMN: item.adsorbate.strip(),
                        SPECIES_FRACTION_COLUMN: (
                            frame.loc[present, item.mole_fraction_column]
                            if item.mole_fraction_column
                            else None
                        ),
                    }
                )
            )
        return pd.concat(parts).sort_index(kind="stable"), unit, issues

    # -------------------------------------------------------------------------
    @staticmethod
    def _mapping_issues(
        mapping: ImportMapping, frame: pd.DataFrame, roles: dict[str, str]
    ) -> list[ImportIssue]:
        issues: list[ImportIssue] = [
            ImportIssue(
                code="unknown_mapping_column",
                severity="error",
                column=column,
                message=f"Mapped column '{column}' does not exist in the uploaded file.",
            )
            for column in sorted(set(mapping.column_roles) - set(frame.columns))
        ]
        for required in ("pressure", "uptake", "temperature", "adsorbate", "adsorbent"):
            if required not in roles and required not in mapping.constants:
                issues.append(
                    ImportIssue(
                        code=f"missing_{required}",
                        severity="error",
                        message=f"A {required.replace('_', ' ')} column or constant is required.",
                    )
                )
        if not mapping.grouping_columns and not mapping.whole_file_grouping:
            issues.append(
                ImportIssue(
                    code="missing_grouping_column",
                    severity="error",
                    message="Select one or more grouping columns, or explicitly choose the entire file as one experiment.",
                )
            )
        for column in mapping.grouping_columns:
            if column not in frame.columns:
                issues.append(
                    ImportIssue(
                        code="missing_grouping_column",
                        severity="error",
                        column=column,
                        message=f"Grouping column '{column}' does not exist.",
                    )
                )
        return issues

    # -------------------------------------------------------------------------
    def _group_rows(
        self,
        frame: pd.DataFrame,
        roles: dict[str, str],
        mapping: ImportMapping,
        detected: dict[str, Any],
        catalog: UnitCatalog,
        issues: list[ImportIssue],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
        """Normalize rows into per-experiment records, appending row issues."""
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        metadata_by_group: dict[str, dict[str, Any]] = {}

        for row_offset, row in frame.iterrows():
            source_row = int(row_offset) + 2
            group_values = [
                str(row.get(column, "")).strip() for column in mapping.grouping_columns
            ]
            if mapping.whole_file_grouping:
                group_values = [
                    mapping.constants.get("experiment_name", mapping.dataset_name)
                ]
            if any(
                not value or value.casefold() in {"nan", "none"}
                for value in group_values
            ):
                issues.append(
                    ImportIssue(
                        code="missing_experiment_identifier",
                        severity="error",
                        source_row=source_row,
                        message="Experiment grouping values must not be empty.",
                    )
                )
                continue
            group_key = " | ".join(group_values)
            try:
                row_records, row_metadata = self._normalize_row(
                    row,
                    source_row,
                    roles,
                    mapping,
                    detected,
                    catalog,
                )
            except (ValueError, UnitConversionError) as exc:
                issues.append(
                    ImportIssue(
                        code="invalid_row",
                        severity="error",
                        source_row=source_row,
                        experiment=group_key,
                        message=str(exc),
                    )
                )
                continue
            existing = metadata_by_group.get(group_key)
            if existing is None:
                existing = metadata_by_group[group_key] = {**row_metadata, "species": {}}
            else:
                for key in ("adsorbent", "temperature_k", "pressure_basis"):
                    if existing[key] != row_metadata[key]:
                        issues.append(
                            ImportIssue(
                                code="inconsistent_experiment_metadata",
                                severity="error",
                                source_row=source_row,
                                experiment=group_key,
                                message=(
                                    f"Experiment '{group_key}' contains inconsistent {key.replace('_', ' ')} values."
                                ),
                                remediation="Correct the source data or include the differing field in the grouping selection.",
                            )
                        )
            # One experiment may hold several species; the first spelling of a
            # name wins, but conflicting SMILES or molar mass values are errors.
            species = existing["species"].setdefault(
                row_metadata["adsorbate"].casefold(),
                {
                    "name": row_metadata["adsorbate"],
                    "molar_mass_g_mol": row_metadata.get("adsorbate_molar_mass_g_mol"),
                    "smiles": row_metadata.get("adsorbate_smiles"),
                },
            )
            for field, label, key in (
                ("smiles", "SMILES", "adsorbate_smiles"),
                ("molar_mass_g_mol", "molar mass", "adsorbate_molar_mass_g_mol"),
            ):
                kept, incoming = species[field], row_metadata.get(key)
                if kept in (None, "") or incoming in (None, "") or kept == incoming:
                    continue
                issues.append(
                    ImportIssue(
                        code="conflicting_species_metadata",
                        severity="error",
                        source_row=source_row,
                        experiment=group_key,
                        message=(
                            f"Species '{species['name']}' in experiment '{group_key}' "
                            f"has conflicting {label} values: '{kept}' and '{incoming}'."
                        ),
                        remediation="Use one consistent value per species within an experiment.",
                    )
                )
            grouped_records.setdefault(group_key, []).extend(row_records)
        return grouped_records, metadata_by_group

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_experiments(
        grouped_records: dict[str, list[dict[str, Any]]],
        metadata_by_group: dict[str, dict[str, Any]],
        mapping: ImportMapping,
        issues: list[ImportIssue],
    ) -> list[dict[str, Any]]:
        """Assemble canonical experiments through the shared normalization gate."""
        experiments: list[dict[str, Any]] = []
        for group_key, observations in grouped_records.items():
            metadata = metadata_by_group.get(group_key)
            if metadata is None:
                continue
            record = {
                "external_key": group_key,
                "name": metadata["name"],
                "adsorbent": {"name": metadata["adsorbent"]},
                "adsorbates": list(metadata["species"].values()),
                "temperature_original": metadata["temperature_original"],
                "temperature_original_unit": metadata["temperature_original_unit"],
                "temperature_k": metadata["temperature_k"],
                "pressure_basis": metadata["pressure_basis"],
                "saturation_pressure_pa": metadata.get("saturation_pressure_pa"),
                "conditions": metadata.get("conditions", {}),
                "provenance": {"source_rows": metadata.get("source_rows", [])},
                "observations": observations,
            }
            try:
                experiment = normalize_experiment(record, mapping.duplicate_policy)
            except CanonicalizationError as exc:
                issues.append(
                    ImportIssue(
                        code=exc.code,
                        severity="error",
                        experiment=group_key,
                        message=str(exc),
                        remediation=(
                            "Choose the average-replicates policy or correct the duplicate rows."
                            if exc.code == "duplicate_measurement"
                            else None
                        ),
                    )
                )
                continue
            if experiment.provenance.get("duplicates") == "averaged":
                issues.append(
                    ImportIssue(
                        code="duplicates_averaged",
                        severity="warning",
                        experiment=group_key,
                        message=(
                            f"Repeated pressures in '{group_key}' were averaged into one "
                            "measurement each; the replicate values are kept in its metadata."
                        ),
                    )
                )
            experiments.append(experiment.model_dump())
        return experiments

    # -------------------------------------------------------------------------
    @staticmethod
    def _mapped_value(
        row: pd.Series,
        role: str,
        roles: dict[str, str],
        mapping: ImportMapping,
    ) -> Any:
        column = roles.get(role)
        if column is not None:
            return row.get(column)
        return mapping.constants.get(role)

    # -------------------------------------------------------------------------
    def _saturation_pressure_pa(
        self,
        row: pd.Series,
        roles: dict[str, str],
        mapping: ImportMapping,
        catalog: UnitCatalog,
        pressure_unit: str,
    ) -> float | None:
        raw = self._mapped_value(row, "saturation_pressure", roles, mapping)
        if raw in (None, ""):
            return None
        saturation_unit = mapping.unit_overrides.get(
            "saturation_pressure", pressure_unit
        )
        return catalog.convert_pressure(
            parse_number(raw, mapping.decimal_separator), saturation_unit, "absolute"
        ).canonical_value

    # -------------------------------------------------------------------------
    def _uptake_stddev(
        self,
        row: pd.Series,
        roles: dict[str, str],
        mapping: ImportMapping,
        catalog: UnitCatalog,
        uptake_unit: str,
        molar_mass: float | None,
    ) -> float | None:
        raw = self._mapped_value(row, "uptake_stddev", roles, mapping)
        if raw in (None, ""):
            return None
        stddev = catalog.convert_uptake(
            parse_number(raw, mapping.decimal_separator), uptake_unit, molar_mass
        ).canonical_value
        if stddev <= 0:
            raise ValueError("Uptake uncertainty must be positive.")
        return stddev

    # -------------------------------------------------------------------------
    def _observation_pairs(
        self, row: pd.Series, roles: dict[str, str], mapping: ImportMapping
    ) -> list[tuple[float, float]]:
        """Return (pressure, uptake) pairs from wide columns or series cells."""
        if mapping.wide_pairs:
            return [
                (
                    parse_number(row.get(pair.pressure_column), mapping.decimal_separator),
                    parse_number(row.get(pair.uptake_column), mapping.decimal_separator),
                )
                for pair in mapping.wide_pairs
            ]
        pressure_values = parse_series_cell(
            self._mapped_value(row, "pressure", roles, mapping),
            delimiter=mapping.series_delimiter,
            decimal_separator=mapping.decimal_separator,
        )
        uptake_values = parse_series_cell(
            self._mapped_value(row, "uptake", roles, mapping),
            delimiter=mapping.series_delimiter,
            decimal_separator=mapping.decimal_separator,
        )
        if len(pressure_values) != len(uptake_values):
            raise ValueError(
                "Pressure and uptake series have unequal lengths "
                f"({len(pressure_values)} and {len(uptake_values)})."
            )
        return list(zip(pressure_values, uptake_values, strict=True))

    # -------------------------------------------------------------------------
    def _normalize_row(
        self,
        row: pd.Series,
        source_row: int,
        roles: dict[str, str],
        mapping: ImportMapping,
        detected: dict[str, ColumnDetection],
        catalog: UnitCatalog,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        adsorbent = str(
            self._mapped_value(row, "adsorbent", roles, mapping) or ""
        ).strip()
        adsorbate = str(
            self._mapped_value(row, "adsorbate", roles, mapping) or ""
        ).strip()
        adsorbate_smiles_value = safe_cell(
            self._mapped_value(row, "adsorbate_smiles", roles, mapping)
        )
        adsorbate_smiles = (
            str(adsorbate_smiles_value).strip()
            if adsorbate_smiles_value is not None
            else None
        )
        if adsorbate_smiles and adsorbate_smiles.casefold() in {"nan", "none"}:
            adsorbate_smiles = None
        if not adsorbent or adsorbent.casefold() in {"nan", "none"}:
            raise ValueError("Adsorbent material is missing.")
        if not adsorbate or adsorbate.casefold() in {"nan", "none"}:
            raise ValueError("Adsorbate species is missing.")

        temperature_value = parse_number(
            self._mapped_value(row, "temperature", roles, mapping),
            mapping.decimal_separator,
        )
        temperature_unit = self._unit_for("temperature", row, roles, mapping, detected)
        temperature = catalog.convert_temperature(
            temperature_value, temperature_unit
        )

        pressure_unit = self._unit_for("pressure", row, roles, mapping, detected)
        uptake_unit = self._unit_for("uptake", row, roles, mapping, detected)
        molar_mass = mapping.constants.get("adsorbate_molar_mass_g_mol")
        if molar_mass is None:
            molar_mass = KNOWN_MOLAR_MASSES_G_MOL.get(adsorbate.casefold())
        molar_mass_value = float(molar_mass) if molar_mass not in (None, "") else None
        name = str(
            self._mapped_value(row, "experiment_name", roles, mapping) or ""
        ).strip()
        if not name or name.casefold() in {"nan", "none"}:
            name = " | ".join(
                str(row.get(column, "")).strip() for column in mapping.grouping_columns
            )

        saturation_pressure_pa = self._saturation_pressure_pa(
            row, roles, mapping, catalog, pressure_unit
        )
        pairs = self._observation_pairs(row, roles, mapping)
        fraction_raw = safe_cell(
            self._mapped_value(row, "gas_mole_fraction", roles, mapping)
        )
        gas_mole_fraction = (
            parse_number(fraction_raw, mapping.decimal_separator)
            if fraction_raw not in (None, "")
            else None
        )
        pressure_raw = self._mapped_value(row, "pressure", roles, mapping)
        uptake_raw = self._mapped_value(row, "uptake", roles, mapping)

        metadata_columns = [
            column
            for column, role in mapping.column_roles.items()
            if role == "metadata"
        ]
        observations: list[dict[str, Any]] = []
        for pressure_value, uptake_value in pairs:
            pressure = catalog.convert_pressure(
                pressure_value, pressure_unit, mapping.pressure_basis
            )
            uptake = catalog.convert_uptake(
                uptake_value, uptake_unit, molar_mass_value
            )
            stddev = self._uptake_stddev(
                row, roles, mapping, catalog, uptake_unit, molar_mass_value
            )
            observations.append(
                {
                    "adsorbate": adsorbate,
                    "source_row": source_row,
                    "pressure_original": pressure.original_value,
                    "pressure_original_unit": pressure.original_unit,
                    "pressure_canonical": pressure.canonical_value,
                    "pressure_canonical_unit": pressure.canonical_unit,
                    "uptake_original": uptake.original_value,
                    "uptake_original_unit": uptake.original_unit,
                    "uptake_mol_kg": uptake.canonical_value,
                    "uptake_stddev_mol_kg": stddev,
                    "gas_mole_fraction": gas_mole_fraction,
                    "conversion_metadata": {
                        "pressure_rule": pressure.rule,
                        "uptake_rule": uptake.rule,
                    },
                    "extra_metadata": {
                        "source_pressure_token": safe_cell(pressure_raw),
                        "source_uptake_token": safe_cell(uptake_raw),
                        **{
                            column: safe_cell(row.get(column))
                            for column in metadata_columns
                        },
                    },
                }
            )
        return observations, {
            "name": name,
            "adsorbent": adsorbent,
            "adsorbate": adsorbate,
            "adsorbate_smiles": adsorbate_smiles,
            "adsorbate_molar_mass_g_mol": molar_mass_value,
            "temperature_original": temperature.original_value,
            "temperature_original_unit": temperature.original_unit,
            "temperature_k": temperature.canonical_value,
            "pressure_basis": mapping.pressure_basis,
            "saturation_pressure_pa": saturation_pressure_pa,
            "source_rows": [source_row],
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _unit_selection_issues(
        mapping: ImportMapping, roles: dict[str, str], catalog: UnitCatalog
    ) -> list[ImportIssue]:
        """Reject explicitly selected units once, before any row is converted."""
        selected: dict[str, str] = {}
        for quantity in UNIT_QUANTITIES:
            if mapping.unit_overrides.get(quantity):
                selected[quantity] = mapping.unit_overrides[quantity]
            elif f"{quantity}_unit" not in roles and mapping.constants.get(
                f"{quantity}_unit"
            ):
                selected[quantity] = str(mapping.constants[f"{quantity}_unit"])
        issues: list[ImportIssue] = []
        for quantity, unit in selected.items():
            try:
                catalog.check(quantity, unit)
            except UnitConversionError as exc:
                issues.append(
                    ImportIssue(
                        code="unsupported_unit",
                        severity="error",
                        column=quantity,
                        message=str(exc),
                        remediation=(
                            "Pick a unit from the catalog, define its conversion factor, "
                            "or convert the source values before importing."
                        ),
                    )
                )
        return issues

    # -------------------------------------------------------------------------
    @staticmethod
    def _unit_override_warnings(
        mapping: ImportMapping,
        roles: dict[str, str],
        detected: dict[str, ColumnDetection],
        catalog: UnitCatalog,
    ) -> list[ImportIssue]:
        warnings: list[ImportIssue] = []
        for quantity in UNIT_QUANTITIES:
            override = mapping.unit_overrides.get(quantity)
            column = roles.get(quantity)
            found = detected[column].detected_unit if column in detected else None
            if not override or not found:
                continue
            if catalog.check(quantity, override) != catalog.check(quantity, found):
                warnings.append(
                    ImportIssue(
                        code="unit_override_applied",
                        severity="warning",
                        column=column,
                        message=(
                            f"The detected {quantity} unit '{found}' was replaced by "
                            f"the manual override '{override}'."
                        ),
                    )
                )
        return warnings

    # -------------------------------------------------------------------------
    @staticmethod
    def _unit_for(
        quantity: str,
        row: pd.Series,
        roles: dict[str, str],
        mapping: ImportMapping,
        detected: dict[str, ColumnDetection],
    ) -> str:
        override = mapping.unit_overrides.get(quantity)
        if override:
            return override
        unit_role = f"{quantity}_unit"
        unit_column = roles.get(unit_role)
        if unit_column:
            value = row.get(unit_column)
            if value is not None and str(value).strip():
                return str(value).strip()
            raise UnitConversionError(
                f"The {quantity.replace('_', ' ')} unit column is empty for this observation."
            )
        value_column = roles.get(quantity)
        if value_column and detected[value_column].detected_unit:
            return str(detected[value_column].detected_unit)
        constant = mapping.constants.get(unit_role)
        if constant:
            return str(constant)
        raise UnitConversionError(
            f"The {quantity.replace('_', ' ')} unit is unknown; choose it explicitly."
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _bundle(
        mapping: ImportMapping,
        payload: bytes,
        experiments: list[dict[str, Any]],
        issues: list[ImportIssue],
    ) -> ValidationBundle:
        errors = [issue for issue in issues if issue.severity == "error"]
        confirmations = [
            issue
            for issue in issues
            if issue.severity == "confirmation"
            and issue.code not in mapping.confirmed_issue_codes
        ]
        status = (
            "invalid"
            if errors
            else "confirmation_required"
            if confirmations
            else "valid"
        )
        previews = [
            NormalizedExperimentPreview(
                external_key=experiment["external_key"],
                name=experiment["name"],
                adsorbent=experiment["adsorbent"]["name"],
                adsorbates=[item["name"] for item in experiment["adsorbates"]],
                temperature_k=experiment["temperature_k"],
                temperature_original=experiment["temperature_original"],
                temperature_original_unit=experiment["temperature_original_unit"],
                pressure_basis=experiment["pressure_basis"],
                observation_count=len(experiment["observations"]),
                observations=[
                    NormalizedObservationPreview(
                        **{
                            key: observation[key]
                            for key in (
                                "adsorbate",
                                "source_row",
                                "pressure_original",
                                "pressure_original_unit",
                                "pressure_canonical",
                                "pressure_canonical_unit",
                                "uptake_original",
                                "uptake_original_unit",
                                "uptake_mol_kg",
                                "gas_mole_fraction",
                            )
                        }
                    )
                    for observation in experiment["observations"][
                        :EXPERIMENT_PREVIEW_POINTS
                    ]
                ],
            )
            for experiment in experiments[:20]
        ]
        response = ImportValidationResponse(
            status=status,  # type: ignore[arg-type]
            source_sha256=source_hash(payload),
            structure=mapping.structure,
            experiment_count=len(experiments),
            observation_count=sum(
                len(experiment["observations"]) for experiment in experiments
            ),
            experiments=previews,
            issues=issues,
        )
        return ValidationBundle(response=response, experiments=experiments)
