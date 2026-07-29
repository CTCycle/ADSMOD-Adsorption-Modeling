from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core_service.domain.datasets import (
    ColumnDetection,
    ImportIssue,
    ImportMapping,
    ImportPreviewResponse,
    ImportValidationResponse,
    NormalizedExperimentPreview,
    NormalizedObservationPreview,
)
from core_service.services.data.units import (
    UnitConversionError,
    UnitRegistry,
    detect_header_unit,
    normalize_token,
    parse_number,
)


PARSER_VERSION = "2.0"
PREVIEW_ROWS = 12
EXPERIMENT_PREVIEW_POINTS = 8


ROLE_ALIASES: dict[str, set[str]] = {
    "experiment_id": {
        "experiment",
        "experiment id",
        "experiment_id",
        "isotherm",
        "isotherm id",
        "isotherm_id",
        "exp id",
        "id esperimento",
        "esperimento",
        "id experimento",
        "experimento",
        "id experience",
        "experience",
        "versuch id",
        "versuch",
        "experiencia id",
    },
    "experiment_name": {
        "experiment name",
        "experiment_name",
        "isotherm name",
        "nome esperimento",
        "nombre experimento",
        "nom experience",
        "versuchsname",
        "nome experimento",
    },
    "pressure": {
        "pressure",
        "equilibrium pressure",
        "absolute pressure",
        "partial pressure",
        "relative pressure",
        "p",
        "peq",
        "p/p0",
        "pressione",
        "pressione equilibrio",
        "presion",
        "presion equilibrio",
        "pression",
        "pression equilibre",
        "druck",
        "gleichgewichtsdruck",
        "pressao",
    },
    "uptake": {
        "uptake",
        "adsorption",
        "adsorbed amount",
        "adsorbed quantity",
        "loading",
        "adsorption capacity",
        "capacity",
        "q",
        "qe",
        "quantita adsorbita",
        "carico",
        "cantidad adsorbida",
        "carga",
        "quantite adsorbee",
        "chargement",
        "beladung",
        "adsorbierte menge",
        "quantidade adsorvida",
    },
    "adsorbate": {
        "adsorbate",
        "adsorbate species",
        "species",
        "gas",
        "guest",
        "sorbate",
        "adsorbato",
        "gas adsorbito",
        "especie",
        "adsorbat",
        "gast",
        "adsorvato",
    },
    "adsorbent": {
        "adsorbent",
        "material",
        "adsorbent material",
        "host",
        "sorbent",
        "adsorbente",
        "materiale",
        "materiau",
        "adsorbens",
        "werkstoff",
        "material adsorvente",
    },
    "temperature": {
        "temperature",
        "temp",
        "t",
        "temperatura",
        "temperature equilibrium",
        "temperatur",
    },
    "pressure_unit": {
        "pressure unit",
        "pressure units",
        "unita pressione",
        "unidad presion",
        "unite pression",
        "druckeinheit",
    },
    "uptake_unit": {
        "uptake unit",
        "adsorption unit",
        "adsorption units",
        "unita adsorbimento",
        "unidad adsorcion",
        "unite adsorption",
        "beladungseinheit",
    },
    "temperature_unit": {
        "temperature unit",
        "unita temperatura",
        "unidad temperatura",
        "unite temperature",
        "temperatureinheit",
    },
    "uptake_stddev": {
        "uptake stddev",
        "uptake standard deviation",
        "adsorption uncertainty",
        "sigma q",
        "q std",
    },
    "saturation_pressure": {
        "saturation pressure",
        "vapor pressure",
        "vapour pressure",
        "p0",
        "pressione saturazione",
        "presion saturacion",
        "pression saturation",
        "sattigungsdruck",
    },
}


def normalize_header(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[\[\](){}]", " ", text)
    text = re.sub(r"[^a-z0-9/°%]+", " ", text)
    return " ".join(text.split())


def safe_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def source_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_tabular(payload: bytes, filename: str | None, *, header_row: int = 0, field_delimiter: str | None = None, encoding: str = "utf-8", worksheet: str | int | None = None) -> pd.DataFrame:
    if not payload:
        raise ValueError("Uploaded dataset is empty.")
    suffix = Path(filename or "").suffix.casefold()
    buffer = io.BytesIO(payload)
    try:
        if suffix in {".xls", ".xlsx"}:
            frame = pd.read_excel(buffer, sheet_name=worksheet if worksheet is not None else 0, header=header_row, dtype=object)
        elif suffix == ".json":
            decoded = json.loads(payload.decode(encoding if encoding else "utf-8-sig"))
            if isinstance(decoded, dict):
                decoded = decoded.get("records", decoded.get("data"))
            if not isinstance(decoded, list) or not all(
                isinstance(row, dict) for row in decoded
            ):
                raise ValueError(
                    "JSON datasets must contain an array of row objects."
                )
            frame = pd.DataFrame.from_records(decoded)
        elif suffix in {".csv", ".txt", ".tsv", ""}:
            frame = pd.read_csv(
                io.StringIO(payload.decode(encoding)),
                sep=field_delimiter or None,
                engine="python",
                header=header_row,
                dtype=object,
                comment="#",
                keep_default_na=True,
            )
        else:
            raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'.")
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to parse '{filename or 'dataset'}': {exc}") from exc
    if frame.empty:
        raise ValueError("Uploaded dataset contains no data rows.")
    if len(frame.columns) > 256:
        raise ValueError("Uploaded dataset contains more than 256 columns.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("Every uploaded column must have a non-empty header.")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Uploaded dataset contains duplicate column headers.")
    return frame


def parse_series_cell(
    value: Any,
    *,
    delimiter: str | None,
    decimal_separator: str,
) -> list[float]:
    if isinstance(value, (list, tuple)):
        raw = list(value)
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    elif isinstance(value, (int, float)):
        return [parse_number(value, decimal_separator)]
    else:
        text = str(value).strip()
        if not text:
            return []
        raw: Any = None
        if text.startswith("[") and text.endswith("]"):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(text)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"Malformed serialized array '{text[:80]}'.") from exc
            if not isinstance(raw, (list, tuple)):
                raise ValueError("Serialized measurement cells must contain an array.")
            raw = list(raw)
        else:
            selected = delimiter
            if selected is None:
                candidates = [
                    token for token in (";", "|", "\t") if token in text
                ]
                if not candidates:
                    return [parse_number(text, decimal_separator)]
                if len(candidates) != 1:
                    raise ValueError(
                        "Delimited series require an explicit, unambiguous delimiter."
                    )
                selected = candidates[0]
            raw = [item.strip() for item in text.split(selected)]
    return [parse_number(item, decimal_separator) for item in raw]


def is_array_like(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    return (
        (text.startswith("[") and text.endswith("]"))
        or sum(token in text for token in (";", "|", "\t")) == 1
    )


def infer_column(column: str, series: pd.Series) -> ColumnDetection:
    header = normalize_header(column)
    non_empty = [safe_cell(value) for value in series.dropna().head(20).tolist()]
    array_ratio = (
        sum(is_array_like(value) for value in non_empty) / len(non_empty)
        if non_empty
        else 0.0
    )
    numeric_count = 0
    for value in non_empty:
        try:
            parse_number(value, ".")
            numeric_count += 1
        except ValueError:
            pass
    numeric_ratio = numeric_count / len(non_empty) if non_empty else 0.0

    scored: list[tuple[float, str, list[str]]] = []
    header_tokens = set(header.split())
    for role, aliases in ROLE_ALIASES.items():
        best = 0.0
        evidence: list[str] = []
        for alias in aliases:
            normalized_alias = normalize_header(alias)
            alias_tokens = set(normalized_alias.split())
            if header == normalized_alias:
                best = max(best, 0.98)
                evidence = [f"header exactly matches '{alias}'"]
            elif alias_tokens and alias_tokens.issubset(header_tokens):
                score = 0.82 if len(alias_tokens) > 1 else 0.72
                if score > best:
                    best = score
                    evidence = [f"header contains scientific alias '{alias}'"]
        if role in {"pressure", "uptake", "temperature", "uptake_stddev"}:
            if numeric_ratio >= 0.8:
                best += 0.08
                evidence.append("representative values are numerical")
            if array_ratio >= 0.5 and role in {"pressure", "uptake"}:
                best += 0.05
                evidence.append("representative values contain numerical series")
        if role == "experiment_id":
            cardinality = series.nunique(dropna=True)
            if 0 < cardinality < max(2, len(series)):
                best += 0.05
                evidence.append("values repeat and can define groups")
        if best:
            scored.append((min(best, 1.0), role, evidence))
    scored.sort(reverse=True)
    confidence, role, evidence = scored[0] if scored else (0.0, "ignore", [])

    detected_unit = None
    quantity = {
        "pressure": "pressure",
        "uptake": "uptake",
        "temperature": "temperature",
    }.get(role)
    if quantity:
        detected_unit = detect_header_unit(column, quantity)
        if detected_unit:
            confidence = min(1.0, confidence + 0.08)
            evidence.append(f"header declares unit '{detected_unit}'")

    inferred_type = (
        "series"
        if array_ratio >= 0.5
        else "number"
        if numeric_ratio >= 0.8
        else "text"
    )
    return ColumnDetection(
        name=column,
        inferred_type=inferred_type,
        sample_values=non_empty[:5],
        proposed_role=role,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        evidence=evidence,
        detected_unit=detected_unit,
        array_like=array_ratio >= 0.5,
    )


def detect_structure(columns: list[ColumnDetection]) -> tuple[str, float]:
    pressure = next(
        (column for column in columns if column.proposed_role == "pressure"), None
    )
    uptake = next(
        (column for column in columns if column.proposed_role == "uptake"), None
    )
    wide_pressure = [
        column
        for column in columns
        if re.search(r"(?:pressure|\bp)\s*\d+$", normalize_header(column.name))
    ]
    wide_uptake = [
        column
        for column in columns
        if re.search(r"(?:uptake|\bq)\s*\d+$", normalize_header(column.name))
    ]
    if wide_pressure and len(wide_pressure) == len(wide_uptake):
        return "aggregated", 0.9
    if pressure and uptake:
        if pressure.array_like and uptake.array_like:
            return "aggregated", 0.95
        if pressure.array_like != uptake.array_like:
            return "mixed", 0.75
        return "atomic", 0.95
    return "ambiguous", 0.25


def infer_pressure_basis(columns: list[ColumnDetection]) -> str | None:
    column = next(
        (item for item in columns if item.proposed_role == "pressure"), None
    )
    if column is None:
        return None
    header = normalize_header(column.name)
    if "relative" in header or "p/p0" in header or column.detected_unit == "1":
        return "relative"
    if "partial" in header:
        return "partial"
    if "absolute" in header or column.detected_unit not in {None, "1", "%"}:
        return "absolute"
    return None


@dataclass
class ValidationBundle:
    response: ImportValidationResponse
    experiments: list[dict[str, Any]]


class AdsorptionImportEngine:
    def preview(self, payload: bytes, filename: str | None) -> ImportPreviewResponse:
        frame = read_tabular(payload, filename)
        columns = [infer_column(name, frame[name]) for name in frame.columns]
        structure, confidence = detect_structure(columns)
        grouping = [
            column.name
            for column in columns
            if column.proposed_role == "experiment_id" and column.confidence >= 0.65
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
            matches = [
                column
                for column in columns
                if column.proposed_role == role and column.confidence >= 0.55
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
                for row in frame.head(PREVIEW_ROWS).to_dict(orient="records")
            ],
            proposed_grouping_columns=grouping,
            proposed_pressure_basis=infer_pressure_basis(columns),  # type: ignore[arg-type]
            issues=issues,
            guidance=[
                "Atomic one-observation-per-row data is the recommended format.",
                "Aggregated formats are accepted only when pressure and uptake series can be paired unambiguously.",
                "Convert the source to atomic format if validation cannot establish a safe interpretation.",
            ],
        )

    def validate(
        self, payload: bytes, filename: str | None, mapping: ImportMapping
    ) -> ValidationBundle:
        frame = read_tabular(payload, filename, header_row=mapping.header_row, field_delimiter=mapping.field_delimiter, encoding=mapping.encoding, worksheet=mapping.worksheet)
        issues: list[ImportIssue] = []
        unknown_columns = sorted(set(mapping.column_roles) - set(frame.columns))
        if unknown_columns:
            issues.extend(
                ImportIssue(
                    code="unknown_mapping_column",
                    severity="error",
                    column=column,
                    message=f"Mapped column '{column}' does not exist in the uploaded file.",
                )
                for column in unknown_columns
            )

        roles = {
            role: column
            for column, role in mapping.column_roles.items()
            if role not in {"ignore", "metadata"}
        }
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
            issues.append(ImportIssue(code="missing_grouping_column", severity="error", message="Select one or more grouping columns, or explicitly choose the entire file as one experiment."))
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
        if issues:
            return self._bundle(mapping, payload, [], issues)

        detected = {
            item.name: item for item in [infer_column(name, frame[name]) for name in frame]
        }
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        metadata_by_group: dict[str, dict[str, Any]] = {}

        for row_offset, row in frame.iterrows():
            source_row = int(row_offset) + 2
            group_values = [str(row.get(column, "")).strip() for column in mapping.grouping_columns]
            if mapping.whole_file_grouping:
                group_values = [mapping.constants.get("experiment_name", mapping.dataset_name)]
            if any(not value or value.casefold() in {"nan", "none"} for value in group_values):
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
            if existing is not None:
                for key in (
                    "adsorbent",
                    "adsorbate",
                    "temperature_k",
                    "pressure_basis",
                ):
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
            else:
                metadata_by_group[group_key] = row_metadata
            grouped_records.setdefault(group_key, []).extend(row_records)

        experiments: list[dict[str, Any]] = []
        for group_key, observations in grouped_records.items():
            if group_key not in metadata_by_group:
                continue
            metadata = metadata_by_group[group_key]
            observations.sort(
                key=lambda item: (
                    item["pressure_canonical"],
                    item["source_row"] or 0,
                    item["sequence_index"],
                )
            )
            duplicate_pressures: dict[float, list[dict[str, Any]]] = {}
            for observation in observations:
                duplicate_pressures.setdefault(
                    observation["pressure_canonical"], []
                ).append(observation)
            repeated = {
                pressure: items
                for pressure, items in duplicate_pressures.items()
                if len(items) > 1
            }
            if repeated and mapping.duplicate_policy == "reject":
                issues.append(
                    ImportIssue(
                        code="duplicate_pressure",
                        severity="confirmation",
                        experiment=group_key,
                        message=(
                            f"Experiment '{group_key}' has repeated pressure values. "
                            "Keeping them weights those pressures as replicate observations."
                        ),
                        remediation="Choose the keep-replicates policy or correct duplicate rows.",
                    )
                )
            elif repeated and mapping.duplicate_policy == "average":
                issues.append(
                    ImportIssue(
                        code="duplicates_average_on_fit",
                        severity="warning",
                        experiment=group_key,
                        message=f"Repeated pressures in '{group_key}' will be averaged deterministically for fitting; all source observations are retained.",
                    )
                )
            elif repeated:
                issues.append(
                    ImportIssue(
                        code="duplicates_kept",
                        severity="warning",
                        experiment=group_key,
                        message=f"Repeated pressures in '{group_key}' were retained as replicates.",
                    )
                )

            if len(observations) < 2:
                issues.append(
                    ImportIssue(
                        code="insufficient_observations",
                        severity="error",
                        experiment=group_key,
                        message="Each experiment requires at least two valid observations.",
                    )
                )
            for sequence_index, observation in enumerate(observations):
                observation["sequence_index"] = sequence_index
            experiments.append(
                {
                    "external_key": group_key,
                    "name": metadata["name"],
                    "adsorbent": {"name": metadata["adsorbent"]},
                    "adsorbates": [{"name": metadata["adsorbate"], "molar_mass_g_mol": metadata.get("adsorbate_molar_mass_g_mol")}],
                    "temperature_original": metadata["temperature_original"],
                    "temperature_original_unit": metadata[
                        "temperature_original_unit"
                    ],
                    "temperature_k": metadata["temperature_k"],
                    "pressure_basis": metadata["pressure_basis"],
                    "duplicate_policy": mapping.duplicate_policy,
                    "saturation_pressure_pa": metadata.get("saturation_pressure_pa"),
                    "conditions": metadata.get("conditions", {}),
                    "provenance": {"source_rows": metadata.get("source_rows", [])},
                    "observations": observations,
                }
            )

        return self._bundle(mapping, payload, experiments, issues)

    def _normalize_row(
        self,
        row: pd.Series,
        source_row: int,
        roles: dict[str, str],
        mapping: ImportMapping,
        detected: dict[str, ColumnDetection],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        def mapped_value(role: str) -> Any:
            column = roles.get(role)
            if column is not None:
                return row.get(column)
            return mapping.constants.get(role)

        adsorbent = str(mapped_value("adsorbent") or "").strip()
        adsorbate = str(mapped_value("adsorbate") or "").strip()
        if not adsorbent or adsorbent.casefold() in {"nan", "none"}:
            raise ValueError("Adsorbent material is missing.")
        if not adsorbate or adsorbate.casefold() in {"nan", "none"}:
            raise ValueError("Adsorbate species is missing.")

        temperature_value = parse_number(
            mapped_value("temperature"), mapping.decimal_separator
        )
        temperature_unit = self._unit_for(
            "temperature", row, roles, mapping, detected
        )
        temperature = UnitRegistry.convert_temperature(
            temperature_value, temperature_unit
        )

        pressure_unit = self._unit_for(
            "pressure", row, roles, mapping, detected
        )
        uptake_unit = self._unit_for("uptake", row, roles, mapping, detected)
        known_molar_masses = {"co2": 44.0095, "carbon dioxide": 44.0095, "n2": 28.0134, "nitrogen": 28.0134, "ch4": 16.0425, "methane": 16.0425, "o2": 31.9988, "h2": 2.01588, "hydrogen": 2.01588}
        molar_mass = mapping.constants.get("adsorbate_molar_mass_g_mol")
        if molar_mass is None:
            molar_mass = known_molar_masses.get(adsorbate.casefold())
        molar_mass_value = float(molar_mass) if molar_mass not in (None, "") else None
        name = str(mapped_value("experiment_name") or "").strip()
        if not name or name.casefold() in {"nan", "none"}:
            name = " | ".join(
                str(row.get(column, "")).strip()
                for column in mapping.grouping_columns
            )

        saturation_pressure_pa = None
        if mapped_value("saturation_pressure") not in (None, ""):
            saturation_value = parse_number(
                mapped_value("saturation_pressure"), mapping.decimal_separator
            )
            saturation_unit = mapping.unit_overrides.get(
                "saturation_pressure", pressure_unit
            )
            saturation_pressure_pa = UnitRegistry.convert_pressure(
                saturation_value, saturation_unit, "absolute"
            ).canonical_value

        pairs: list[tuple[float, float, int]] = []
        pressure_raw = mapped_value("pressure")
        uptake_raw = mapped_value("uptake")
        if mapping.wide_pairs:
            for pair_index, pair in enumerate(mapping.wide_pairs):
                pressure = parse_number(
                    row.get(pair.pressure_column), mapping.decimal_separator
                )
                uptake = parse_number(
                    row.get(pair.uptake_column), mapping.decimal_separator
                )
                pairs.append((pressure, uptake, pair_index))
        else:
            pressure_values = parse_series_cell(
                mapped_value("pressure"),
                delimiter=mapping.series_delimiter,
                decimal_separator=mapping.decimal_separator,
            )
            uptake_values = parse_series_cell(
                mapped_value("uptake"),
                delimiter=mapping.series_delimiter,
                decimal_separator=mapping.decimal_separator,
            )
            if len(pressure_values) != len(uptake_values):
                raise ValueError(
                    "Pressure and uptake series have unequal lengths "
                    f"({len(pressure_values)} and {len(uptake_values)})."
                )
            pairs = [
                (pressure, uptake, index)
                for index, (pressure, uptake) in enumerate(
                    zip(pressure_values, uptake_values, strict=True)
                )
            ]

        metadata_columns = [
            column
            for column, role in mapping.column_roles.items()
            if role == "metadata"
        ]
        observations: list[dict[str, Any]] = []
        for pressure_value, uptake_value, sequence_index in pairs:
            pressure = UnitRegistry.convert_pressure(
                pressure_value, pressure_unit, mapping.pressure_basis
            )
            uptake = UnitRegistry.convert_uptake(uptake_value, uptake_unit, molar_mass_value)
            stddev = None
            if mapped_value("uptake_stddev") not in (None, ""):
                stddev_value = parse_number(
                    mapped_value("uptake_stddev"), mapping.decimal_separator
                )
                stddev = UnitRegistry.convert_uptake(
                    stddev_value, uptake_unit, molar_mass_value
                ).canonical_value
                if stddev <= 0:
                    raise ValueError("Uptake uncertainty must be positive.")
            observations.append(
                {
                    "adsorbate": adsorbate,
                    "sequence_index": sequence_index,
                    "source_row": source_row,
                    "pressure_original": pressure.original_value,
                    "pressure_original_unit": pressure.original_unit,
                    "pressure_canonical": pressure.canonical_value,
                    "pressure_canonical_unit": pressure.canonical_unit,
                    "uptake_original": uptake.original_value,
                    "uptake_original_unit": uptake.original_unit,
                    "uptake_mol_kg": uptake.canonical_value,
                    "uptake_stddev_mol_kg": stddev,
                    "conversion_metadata": {
                        "pressure_rule": pressure.rule,
                        "uptake_rule": uptake.rule,
                    },
                    "extra_metadata": {
                        "source_pressure_token": safe_cell(pressure_raw),
                        "source_uptake_token": safe_cell(uptake_raw),
                        **{column: safe_cell(row.get(column)) for column in metadata_columns},
                    },
                }
            )
        return observations, {
            "name": name,
            "adsorbent": adsorbent,
            "adsorbate": adsorbate,
            "adsorbate_molar_mass_g_mol": molar_mass_value,
            "temperature_original": temperature.original_value,
            "temperature_original_unit": temperature.original_unit,
            "temperature_k": temperature.canonical_value,
            "pressure_basis": mapping.pressure_basis,
            "saturation_pressure_pa": saturation_pressure_pa,
            "source_rows": [source_row],
        }

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
            raise UnitConversionError(f"The {quantity.replace('_', ' ')} unit column is empty for this observation.")
        value_column = roles.get(quantity)
        if value_column and detected[value_column].detected_unit:
            return str(detected[value_column].detected_unit)
        constant = mapping.constants.get(unit_role)
        if constant:
            return str(constant)
        raise UnitConversionError(
            f"The {quantity.replace('_', ' ')} unit is unknown; choose it explicitly."
        )

    @staticmethod
    def _average_duplicates(
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[float, list[dict[str, Any]]] = {}
        for item in observations:
            grouped.setdefault(item["pressure_canonical"], []).append(item)
        averaged: list[dict[str, Any]] = []
        for pressure, items in sorted(grouped.items()):
            if len(items) == 1:
                averaged.append(items[0])
                continue
            base = dict(items[0])
            base["uptake_mol_kg"] = sum(
                item["uptake_mol_kg"] for item in items
            ) / len(items)
            base["uptake_original"] = base["uptake_mol_kg"]
            base["uptake_original_unit"] = "mol/kg"
            base["conversion_metadata"] = {
                **base["conversion_metadata"],
                "duplicate_policy": "average",
                "replicate_count": len(items),
                "source_uptake_values_mol_kg": [
                    item["uptake_mol_kg"] for item in items
                ],
            }
            averaged.append(base)
        return averaged

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
        status = "invalid" if errors else "confirmation_required" if confirmations else "valid"
        previews = [
            NormalizedExperimentPreview(
                external_key=experiment["external_key"],
                name=experiment["name"],
                adsorbent=experiment["adsorbent"]["name"],
                adsorbate=experiment["adsorbates"][0]["name"],
                temperature_k=experiment["temperature_k"],
                pressure_basis=experiment["pressure_basis"],
                observation_count=len(experiment["observations"]),
                observations=[
                    NormalizedObservationPreview(
                        **{
                            key: observation[key]
                            for key in (
                                "source_row",
                                "sequence_index",
                                "pressure_original",
                                "pressure_original_unit",
                                "pressure_canonical",
                                "pressure_canonical_unit",
                                "uptake_original",
                                "uptake_original_unit",
                                "uptake_mol_kg",
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
