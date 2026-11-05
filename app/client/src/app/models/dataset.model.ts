// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export type DatasetSource = 'uploaded' | 'nist';
export type DatasetStructure = 'atomic' | 'aggregated' | 'mixed' | 'ambiguous';
export type ImportableStructure = Exclude<DatasetStructure, 'ambiguous'>;
export type PressureBasis = 'absolute' | 'partial' | 'relative';
export type ColumnRole =
    | 'experiment_id'
    | 'experiment_name'
    | 'pressure'
    | 'uptake'
    | 'adsorbate'
    | 'adsorbate_smiles'
    | 'adsorbent'
    | 'temperature'
    | 'pressure_unit'
    | 'uptake_unit'
    | 'temperature_unit'
    | 'uptake_stddev'
    | 'saturation_pressure'
    | 'metadata'
    | 'ignore';

export type UnitQuantity = 'pressure' | 'uptake' | 'temperature';

export interface UnitOption {
    quantity: UnitQuantity;
    symbol: string;
    label: string;
    name: string;
    group: string;
    note: string;
    aliases: string[];
    built_in: boolean;
    has_conversion: boolean;
    factor_to_canonical: number | null;
    id: number | null;
}

export interface CustomUnitDefinition {
    quantity: UnitQuantity;
    symbol: string;
    factor_to_canonical: number | null;
}

export interface UnitCatalog {
    pressure: UnitOption[];
    uptake: UnitOption[];
    temperature: UnitOption[];
    allowed_extensions: string[];
}

export interface DatasetSummary {
    id: number;
    name: string;
    source: DatasetSource;
    created_at: string;
    last_update: string;
    experiment_count: number;
    observation_count: number;
    edited_experiment_count: number;
    material_count: number;
    species_count: number;
    material_names: string[];
    species_names: string[];
    tags: string[];
    description: string;
}

export interface DatasetImportSummary {
    original_filename: string;
    source_structure: ImportableStructure;
    parser_version: string;
    created_at: string;
}

export interface DatasetDetail {
    status: 'success';
    dataset: DatasetSummary;
    provenance: Record<string, unknown>;
    latest_import: DatasetImportSummary | null;
}

export interface DatasetDeleteImpact {
    dataset_id: number;
    experiment_count: number;
    observation_count: number;
    fitting_run_count: number;
}

export interface ImportIssue {
    code: string;
    severity: 'error' | 'warning' | 'confirmation';
    message: string;
    column: string | null;
    source_row: number | null;
    experiment: string | null;
    remediation: string | null;
}

export interface ColumnDetection {
    name: string;
    inferred_type: string;
    sample_values: unknown[];
    proposed_role: ColumnRole;
    confidence: number;
    evidence: string[];
    detected_unit: string | null;
    array_like: boolean;
}

export interface ImportPreview {
    status: 'success';
    filename: string;
    source_sha256: string;
    row_count: number;
    column_count: number;
    detected_structure: DatasetStructure;
    structure_confidence: number;
    columns: ColumnDetection[];
    preview_rows: Record<string, unknown>[];
    proposed_grouping_columns: string[];
    proposed_pressure_basis: PressureBasis | null;
    issues: ImportIssue[];
    guidance: string[];
}

export interface WidePair {
    pressure_column: string;
    uptake_column: string;
}

export interface ImportMapping {
    dataset_name: string;
    structure: ImportableStructure;
    column_roles: Record<string, ColumnRole>;
    grouping_columns: string[];
    whole_file_grouping?: boolean;
    constants: Record<string, string | number>;
    unit_overrides: Record<string, string>;
    unit_definitions: CustomUnitDefinition[];
    pressure_basis: PressureBasis;
    decimal_separator: 'auto' | '.' | ',';
    field_delimiter?: string | null;
    series_delimiter: string | null;
    wide_pairs: WidePair[];
    worksheet?: string | number | null;
    header_row?: number;
    thousands_separator?: string | null;
    encoding?: string;
    duplicate_policy: 'keep' | 'average' | 'reject';
    confirmed_issue_codes: string[];
}

export interface NormalizedObservationPreview {
    source_row: number | null;
    sequence_index: number;
    pressure_original: number;
    pressure_original_unit: string;
    pressure_canonical: number;
    pressure_canonical_unit: string;
    uptake_original: number;
    uptake_original_unit: string;
    uptake_mol_kg: number;
}

export interface NormalizedExperimentPreview {
    external_key: string;
    name: string;
    adsorbent: string;
    adsorbate: string;
    adsorbate_smiles: string | null;
    temperature_k: number;
    temperature_original: number;
    temperature_original_unit: string;
    pressure_basis: PressureBasis;
    observation_count: number;
    observations: NormalizedObservationPreview[];
}

export interface ImportValidation {
    status: 'valid' | 'invalid' | 'confirmation_required';
    source_sha256: string;
    structure: ImportableStructure;
    experiment_count: number;
    observation_count: number;
    experiments: NormalizedExperimentPreview[];
    issues: ImportIssue[];
}

export interface DatasetImportResponse {
    status: 'success';
    dataset: DatasetSummary;
    validation: ImportValidation;
}

export interface DatasetImportInspection {
    status: 'success';
    dataset_id: number;
    original_filename: string;
    source_sha256: string;
    source_structure: ImportableStructure;
    parser_version: string;
    mapping: ImportMapping;
    validation: ImportValidation;
    warnings: ImportIssue[];
}

export interface ExperimentComponent {
    id: number;
    position: number;
    adsorbate_id: number;
    name: string;
    mole_fraction: number | null;
}

export interface ExperimentSummary {
    id: number;
    dataset_id: number;
    external_key: string;
    name: string;
    adsorbent_id: number;
    adsorbent: string;
    adsorbates: string[];
    components: ExperimentComponent[];
    temperature_k: number;
    temperature_original: number | null;
    temperature_original_unit: string | null;
    pressure_basis: PressureBasis;
    saturation_pressure_pa: number | null;
    duplicate_policy: 'reject' | 'keep' | 'average';
    observation_count: number;
    pressure_min: number | null;
    pressure_max: number | null;
    pressure_canonical_unit: string | null;
    pressure_original_unit: string | null;
    uptake_original_unit: string | null;
    edited_at: string | null;
    fitting_eligible: boolean;
    ineligibility_reason: string | null;
}

export interface ExperimentPage {
    status: 'success';
    dataset_id: number;
    offset: number;
    limit: number;
    total: number;
    experiments: ExperimentSummary[];
}

export interface ExperimentFilters {
    q?: string;
    adsorbent_id?: number | null;
    adsorbate_id?: number | null;
    edited_only?: boolean;
}

export interface ExperimentProvenance {
    source: string;
    external_id: string;
    source_url: string | null;
    doi: string[];
}

export interface ExperimentDetail {
    status: 'success';
    dataset_name: string;
    experiment: ExperimentSummary;
    conditions: Record<string, unknown>;
    provenance: ExperimentProvenance[];
}

export interface ComponentUpdate {
    id: number;
    adsorbate_id?: number;
    mole_fraction?: number | null;
}

export interface ExperimentUpdate {
    id: number;
    name?: string;
    adsorbent_id?: number;
    temperature_original?: number;
    temperature_original_unit?: string;
    pressure_unit?: string;
    uptake_unit?: string;
    pressure_basis?: PressureBasis;
    saturation_pressure_pa?: number | null;
    components?: ComponentUpdate[];
}

export interface ObservationUpdate {
    id: number;
    pressure_original?: number | null;
    uptake_original?: number | null;
    uptake_stddev_mol_kg?: number | null;
}

export interface DatasetUpdateRequest {
    name?: string;
    description?: string;
    tags?: string[];
    experiments?: ExperimentUpdate[];
    observations?: ObservationUpdate[];
}

export interface DatasetEditIssue {
    scope: 'dataset' | 'experiment' | 'observation';
    field: string;
    message: string;
    experiment_id: number | null;
    observation_id: number | null;
}

export interface ObservationRow {
    id: number;
    component_id: number;
    sequence_index: number;
    source_row: number | null;
    pressure_original: number;
    pressure_original_unit: string;
    pressure_canonical: number;
    pressure_canonical_unit: string;
    uptake_original: number;
    uptake_original_unit: string;
    uptake_mol_kg: number;
    uptake_stddev_mol_kg: number | null;
}

export interface ObservationPage {
    status: 'success';
    dataset_id: number;
    isotherm_id: number;
    offset: number;
    limit: number;
    total: number;
    rows: ObservationRow[];
}
