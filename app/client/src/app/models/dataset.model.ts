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
    | 'adsorbent'
    | 'temperature'
    | 'pressure_unit'
    | 'uptake_unit'
    | 'temperature_unit'
    | 'uptake_stddev'
    | 'saturation_pressure'
    | 'metadata'
    | 'ignore';

export interface DatasetMetadata {
    tags: string[];
    description: string;
}

export interface DatasetSummary extends DatasetMetadata {
    id: number;
    name: string;
    source: DatasetSource;
    created_at: string;
    experiment_count: number;
    observation_count: number;
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
    temperature_k: number;
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

export interface ExperimentSummary {
    id: number;
    dataset_id: number;
    external_key: string;
    name: string;
    adsorbent: string;
    adsorbates: string[];
    temperature_k: number;
    pressure_basis: PressureBasis;
    observation_count: number;
    fitting_eligible: boolean;
    ineligibility_reason: string | null;
}

export interface ObservationPage {
    status: 'success';
    dataset_id: number;
    isotherm_id: number;
    offset: number;
    limit: number;
    total: number;
    rows: Record<string, unknown>[];
}
