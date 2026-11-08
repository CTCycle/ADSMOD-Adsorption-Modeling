// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export type OptimizationMethod = 'trf' | 'dogbox';
export type FittingWeighting = 'unweighted' | 'inverse_sigma';
export type FittingSessionStatus = 'running' | 'completed' | 'warning' | 'failed' | 'cancelled';

/** Search bounds in canonical units (Pa, mol/kg). */
export interface ParameterBounds {
    lower: number;
    upper: number;
}

export type ModelBounds = Record<string, ParameterBounds>;

export interface ModelCatalogParameter {
    name: string;
    label: string;
    lower: number;
    upper: number;
    canonical_lower: number;
    canonical_upper: number;
    unit: string;
    customized: boolean;
}

export interface ModelCatalogEntry {
    key: string;
    name: string;
    equation_latex: string;
    assumptions: string;
    pressure_requirement: string;
    requires_temperature: boolean;
    reference: string;
    enabled: boolean;
    parameters: ModelCatalogParameter[];
}

export interface ModelCatalogResponse {
    status: 'success';
    models: ModelCatalogEntry[];
}

export interface ModelSettingsUpdate {
    enabled: boolean;
    parameters: ModelBounds;
}

export interface NumericBounds { minimum: number; maximum: number; }

export interface FittingConfiguration {
    status: 'success';
    supported_optimizers: OptimizationMethod[];
    default_optimizer: OptimizationMethod;
    default_max_evaluations: number;
    max_evaluations_bounds: NumericBounds;
    weighting_options: FittingWeighting[];
    default_weighting: FittingWeighting;
    display_units: {
        pressure: string[];
        uptake: string[];
        default_pressure: string;
        default_uptake: string;
    };
}

export interface FittingExperimentReadiness {
    isotherm_id: number;
    name: string;
    eligible: boolean;
    reason: string | null;
}

export interface FittingDatasetReadiness {
    status: 'success';
    dataset_id: number;
    dataset_name: string;
    experiment_count: number;
    eligible_count: number;
    ready: boolean;
    inverse_sigma_available: boolean;
    model_incompatibility: Record<string, string>;
    experiments: FittingExperimentReadiness[];
}

export interface FittingSessionPayload {
    name: string;
    dataset_id: number;
    models: string[];
    optimizer: OptimizationMethod;
    max_evaluations: number;
    weighting: FittingWeighting;
    parameter_overrides: Record<string, ModelBounds>;
    display_units: { pressure: string; uptake: string };
}

export interface FittingSessionStartResponse {
    status: 'success';
    session_id: number;
    job_id: string;
    poll_interval: number;
}

export interface FittedParameter {
    name: string;
    label: string;
    value: number;
    standard_error: number | null;
    ci95_low: number | null;
    ci95_high: number | null;
    unit: string;
}

export interface FitMetrics {
    sse: number | null;
    rmse: number | null;
    mae: number | null;
    r_squared: number | null;
    adjusted_r_squared: number | null;
    chi_square: number | null;
    aic: number | null;
    aicc: number | null;
    bic: number | null;
}

export interface SessionModelResult {
    model: string;
    name: string;
    status: 'success' | 'warning' | 'failed';
    convergence_message: string;
    rank: number | null;
    observation_count: number;
    parameters: FittedParameter[];
    metrics: FitMetrics;
    warnings: string[];
}

export interface ExperimentFitResult {
    isotherm_id: number | null;
    experiment_name: string;
    pressure_unit: string;
    uptake_unit: string;
    observation_count: number;
    best_model: string | null;
    results: SessionModelResult[];
}

export interface FittingSessionSummary {
    id: number;
    name: string;
    dataset_id: number;
    dataset_name: string;
    status: FittingSessionStatus;
    message: string;
    models: string[];
    experiment_count: number;
    optimizer: OptimizationMethod;
    weighting: FittingWeighting;
    max_evaluations: number;
    created_at: string;
    completed_at: string | null;
    duration_seconds: number | null;
    best_models: Record<string, string | null>;
}

export interface FittingSessionSnapshot {
    models: string[];
    model_bounds: Record<string, ModelBounds>;
    session_overrides: Record<string, ModelBounds>;
    experiments: { isotherm_id: number; name: string; input_sha256: string }[];
    [key: string]: unknown;
}

export interface FittingSessionDetail extends FittingSessionSummary {
    configuration: FittingSessionSnapshot;
    log: string[];
    experiments: ExperimentFitResult[];
    job_id: string | null;
    stale: boolean;
    stale_reason: string | null;
}

export interface FittingSessionListResponse {
    status: 'success';
    items: FittingSessionSummary[];
    total: number;
    page: number;
    page_size: number;
}

export type FittingSessionSort = 'created_at' | 'name' | 'status';

export interface FittingSessionQuery {
    query: string;
    statuses: FittingSessionStatus[];
    datasetId: number | null;
    model: string | null;
    createdFrom: string;
    createdTo: string;
    sort: FittingSessionSort;
    direction: 'asc' | 'desc';
    page: number;
    pageSize: number;
}
