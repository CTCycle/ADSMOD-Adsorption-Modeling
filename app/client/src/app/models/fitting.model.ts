export interface ParameterConfiguration {
    lower: number;
    upper: number;
    initial: number;
}

export interface DisplayUnits {
    pressure: string;
    uptake: string;
}

export interface FittingPayload {
    dataset_id: number;
    isotherm_id: number;
    models: string[];
    optimizer: 'trf' | 'dogbox';
    max_evaluations: number;
    weighting: 'unweighted' | 'inverse_sigma';
    parameter_configuration: Record<
        string,
        Record<string, ParameterConfiguration>
    >;
    display_units: DisplayUnits;
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

export interface PredictionPoint {
    pressure: number;
    observed: number | null;
    predicted: number;
    residual: number | null;
}

export interface ModelFitResult {
    model: string;
    name: string;
    status: 'success' | 'warning' | 'failed';
    convergence_message: string;
    function_evaluations: number | null;
    jacobian_rank: number | null;
    condition_number: number | null;
    parameters: FittedParameter[];
    metrics: FitMetrics;
    observed_predictions: PredictionPoint[];
    curve: PredictionPoint[];
    warnings: string[];
    rank: number | null;
}

export interface FittingResponse {
    status: 'success' | 'warning' | 'error';
    run_id: number | null;
    dataset_id: number;
    isotherm_id: number;
    dataset_name: string;
    experiment_name: string;
    adsorbent: string;
    adsorbate: string;
    temperature_k: number;
    pressure_basis: string;
    pressure_unit: string;
    uptake_unit: string;
    observation_count: number;
    best_model: string | null;
    results: ModelFitResult[];
    summary: string;
}

export interface ModelParameters {
    [parameterName: string]: { min: number; max: number };
}

export interface ModelCatalogParameter { name: string; label: string; lower: number; upper: number; initial: number; unit: string; }
export interface ModelCatalogEntry { key: string; name: string; equation_latex: string; assumptions: string; parameters: ModelCatalogParameter[]; }
export interface ModelCatalogResponse { status: 'success'; pressure_unit: string; uptake_unit: string; models: ModelCatalogEntry[]; }

export type ParameterKey = [string, string, string];
