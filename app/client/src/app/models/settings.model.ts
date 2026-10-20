export type RuntimeSettings = {
    host: '127.0.0.1' | 'localhost' | '::1';
    backend_port: number;
    frontend_port: number;
};

export type SettingsModelMetric =
    | 'sse'
    | 'rmse'
    | 'mae'
    | 'r_squared'
    | 'adjusted_r_squared'
    | 'chi_square'
    | 'aic'
    | 'aicc'
    | 'bic';

export interface UserSettings {
    runtime: RuntimeSettings;
    datasets: {
        allowed_extensions: Array<'.csv' | '.xls' | '.xlsx'>;
        column_detection_cutoff: number;
    };
    nist: { parallel_tasks: number };
    public_data: {
        request_timeout_seconds: number;
        retry_attempts: number;
        retry_backoff_max_seconds: number;
        pubchem_parallel_requests: number;
        cod_max_interactive_results: number;
        health_cache_ttl_seconds: number;
        sync_batch_size: number;
        sync_max_records_per_run: number;
        sync_max_concurrent_jobs: number;
        sync_response_size_limit_bytes: number;
        sync_resume_on_startup: boolean;
        auto_sync_enabled: boolean;
        auto_sync_categories: Array<'experiments' | 'guest' | 'host'>;
        auto_sync_interval_hours: number;
        auto_sync_fraction: number;
    };
    fitting: {
        default_max_iterations: number;
        max_iterations_upper_bound: number;
        default_parameter_initial: number;
        default_parameter_min: number;
        default_parameter_max: number;
        preview_row_limit: number;
        best_model_metric: SettingsModelMetric;
    };
    jobs: { polling_interval: number };
    training: {
        use_jit: boolean;
        jit_backend: 'inductor' | 'eager' | 'aot_eager' | 'cudagraphs';
        use_mixed_precision: boolean;
        dataloader_workers: number;
        persistent_workers: boolean;
    };
}

export interface SettingsResponse {
    status: 'success';
    current: UserSettings;
    defaults: UserSettings;
    restart_required: boolean;
    restart_required_fields: string[];
    pending_restart_fields: string[];
}
