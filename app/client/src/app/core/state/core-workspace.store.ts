import { computed, Injectable, signal } from '@angular/core';
import type {
    DatasetMetadata,
    DatasetSummary,
    ExperimentSummary,
} from '../../models/dataset.model';
import type {
    FittingConfiguration,
    FittingPayload,
    FittingResponse,
    FittingResultSummary,
    ModelParameters,
    ModelCatalogResponse,
    PersistedFittingRunResponse,
} from '../../models/fitting.model';
import {
    deleteDataset,
    fetchDatasets,
    fetchExperiments,
    renameDataset,
    updateMetadata,
} from '../../services/dataset.service';
import {
    cancelFittingJob,
    fetchPersistedFittingRun,
    pollFittingJobUntilComplete,
    startFittingJob,
    fetchFittingConfiguration,
    fetchModelCatalog,
} from '../../services/fitting.service';

export type OptimizationMethod = FittingPayload['optimizer'];

const FITTING_LAST_RUN_KEY = 'adsmod.fitting.last-run';

interface StoredFittingRunContext {
    run_id: number;
    dataset_name: string;
    experiment_name: string;
    observation_count: number;
    best_model: string | null;
    model_names: Record<string, string>;
}

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

const initialModels = (): Record<string, ModelState> => ({});

@Injectable({ providedIn: 'root' })
export class CoreWorkspaceStore {
    readonly fittingConfiguration = signal<FittingConfiguration | null>(null);
    readonly fittingConfigurationError = signal<string | null>(null);
    readonly maxEvaluations = signal<number | null>(null);
    readonly optimizationMethod = signal<OptimizationMethod | null>(null);
    readonly weighting = signal<FittingPayload['weighting'] | null>(null);
    readonly fittingStatus = signal('');
    readonly fittingResult = signal<FittingResultSummary | null>(null);
    readonly fittingRunning = signal(false);
    readonly fittingJobId = signal<string | null>(null);
    readonly fittingCancellationPending = signal(false);
    readonly datasets = signal<DatasetSummary[]>([]);
    readonly customDatasets = computed(() => this.datasets().filter((dataset) => dataset.source === 'uploaded'));
    readonly selectedDatasetId = signal<number | null>(null);
    readonly experiments = signal<ExperimentSummary[]>([]);
    readonly selectedExperimentId = signal<number | null>(null);
    readonly experimentsLoading = signal(false);
    readonly managementStatus = signal('');
    readonly modelStates = signal<Record<string, ModelState>>(initialModels());
    readonly modelCatalog = signal<ModelCatalogResponse | null>(null);
    readonly modelCatalogError = signal<string | null>(null);

    private experimentLoadRevision = 0;
    private fittingRevision = 0;

    readonly selectedDataset = computed(() =>
        this.datasets().find(
            (dataset) => dataset.id === this.selectedDatasetId(),
        ),
    );
    readonly selectedExperiment = computed(() =>
        this.experiments().find(
            (experiment) => experiment.id === this.selectedExperimentId(),
        ),
    );
    readonly selectedModelCount = computed(
        () =>
            Object.values(this.modelStates()).filter((state) => state.enabled)
                .length,
    );

    constructor() {
        void this.refreshDatasets();
        void this.loadConfiguration();
        void this.loadCatalog();
        void this.restoreLastFittingRun();
    }

    async loadConfiguration(): Promise<void> {
        const result = await fetchFittingConfiguration();
        this.fittingConfigurationError.set(result.error);
        if (!result.data) {
            this.fittingConfiguration.set(null);
            this.maxEvaluations.set(null);
            this.optimizationMethod.set(null);
            this.weighting.set(null);
            return;
        }
        this.fittingConfiguration.set(result.data);
        this.maxEvaluations.set(result.data.default_max_evaluations);
        this.optimizationMethod.set(result.data.default_optimizer);
        this.weighting.set(result.data.default_weighting);
    }

    async loadCatalog(): Promise<void> {
        const result = await fetchModelCatalog();
        this.modelCatalogError.set(result.error);
        if (!result.data) {
            this.modelCatalog.set(null);
            this.modelStates.set(initialModels());
            return;
        }
        this.modelCatalog.set(result.data);
        this.modelStates.set(Object.fromEntries(result.data.models.map((model) => [
            model.key,
            {
                enabled: true,
                config: Object.fromEntries(model.parameters.map((parameter) => [
                    parameter.name,
                    { min: parameter.lower, max: parameter.upper },
                ])),
            },
        ])));
    }

    async refreshDatasets(selectId?: number): Promise<void> {
        const result = await fetchDatasets();
        if (result.error || !result.data) {
            this.managementStatus.set(
                result.error || 'Failed to load datasets.',
            );
            return;
        }
        this.datasets.set(result.data.datasets);
        if (selectId !== undefined) {
            await this.selectDataset(selectId);
        } else if (
            this.selectedDatasetId() !== null &&
            !result.data.datasets.some(
                (dataset) => dataset.id === this.selectedDatasetId(),
            )
        ) {
            await this.selectDataset(null);
        }
    }

    async selectDataset(datasetId: number | null): Promise<void> {
        const revision = ++this.experimentLoadRevision;
        this.fittingRevision += 1;
        this.fittingRunning.set(false);
        this.selectedDatasetId.set(datasetId);
        this.selectedExperimentId.set(null);
        this.experiments.set([]);
        this.fittingResult.set(null);
        this.clearStoredFittingRun();
        if (datasetId === null) {
            this.experimentsLoading.set(false);
            return;
        }
        this.experimentsLoading.set(true);
        const result = await fetchExperiments(datasetId);
        if (
            revision !== this.experimentLoadRevision ||
            this.selectedDatasetId() !== datasetId
        ) {
            return;
        }
        this.experimentsLoading.set(false);
        if (result.error || !result.data) {
            this.managementStatus.set(
                result.error || 'Failed to load experiments.',
            );
            return;
        }
        this.experiments.set(result.data.experiments);
        if (result.data.experiments.length === 1) {
            this.selectedExperimentId.set(result.data.experiments[0].id);
        }
    }

    setSelectedExperiment(experimentId: number | null): void {
        this.fittingRevision += 1;
        this.fittingRunning.set(false);
        this.selectedExperimentId.set(experimentId);
        this.fittingResult.set(null);
        this.clearStoredFittingRun();
    }

    async deleteDataset(datasetId: number): Promise<void> {
        const result = await deleteDataset(datasetId);
        if (result.error) {
            this.managementStatus.set(result.error);
            return;
        }
        if (this.selectedDatasetId() === datasetId) {
            await this.selectDataset(null);
        }
        await this.refreshDatasets();
    }

    async renameDataset(datasetId: number, newName: string): Promise<void> {
        const result = await renameDataset(datasetId, newName);
        if (result.error) {
            this.managementStatus.set(result.error);
            return;
        }
        await this.refreshDatasets();
    }

    async saveMetadata(
        datasetId: number,
        metadata: DatasetMetadata,
    ): Promise<void> {
        const result = await updateMetadata(datasetId, metadata);
        if (result.error) {
            this.managementStatus.set(result.error);
            return;
        }
        await this.refreshDatasets();
    }

    setOptimizationMethod(method: OptimizationMethod): void {
        this.optimizationMethod.set(method);
    }

    setMaxEvaluations(value: number): void {
        this.maxEvaluations.set(Math.round(value));
    }

    setWeighting(weighting: FittingPayload['weighting']): void {
        this.weighting.set(weighting);
    }

    resetFittingStatus(): void {
        this.fittingStatus.set('');
        this.fittingResult.set(null);
        this.clearStoredFittingRun();
    }

    setModelEnabled(modelId: string, enabled: boolean): void {
        this.modelStates.update((current) => {
            const model = current[modelId];
            return model
                ? { ...current, [modelId]: { ...model, enabled } }
                : current;
        });
    }

    setModelParameters(modelId: string, config: ModelParameters): void {
        this.modelStates.update((current) => {
            const model = current[modelId];
            return model
                ? { ...current, [modelId]: { ...model, config } }
                : current;
        });
    }

    async startFitting(): Promise<void> {
        if (this.fittingRunning()) return;
        const datasetId = this.selectedDatasetId();
        const experiment = this.selectedExperiment();
        if (datasetId === null) {
            this.fittingStatus.set('[ERROR] Select one dataset.');
            return;
        }
        if (!experiment) {
            this.fittingStatus.set('[ERROR] Select one experiment or isotherm.');
            return;
        }
        if (!experiment.fitting_eligible) {
            this.fittingStatus.set(
                `[ERROR] ${experiment.ineligibility_reason || 'This experiment is not eligible for fitting.'}`,
            );
            return;
        }
        const models = Object.entries(this.modelStates())
            .filter(([, state]) => state.enabled)
            .map(([model]) => model);
        if (!models.length) {
            this.fittingStatus.set('[ERROR] Select at least one model.');
            return;
        }
        const configuration = this.fittingConfiguration();
        const optimizer = this.optimizationMethod();
        const maxEvaluations = this.maxEvaluations();
        const weighting = this.weighting();
        if (!configuration || !optimizer || maxEvaluations === null || !weighting) {
            this.fittingStatus.set(
                `[ERROR] Fitting configuration is unavailable${this.fittingConfigurationError() ? `: ${this.fittingConfigurationError()}` : '.'}`,
            );
            return;
        }

        const revision = ++this.fittingRevision;
        this.clearStoredFittingRun();
        const fittingContext = {
            datasetId,
            experimentId: experiment.id,
        };

        const payload: FittingPayload = {
            dataset_id: datasetId,
            isotherm_id: experiment.id,
            models,
            optimizer,
            max_evaluations: maxEvaluations,
            weighting,
            parameter_configuration: Object.fromEntries(
                Object.entries(this.modelStates())
                    .filter(([, state]) => state.enabled)
                    .map(([modelId, state]) => {
                        const model = this.modelCatalog()?.models.find(
                            (candidate) => candidate.key === modelId,
                        );
                        const parameters = Object.fromEntries(
                            Object.entries(state.config).map(([name, bounds]) => {
                                const catalogInitial = model?.parameters.find(
                                    (parameter) => parameter.name === name,
                                )?.initial;
                                const initial = catalogInitial ?? (bounds.min + bounds.max) / 2;
                                return [name, {
                                    lower: bounds.min,
                                    upper: bounds.max,
                                    initial: Math.min(bounds.max, Math.max(bounds.min, initial)),
                                }];
                            }),
                        );
                        return [modelId, parameters];
                    }),
            ),
            display_units: {
                pressure: experiment.pressure_basis === 'relative' ? '1' : configuration.display_units.default_pressure,
                uptake: configuration.display_units.default_uptake,
            },
        };
        this.fittingRunning.set(true);
        this.fittingJobId.set(null);
        this.fittingCancellationPending.set(false);
        this.fittingStatus.set('[INFO] Fitting canonical observation series…');
        this.fittingResult.set(null);
        const started = await startFittingJob(payload);
        if (
            revision !== this.fittingRevision ||
            this.selectedDatasetId() !== fittingContext.datasetId ||
            this.selectedExperimentId() !== fittingContext.experimentId
        ) {
            if (started.jobId) await cancelFittingJob(started.jobId);
            this.fittingRunning.set(false);
            this.fittingJobId.set(null);
            this.fittingCancellationPending.set(false);
            return;
        }
        if (started.error || !started.jobId) {
            this.fittingRunning.set(false);
            this.fittingJobId.set(null);
            this.fittingStatus.set(
                `[ERROR] ${started.error || 'Failed to start fitting.'}`,
            );
            return;
        }
        this.fittingJobId.set(started.jobId);
        const result = await pollFittingJobUntilComplete(
            started.jobId,
            started.pollInterval,
        );
        if (
            revision !== this.fittingRevision ||
            this.selectedDatasetId() !== fittingContext.datasetId ||
            this.selectedExperimentId() !== fittingContext.experimentId
        ) {
            this.fittingRunning.set(false);
            this.fittingJobId.set(null);
            this.fittingCancellationPending.set(false);
            return;
        }
        this.fittingRunning.set(false);
        this.fittingJobId.set(null);
        this.fittingCancellationPending.set(false);
        this.fittingStatus.set(result.message);
        this.fittingResult.set(result.data);
        this.storeFittingRunContext(result.data);
    }

    async cancelFitting(): Promise<void> {
        const jobId = this.fittingJobId();
        if (!jobId || !this.fittingRunning() || this.fittingCancellationPending()) {
            return;
        }
        this.fittingCancellationPending.set(true);
        const result = await cancelFittingJob(jobId);
        if (result.error) {
            this.fittingCancellationPending.set(false);
            this.fittingStatus.set(
                `[WARN] Cancellation could not be confirmed: ${result.error}`,
            );
            return;
        }
        this.fittingStatus.set('[INFO] Cancellation requested…');
    }

    private async restoreLastFittingRun(): Promise<void> {
        const context = this.readStoredFittingRunContext();
        if (!context) return;

        const revision = this.fittingRevision;
        const restored = await fetchPersistedFittingRun(context.run_id);
        if (
            revision !== this.fittingRevision ||
            this.fittingRunning() ||
            !this.readStoredFittingRunContext()
        ) {
            return;
        }
        if (!restored.data) {
            this.fittingStatus.set(
                `[WARN] Could not restore the last fitting result: ${restored.error || 'No result was returned.'}`,
            );
            if (restored.error?.toLowerCase().includes('does not exist')) {
                this.clearStoredFittingRun();
            }
            return;
        }
        if (
            restored.data.status_detail !== 'completed' &&
            restored.data.status_detail !== 'warning'
        ) {
            this.clearStoredFittingRun();
            return;
        }

        this.fittingResult.set(
            this.toFittingResultSummary(restored.data, context),
        );
        this.fittingStatus.set('[INFO] Restored the last completed fitting run.');
    }

    private toFittingResultSummary(
        run: PersistedFittingRunResponse,
        context: StoredFittingRunContext,
    ): FittingResultSummary {
        return {
            status: run.status_detail === 'warning' ? 'warning' : 'success',
            run_id: run.run_id,
            dataset_id: run.dataset_id,
            isotherm_id: run.isotherm_id,
            dataset_name: context.dataset_name,
            experiment_name: context.experiment_name,
            observation_count: context.observation_count,
            best_model: context.best_model,
            results: run.results.map((result) => ({
                model: result.model,
                name: context.model_names[result.model] ?? result.model,
                status: result.status,
                metrics: result.metrics,
            })),
            summary: run.message,
        };
    }

    private storeFittingRunContext(result: FittingResponse | null): void {
        if (!result?.run_id || result.status === 'error') return;
        try {
            const context: StoredFittingRunContext = {
                run_id: result.run_id,
                dataset_name: result.dataset_name,
                experiment_name: result.experiment_name,
                observation_count: result.observation_count,
                best_model: result.best_model,
                model_names: Object.fromEntries(
                    result.results.map((fit) => [fit.model, fit.name]),
                ),
            };
            window.localStorage.setItem(
                FITTING_LAST_RUN_KEY,
                JSON.stringify(context),
            );
        } catch {
            // Local storage can be disabled; the live result remains usable.
        }
    }

    private readStoredFittingRunContext(): StoredFittingRunContext | null {
        try {
            const stored = window.localStorage.getItem(FITTING_LAST_RUN_KEY);
            if (!stored) return null;
            const context = JSON.parse(stored) as Partial<StoredFittingRunContext>;
            if (
                !Number.isInteger(context.run_id) ||
                (context.run_id ?? 0) < 1 ||
                typeof context.dataset_name !== 'string' ||
                typeof context.experiment_name !== 'string' ||
                !Number.isInteger(context.observation_count) ||
                !context.model_names ||
                typeof context.model_names !== 'object'
            ) {
                this.clearStoredFittingRun();
                return null;
            }
            return context as StoredFittingRunContext;
        } catch {
            this.clearStoredFittingRun();
            return null;
        }
    }

    private clearStoredFittingRun(): void {
        try {
            window.localStorage.removeItem(FITTING_LAST_RUN_KEY);
        } catch {
            // Local storage can be disabled.
        }
    }
}
