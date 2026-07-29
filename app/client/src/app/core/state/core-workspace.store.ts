import { computed, Injectable, signal } from '@angular/core';
import type {
    DatasetMetadata,
    DatasetSummary,
    ExperimentSummary,
} from '../../models/dataset.model';
import type {
    FittingPayload,
    FittingResponse,
    ModelParameters,
    ModelCatalogResponse,
} from '../../models/fitting.model';
import {
    deleteDataset,
    fetchDatasets,
    fetchExperiments,
    renameDataset,
    updateMetadata,
} from '../../services/dataset.service';
import {
    pollFittingJobUntilComplete,
    startFittingJob, fetchModelCatalog,
} from '../../services/fitting.service';

export type CorePageId = 'datasets' | 'fitting';
export type OptimizationMethod = FittingPayload['optimizer'];

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

const initialModels = (): Record<string, ModelState> => ({});

@Injectable({ providedIn: 'root' })
export class CoreWorkspaceStore {
    readonly currentPage = signal<CorePageId>('datasets');
    readonly maxEvaluations = signal(10_000);
    readonly optimizationMethod = signal<OptimizationMethod>('trf');
    readonly weighting = signal<FittingPayload['weighting']>('unweighted');
    readonly fittingStatus = signal('');
    readonly fittingResult = signal<FittingResponse | null>(null);
    readonly fittingRunning = signal(false);
    readonly userDatasets = signal<DatasetSummary[]>([]);
    readonly selectedDatasetId = signal<number | null>(null);
    readonly experiments = signal<ExperimentSummary[]>([]);
    readonly selectedExperimentId = signal<number | null>(null);
    readonly experimentsLoading = signal(false);
    readonly managementStatus = signal('');
    readonly modelStates = signal<Record<string, ModelState>>(initialModels());
    readonly modelCatalog = signal<ModelCatalogResponse | null>(null);

    readonly selectedDataset = computed(() =>
        this.userDatasets().find(
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
        void this.loadCatalog();
    }

    async loadCatalog(): Promise<void> { const catalog = await fetchModelCatalog(); if (catalog) { this.modelCatalog.set(catalog); this.modelStates.set(Object.fromEntries(catalog.models.map((model) => [model.key, { enabled: true, config: Object.fromEntries(model.parameters.map((parameter) => [parameter.name, { min: parameter.lower, max: parameter.upper }])) }]))); } }

    setCurrentPage(page: CorePageId): void {
        this.currentPage.set(page);
    }

    async refreshDatasets(selectId?: number): Promise<void> {
        const result = await fetchDatasets();
        if (result.error || !result.data) {
            this.managementStatus.set(
                result.error || 'Failed to load datasets.',
            );
            return;
        }
        this.userDatasets.set(result.data.datasets);
        if (selectId !== undefined) {
            await this.selectDataset(selectId);
        } else if (
            this.selectedDatasetId() !== null &&
            !result.data.datasets.some(
                (dataset) => dataset.id === this.selectedDatasetId(),
            )
        ) {
            this.selectedDatasetId.set(null);
            this.experiments.set([]);
            this.selectedExperimentId.set(null);
        }
    }

    async selectDataset(datasetId: number | null): Promise<void> {
        this.selectedDatasetId.set(datasetId);
        this.selectedExperimentId.set(null);
        this.experiments.set([]);
        this.fittingResult.set(null);
        if (datasetId === null) {
            return;
        }
        this.experimentsLoading.set(true);
        const result = await fetchExperiments(datasetId);
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
        this.selectedExperimentId.set(experimentId);
        this.fittingResult.set(null);
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
        this.maxEvaluations.set(
            Math.min(1_000_000, Math.max(10, Math.round(value))),
        );
    }

    resetFittingStatus(): void {
        this.fittingStatus.set('');
        this.fittingResult.set(null);
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

        const payload: FittingPayload = {
            dataset_id: datasetId,
            isotherm_id: experiment.id,
            models,
            optimizer: this.optimizationMethod(),
            max_evaluations: this.maxEvaluations(),
            weighting: this.weighting(),
            parameter_configuration: {},
            display_units: {
                pressure: experiment.pressure_basis === 'relative' ? '1' : 'bar',
                uptake: 'mmol/g',
            },
        };
        this.fittingRunning.set(true);
        this.fittingStatus.set('[INFO] Fitting canonical observation series…');
        this.fittingResult.set(null);
        const started = await startFittingJob(payload);
        if (started.error || !started.jobId) {
            this.fittingRunning.set(false);
            this.fittingStatus.set(
                `[ERROR] ${started.error || 'Failed to start fitting.'}`,
            );
            return;
        }
        const result = await pollFittingJobUntilComplete(
            started.jobId,
            started.pollInterval,
        );
        this.fittingRunning.set(false);
        this.fittingStatus.set(result.message);
        this.fittingResult.set(result.data);
    }
}
