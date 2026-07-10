import { computed, Injectable, signal } from '@angular/core';
import { ADSORPTION_MODELS } from '../constants/adsorption-models';
import type { DatasetMetadata, DatasetRowMutation, DatasetRowsPage, DatasetSummary } from '../../models/dataset.model';
import type { FittingPayload, ModelConfiguration, ModelParameters } from '../../models/fitting.model';
import { deleteDataset, fetchDatasets, fetchRows, mutateRows, renameDataset, updateMetadata, uploadDataset } from '../../services/dataset.service';
import { pollFittingJobUntilComplete, startFittingJob } from '../../services/fitting.service';

export type CorePageId = 'source' | 'fitting';
export type OptimizationMethod = FittingPayload['optimization_method'];

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

const NIST_DATASET_OPTION = '__NIST_A_COLLECTION__';

const initialModels = (): Record<string, ModelState> => Object.fromEntries(
    ADSORPTION_MODELS.map((model) => [
        model.name,
        {
            enabled: true,
            config: Object.fromEntries(
                Object.entries(model.parameterDefaults).map(([name, [min, max]]) => [name, { min, max }])
            ),
        },
    ])
);

@Injectable({ providedIn: 'root' })
export class CoreWorkspaceStore {
    readonly currentPage = signal<CorePageId>('source');
    readonly maxIterations = signal(10000);
    readonly optimizationMethod = signal<OptimizationMethod>('LSS');
    readonly fittingStatus = signal('');
    readonly datasetStats = signal('Select a dataset to view its summary.');
    readonly pendingFile = signal<File | null>(null);
    readonly isDatasetUploading = signal(false);
    readonly userDatasets = signal<DatasetSummary[]>([]);
    readonly selectedDataset = signal<string | null>(null);
    readonly editorPage = signal<DatasetRowsPage | null>(null);
    readonly editorLimit = signal(100);
    readonly editorLoading = signal(false);
    readonly managementStatus = signal('');
    readonly modelStates = signal<Record<string, ModelState>>(initialModels());
    readonly nistDatasetOption = NIST_DATASET_OPTION;

    readonly availableDatasets = computed(() => this.userDatasets().map((dataset) => dataset.name));
    readonly selectedModelCount = computed(() => Object.values(this.modelStates()).filter((state) => state.enabled).length);

    constructor() {
        void this.refreshUserDatasets();
    }

    setCurrentPage(page: CorePageId): void {
        this.currentPage.set(page);
    }

    setPendingFile(file: File): void {
        this.pendingFile.set(file);
    }

    async uploadPendingDataset(): Promise<void> {
        const file = this.pendingFile();
        if (!file) {
            this.managementStatus.set('Choose a file first.');
            return;
        }

        this.isDatasetUploading.set(true);
        const result = await uploadDataset(file);
        this.isDatasetUploading.set(false);
        if (result.error || !result.data) {
            this.managementStatus.set(result.error || 'Upload failed.');
            return;
        }

        this.pendingFile.set(null);
        this.datasetStats.set(result.data.summary);
        await this.refreshUserDatasets();
        await this.openDataset(result.data.dataset.name);
    }

    async refreshUserDatasets(): Promise<void> {
        const result = await fetchDatasets();
        if (result.error || !result.data) {
            this.managementStatus.set(result.error || 'Failed to load datasets.');
            return;
        }

        this.userDatasets.set(result.data.datasets);
    }

    async openDataset(name: string): Promise<void> {
        this.selectedDataset.set(name);
        await this.loadDatasetPage(0, this.editorLimit());
    }

    async loadDatasetPage(offset: number, limit = this.editorLimit()): Promise<void> {
        const name = this.selectedDataset();
        if (!name) {
            return;
        }

        this.editorLoading.set(true);
        const result = await fetchRows(name, offset, limit);
        this.editorLoading.set(false);
        if (result.error || !result.data) {
            this.managementStatus.set(result.error || 'Failed to load rows.');
            return;
        }

        this.editorLimit.set(limit);
        this.editorPage.set(result.data);
        const summary = this.userDatasets().find((dataset) => dataset.name === name);
        if (summary) {
            this.datasetStats.set(
                `### Dataset overview\n\n| Metric | Value |\n|---|---:|\n| Rows | ${summary.row_count} |\n| Columns | ${summary.column_count} |`
            );
        }
    }

    async deleteDataset(name: string): Promise<void> {
        const result = await deleteDataset(name);
        if (result.error) {
            this.managementStatus.set(result.error);
            return;
        }

        if (this.selectedDataset() === name) {
            this.selectedDataset.set(null);
            this.editorPage.set(null);
        }
        await this.refreshUserDatasets();
    }

    async renameDataset(name: string, newName: string): Promise<void> {
        const result = await renameDataset(name, newName);
        if (result.error || !result.data) {
            this.managementStatus.set(result.error || 'Rename failed.');
            return;
        }

        if (this.selectedDataset() === name) {
            this.selectedDataset.set(newName);
        }
        await this.refreshUserDatasets();
    }

    async saveMetadata(metadata: DatasetMetadata): Promise<void> {
        const name = this.selectedDataset();
        if (!name) {
            return;
        }

        const result = await updateMetadata(name, metadata);
        if (result.error) {
            this.managementStatus.set(result.error);
            return;
        }
        await this.refreshUserDatasets();
    }

    async saveMutations(operations: DatasetRowMutation[]): Promise<void> {
        const name = this.selectedDataset();
        const page = this.editorPage();
        if (!name || !page || !operations.length) {
            return;
        }

        const result = await mutateRows(name, operations);
        if (result.error) {
            this.managementStatus.set(result.error);
            return;
        }

        await this.refreshUserDatasets();
        await this.loadDatasetPage(page.offset, page.limit);
    }

    setSelectedDataset(name: string): void {
        this.selectedDataset.set(name || null);
    }

    setOptimizationMethod(method: OptimizationMethod): void {
        this.optimizationMethod.set(method);
    }

    setMaxIterations(value: number): void {
        this.maxIterations.set(Math.max(1, Math.round(value)));
    }

    resetFittingStatus(): void {
        this.fittingStatus.set('');
    }

    setModelEnabled(name: string, enabled: boolean): void {
        this.modelStates.update((current) => {
            const model = current[name];
            if (!model) {
                return current;
            }

            return { ...current, [name]: { ...model, enabled } };
        });
    }

    setModelParameters(name: string, config: ModelParameters): void {
        this.modelStates.update((current) => {
            const model = current[name];
            if (!model) {
                return current;
            }

            return { ...current, [name]: { ...model, config } };
        });
    }

    async startFitting(): Promise<void> {
        const name = this.selectedDataset();
        if (!name) {
            this.fittingStatus.set('[ERROR] Select a dataset.');
            return;
        }

        const enabledModels = Object.entries(this.modelStates()).filter(([, state]) => state.enabled);
        if (!enabledModels.length) {
            this.fittingStatus.set('[ERROR] Select at least one model.');
            return;
        }

        const parameter_bounds: Record<string, ModelConfiguration> = {};
        for (const [modelName, modelState] of enabledModels) {
            const min: Record<string, number> = {};
            const max: Record<string, number> = {};
            const initial: Record<string, number> = {};

            for (const [parameterName, bounds] of Object.entries(modelState.config)) {
                const lower = Math.min(bounds.min, bounds.max);
                const upper = Math.max(bounds.min, bounds.max);
                min[parameterName] = lower;
                max[parameterName] = upper;
                initial[parameterName] = (lower + upper) / 2;
            }

            parameter_bounds[modelName] = { min, max, initial };
        }

        const dataset: FittingPayload['dataset'] = name === NIST_DATASET_OPTION
            ? { source: 'nist' }
            : { source: 'uploaded', dataset_name: name };
        const started = await startFittingJob({
            max_iterations: this.maxIterations(),
            optimization_method: this.optimizationMethod(),
            parameter_bounds,
            dataset,
        });
        if (started.error || !started.jobId) {
            this.fittingStatus.set(`[ERROR] ${started.error || 'Failed to start fitting.'}`);
            return;
        }

        const result = await pollFittingJobUntilComplete(started.jobId, started.pollInterval);
        this.fittingStatus.set(result.message);
    }
}