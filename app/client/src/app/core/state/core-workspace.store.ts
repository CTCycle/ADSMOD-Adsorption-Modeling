import { computed, Injectable, signal } from '@angular/core';
import { ADSORPTION_MODELS } from '../constants/adsorption-models';
import type { DatasetPayload } from '../../models/dataset.model';
import type { FittingPayload, ModelConfiguration, ModelParameters } from '../../models/fitting.model';
import type { JobStatusResponse } from '../../models/job.model';
import {
    fetchDatasetByName,
    fetchDatasetNames,
    loadDataset,
} from '../../services/dataset.service';
import {
    pollFittingJobUntilComplete,
    startFittingJob,
} from '../../services/fitting.service';
import { fetchNistDataForFitting } from '../../services/nist.service';

export type CorePageId = 'source' | 'fitting';
export type OptimizationMethod = FittingPayload['optimization_method'];

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

const NIST_DATASET_OPTION = '__NIST_A_COLLECTION__';

const formatFileSize = (bytes: number): string => {
    const kb = Math.max(1, Math.round(bytes / 1024));
    return `${kb} kb`;
};

const escapeMarkdownTableCell = (value: unknown): string => String(value).replace(/\|/g, '\\|').replace(/\n/g, ' ');

const inferColumnDtype = (values: unknown[]): string => {
    const firstValue = values.find((value) => value !== null && value !== undefined && !(typeof value === 'number' && Number.isNaN(value)));
    if (firstValue === undefined) {
        return 'unknown';
    }
    if (typeof firstValue === 'number') {
        return Number.isInteger(firstValue) ? 'int64' : 'float64';
    }
    if (typeof firstValue === 'boolean') {
        return 'bool';
    }
    return 'str';
};

const buildDatasetSummary = (payload: DatasetPayload): string => {
    const records = Array.isArray(payload.records) ? payload.records : [];
    const providedColumns = Array.isArray(payload.columns) ? payload.columns : [];
    const discoveredColumns = new Set<string>(providedColumns);
    records.forEach((row) => {
        Object.keys(row || {}).forEach((key) => discoveredColumns.add(key));
    });
    const orderedColumns = Array.from(discoveredColumns);

    let totalMissing = 0;
    const columnLines = orderedColumns.map((column) => {
        const values = records.map((row) => row?.[column]);
        const missing = values.filter(
            (value) => value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value))
        ).length;
        totalMissing += missing;
        const dtype = inferColumnDtype(values);
        return `| \`${escapeMarkdownTableCell(column)}\` | \`${escapeMarkdownTableCell(dtype)}\` | ${missing} |`;
    });

    return [
        '### Dataset overview',
        '',
        '| Metric | Value |',
        '|---|---:|',
        `| Rows | ${records.length} |`,
        `| Columns | ${orderedColumns.length} |`,
        `| NaN cells | ${totalMissing} |`,
        '',
        '### Column details',
        '',
        '| Column | Dtype | Missing |',
        '|---|---|---:|',
        ...columnLines,
    ].join('\n');
};

const createInitialModelStates = (): Record<string, ModelState> => {
    const initial: Record<string, ModelState> = {};
    ADSORPTION_MODELS.forEach((model) => {
        const config: ModelParameters = {};
        Object.entries(model.parameterDefaults).forEach(([paramName, [min, max]]) => {
            config[paramName] = { min, max };
        });
        initial[model.name] = { enabled: true, config };
    });
    return initial;
};

@Injectable({ providedIn: 'root' })
export class CoreWorkspaceStore {
    readonly currentPage = signal<CorePageId>('source');
    readonly maxIterations = signal(10000);
    readonly optimizationMethod = signal<OptimizationMethod>('LSS');
    readonly datasetStats = signal('No dataset loaded.');
    readonly nistStatusMessage = signal('NIST-A updates will appear here.');
    readonly fittingStatus = signal('');
    readonly dataset = signal<DatasetPayload | null>(null);
    readonly datasetName = signal<string | null>(null);
    readonly datasetSamples = signal(0);
    readonly datasetSizeKb = signal<string | null>(null);
    readonly pendingFile = signal<File | null>(null);
    readonly pendingFileSize = signal<string | null>(null);
    readonly isDatasetUploading = signal(false);
    readonly modelStates = signal<Record<string, ModelState>>(createInitialModelStates());
    readonly availableDatasets = signal<string[]>([]);
    readonly selectedDataset = signal<string | null>(null);
    readonly selectedModelCount = computed(() => Object.values(this.modelStates()).filter((state) => state.enabled).length);
    readonly nistDatasetOption = NIST_DATASET_OPTION;

    constructor() {
        void this.initialize();
    }

    async initialize(): Promise<void> {
        const result = await fetchDatasetNames();
        if (!result.error) {
            this.availableDatasets.set(result.names);
        }
    }

    setCurrentPage(page: CorePageId): void {
        this.currentPage.set(page);
    }

    setPendingFile(file: File): void {
        this.pendingFile.set(file);
        this.pendingFileSize.set(formatFileSize(file.size));
    }

    async uploadPendingDataset(): Promise<void> {
        const pendingFile = this.pendingFile();
        if (!pendingFile) {
            this.datasetStats.set('[ERROR] Please select a dataset file before uploading.');
            return;
        }

        this.isDatasetUploading.set(true);
        this.datasetStats.set('[INFO] Uploading dataset...');
        const activeFileSize = this.pendingFileSize() || formatFileSize(pendingFile.size);
        const result = await loadDataset(pendingFile);

        if (result.dataset) {
            this.setLoadedDataset(result.dataset, activeFileSize);
            this.datasetStats.set(buildDatasetSummary(result.dataset));
            this.pendingFile.set(null);
            this.pendingFileSize.set(null);
        } else {
            this.datasetStats.set(result.message);
        }

        this.isDatasetUploading.set(false);
        const namesResult = await fetchDatasetNames();
        if (!namesResult.error) {
            this.availableDatasets.set(namesResult.names);
            if (result.dataset?.dataset_name) {
                this.selectedDataset.set(result.dataset.dataset_name);
            }
        }
    }

    setSelectedDataset(datasetName: string): void {
        this.selectedDataset.set(datasetName || null);
    }

    setOptimizationMethod(method: OptimizationMethod): void {
        this.optimizationMethod.set(method);
    }

    setMaxIterations(value: number): void {
        this.maxIterations.set(Math.max(1, Math.round(value)));
    }

    setNistStatusMessage(message: string): void {
        this.nistStatusMessage.set(message);
    }

    resetFittingStatus(): void {
        this.fittingStatus.set('');
    }

    setModelEnabled(modelName: string, enabled: boolean): void {
        this.modelStates.update((prev) => ({
            ...prev,
            [modelName]: { ...prev[modelName], enabled },
        }));
    }

    setModelParameters(modelName: string, config: ModelParameters): void {
        this.modelStates.update((prev) => ({
            ...prev,
            [modelName]: { ...prev[modelName], config },
        }));
    }

    async startFitting(): Promise<void> {
        const fittingDataset = await this.resolveFittingDataset();
        if (!fittingDataset) {
            return;
        }

        const selectedModels = Object.entries(this.modelStates())
            .filter(([, state]) => state.enabled)
            .map(([name]) => name);

        if (selectedModels.length === 0) {
            this.fittingStatus.set('[ERROR] Please select at least one model before starting the fitting process.');
            return;
        }

        const parameterBounds = this.buildParameterBounds(selectedModels);
        const payload: FittingPayload = {
            max_iterations: this.maxIterations(),
            optimization_method: this.optimizationMethod(),
            parameter_bounds: parameterBounds,
            dataset: fittingDataset,
        };

        this.fittingStatus.set('[INFO] Starting fitting job...');
        const startResult = await startFittingJob(payload);
        if (startResult.error || !startResult.jobId) {
            this.fittingStatus.set(`[ERROR] ${startResult.error || 'Failed to start fitting job.'}`);
            return;
        }

        const result = await pollFittingJobUntilComplete(
            startResult.jobId,
            startResult.pollInterval,
            (status: JobStatusResponse) => {
                this.fittingStatus.set(`[INFO] Fitting job ${status.status}. Progress: ${Math.round(status.progress)}%`);
            }
        );
        this.fittingStatus.set(result.message);
    }

    private setLoadedDataset(dataset: DatasetPayload, fileSize: string | null): void {
        this.dataset.set(dataset);
        this.datasetName.set(dataset.dataset_name);
        this.datasetSizeKb.set(fileSize);
        this.datasetSamples.set(Array.isArray(dataset.records) ? dataset.records.length : 0);
    }

    private async resolveFittingDataset(): Promise<DatasetPayload | null> {
        const selectedDataset = this.selectedDataset();
        if (selectedDataset === NIST_DATASET_OPTION) {
            this.fittingStatus.set('[INFO] Loading NIST single-component data...');
            const nistResult = await fetchNistDataForFitting();
            if (nistResult.error || !nistResult.dataset) {
                this.fittingStatus.set(`[ERROR] ${nistResult.error || 'Failed to load NIST data.'}`);
                return null;
            }
            return nistResult.dataset;
        }

        const currentDataset = this.dataset();
        if (selectedDataset && (!currentDataset || currentDataset.dataset_name !== selectedDataset)) {
            this.fittingStatus.set(`[INFO] Loading dataset "${selectedDataset}"...`);
            const lookup = await fetchDatasetByName(selectedDataset);
            if (lookup.error || !lookup.dataset) {
                this.fittingStatus.set(`[ERROR] ${lookup.error || 'Failed to load dataset.'}`);
                return null;
            }
            this.setLoadedDataset(lookup.dataset, this.datasetSizeKb());
            this.datasetStats.set(buildDatasetSummary(lookup.dataset));
            return lookup.dataset;
        }

        if (!currentDataset) {
            this.fittingStatus.set('[ERROR] Please load a dataset before starting the fitting process.');
            return null;
        }
        return currentDataset;
    }

    private buildParameterBounds(selectedModels: string[]): Record<string, ModelConfiguration> {
        const parameterBounds: Record<string, ModelConfiguration> = {};
        selectedModels.forEach((modelName) => {
            const state = this.modelStates()[modelName];
            const modelConfig: ModelConfiguration = { min: {}, max: {}, initial: {} };

            Object.entries(state.config).forEach(([paramName, bounds]) => {
                let { min, max } = bounds;
                if (max < min) {
                    [min, max] = [max, min];
                }
                modelConfig.min[paramName] = min;
                modelConfig.max[paramName] = max;
                modelConfig.initial[paramName] = (min + max) / 2;
            });

            parameterBounds[modelName] = modelConfig;
        });
        return parameterBounds;
    }
}
