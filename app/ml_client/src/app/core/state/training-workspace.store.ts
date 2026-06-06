import { Injectable, signal } from '@angular/core';
import type { InfoModalData } from '../../models/json.model';
import type {
    CheckpointFullDetails,
    DatasetFullInfo,
    CheckpointInfo,
    ProcessedDatasetInfo,
    ResumeTrainingConfig,
    TrainingConfig,
    TrainingStatus,
} from '../../models/training.model';
import {
    deleteCheckpoint,
    fetchCheckpointDetails,
    fetchCheckpoints,
    getTrainingStatus,
    startTraining,
    resumeTraining,
    stopTraining,
} from '../../services/training.service';
import {
    deleteDataset,
    fetchProcessedDatasets,
    getTrainingDatasetInfo,
} from '../../services/dataset-builder.service';

export type TrainingViewId = 'processing' | 'datasets' | 'checkpoints' | 'dashboard';

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
    batch_size: 16,
    shuffle_dataset: true,
    max_buffer_size: 256,
    selected_model: 'SCADS Series',
    dropout_rate: 0.1,
    num_attention_heads: 2,
    num_encoders: 2,
    molecular_embedding_size: 64,
    epochs: 2,
    dataloader_workers: 0,
    prefetch_factor: 1,
    pin_memory: true,
    use_device_GPU: true,
    device_ID: 0,
    use_mixed_precision: false,
    use_jit: false,
    jit_backend: 'inductor',
    use_lr_scheduler: false,
    initial_lr: 1e-4,
    target_lr: 1e-5,
    constant_steps: 5,
    decay_steps: 10,
    save_checkpoints: false,
    checkpoints_frequency: 5,
    custom_name: '',
};

const INITIAL_TRAINING_STATUS: TrainingStatus = {
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    metrics: {},
    history: [],
    log: [],
};

const ARCHIVED_DATASET_PREFIX = 'archived::';

const sanitizeProcessedDatasets = (datasets: ProcessedDatasetInfo[]): ProcessedDatasetInfo[] => {
    const uniqueByLabel = new Map<string, ProcessedDatasetInfo>();
    datasets.forEach((dataset) => {
        const label = String(dataset.dataset_label || '').trim();
        if (!label || label.startsWith(ARCHIVED_DATASET_PREFIX)) {
            return;
        }
        if (!uniqueByLabel.has(label)) {
            uniqueByLabel.set(label, dataset);
        }
    });
    return Array.from(uniqueByLabel.values());
};

@Injectable({ providedIn: 'root' })
export class TrainingWorkspaceStore {
    readonly activeView = signal<TrainingViewId>('processing');
    readonly config = signal<TrainingConfig>({ ...DEFAULT_TRAINING_CONFIG });
    readonly checkpoints = signal<CheckpointInfo[]>([]);
    readonly isLoading = signal(false);
    readonly showNewTrainingWizard = signal(false);
    readonly showResumeTrainingWizard = signal(false);
    readonly resumeConfig = signal<ResumeTrainingConfig>({
        checkpoint_name: '',
        additional_epochs: 10,
    });
    readonly processedDatasets = signal<ProcessedDatasetInfo[]>([]);
    readonly selectedDatasetLabel = signal<string | null>(null);
    readonly selectedDatasetHash = signal<string | null>(null);
    readonly selectedCheckpointName = signal<string | null>(null);
    readonly infoModalOpen = signal(false);
    readonly infoModalTitle = signal('');
    readonly infoModalData = signal<InfoModalData | null>(null);
    readonly trainingStatus = signal<TrainingStatus>(INITIAL_TRAINING_STATUS);
    readonly trainingStatusError = signal<string | null>(null);
    readonly actionLoading = signal(false);

    constructor() {
        void this.refreshWorkspace();
    }

    setActiveView(view: TrainingViewId): void {
        this.activeView.set(view);
    }

    async refreshWorkspace(): Promise<void> {
        this.isLoading.set(true);
        await Promise.all([
            this.loadCheckpoints(),
            this.loadProcessedDatasets(),
            this.checkStatus(),
        ]);
        this.isLoading.set(false);
    }

    async loadCheckpoints(): Promise<void> {
        const result = await fetchCheckpoints();
        if (!result.error) {
            this.checkpoints.set(result.checkpoints);
        }
    }

    async loadProcessedDatasets(): Promise<void> {
        const result = await fetchProcessedDatasets();
        if (!result.error) {
            this.processedDatasets.set(sanitizeProcessedDatasets(result.datasets));
        }
    }

    async checkStatus(): Promise<void> {
        const status = await getTrainingStatus();
        this.trainingStatusError.set(status.error);
        const { error: _error, ...trainingStatus } = status;
        this.trainingStatus.set(trainingStatus);
    }

    selectDataset(dataset: ProcessedDatasetInfo): void {
        this.selectedDatasetLabel.set(dataset.dataset_label);
        this.selectedDatasetHash.set(dataset.dataset_hash);
        this.config.update((config) => ({
            ...config,
            dataset_label: dataset.dataset_label,
            dataset_hash: dataset.dataset_hash,
        }));
    }

    selectCheckpoint(name: string | null): void {
        this.selectedCheckpointName.set(name);
        if (name) {
            this.resumeConfig.update((config) => ({ ...config, checkpoint_name: name }));
        }
    }

    setConfig(config: TrainingConfig): void {
        this.config.set(config);
    }

    setResumeConfig(config: ResumeTrainingConfig): void {
        this.resumeConfig.set(config);
    }

    setActionLoading(loading: boolean): void {
        this.actionLoading.set(loading);
    }

    appendTrainingLog(message: string): void {
        this.trainingStatus.update((status) => ({
            ...status,
            log: [...(status.log ?? []), message],
        }));
    }

    clearTrainingLog(): void {
        this.trainingStatus.update((status) => ({
            ...status,
            log: ['Ready to start training...'],
        }));
    }

    setTrainingStatus(status: TrainingStatus): void {
        this.trainingStatus.set(status);
    }

    showNewTrainingWizardFor(datasetLabel: string): void {
        const dataset = this.processedDatasets().find((entry) => entry.dataset_label === datasetLabel);
        if (dataset) {
            this.selectDataset(dataset);
        }
        this.showNewTrainingWizard.set(true);
    }

    showResumeTrainingWizardFor(checkpointName: string): void {
        this.selectCheckpoint(checkpointName);
        this.showResumeTrainingWizard.set(true);
    }

    closeNewTrainingWizard(): void {
        this.showNewTrainingWizard.set(false);
    }

    closeResumeTrainingWizard(): void {
        this.showResumeTrainingWizard.set(false);
    }

    async startTraining(): Promise<Awaited<ReturnType<typeof startTraining>>> {
        return startTraining({
            ...this.config(),
            dataset_label: this.selectedDatasetLabel() ?? undefined,
            dataset_hash: this.selectedDatasetHash(),
        });
    }

    async resumeTraining(): Promise<Awaited<ReturnType<typeof resumeTraining>>> {
        return resumeTraining(this.resumeConfig());
    }

    async stopTraining(): Promise<Awaited<ReturnType<typeof stopTraining>>> {
        return stopTraining();
    }

    async deleteProcessedDataset(label: string): Promise<{ success: boolean; message: string }> {
        return deleteDataset(label);
    }

    async fetchDatasetMetadata(label: string): Promise<DatasetFullInfo> {
        return getTrainingDatasetInfo(label);
    }

    async deleteCheckpoint(name: string): Promise<{ success: boolean; error: string | null }> {
        return deleteCheckpoint(name);
    }

    async fetchCheckpointDetails(name: string): Promise<{ details: CheckpointFullDetails | null; error: string | null }> {
        return fetchCheckpointDetails(name);
    }

    openErrorModal(title: string, message: string): void {
        this.infoModalTitle.set(title);
        this.infoModalData.set({ Message: message });
        this.infoModalOpen.set(true);
    }

    openInfoModal(title: string, data: InfoModalData): void {
        this.infoModalTitle.set(title);
        this.infoModalData.set(data);
        this.infoModalOpen.set(true);
    }

    closeInfoModal(): void {
        this.infoModalOpen.set(false);
    }
}
