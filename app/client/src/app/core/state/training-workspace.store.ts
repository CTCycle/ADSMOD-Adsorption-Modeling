import { Injectable, signal } from '@angular/core';
import type { InfoModalData } from '../../models/json.model';
import type {
    CheckpointFullDetails,
    DatasetFullInfo,
    CheckpointInfo,
    ProcessedDatasetInfo,
    ResumeTrainingConfig,
    TrainingConfig,
    TrainingConfiguration,
    TrainingStatus,
} from '../../models/training.model';
import { fetchTrainingConfiguration } from '../../services/system.service';
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

const INITIAL_TRAINING_STATUS: TrainingStatus = {
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    metrics: {},
    history: [],
    log: [],
};

@Injectable({ providedIn: 'root' })
export class TrainingWorkspaceStore {
    readonly trainingConfiguration = signal<TrainingConfiguration | null>(null);
    readonly trainingConfigurationError = signal<string | null>(null);
    readonly config = signal<TrainingConfig | null>(null);
    readonly checkpoints = signal<CheckpointInfo[]>([]);
    readonly isLoading = signal(false);
    readonly showNewTrainingWizard = signal(false);
    readonly showResumeTrainingWizard = signal(false);
    readonly resumeConfig = signal<ResumeTrainingConfig | null>(null);
    readonly processedDatasets = signal<ProcessedDatasetInfo[]>([]);
    readonly infoModalOpen = signal(false);
    readonly infoModalTitle = signal('');
    readonly infoModalData = signal<InfoModalData | null>(null);
    readonly trainingStatus = signal<TrainingStatus>(INITIAL_TRAINING_STATUS);
    readonly trainingStatusError = signal<string | null>(null);
    readonly actionLoading = signal(false);

    async refreshWorkspace(): Promise<void> {
        this.isLoading.set(true);
        await Promise.all([
            this.loadConfiguration(),
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
            this.processedDatasets.set(result.datasets);
        }
    }

    async loadConfiguration(): Promise<void> {
        const result = await fetchTrainingConfiguration();
        this.trainingConfigurationError.set(result.error);
        if (!result.data) {
            this.trainingConfiguration.set(null);
            this.config.set(null);
            this.resumeConfig.set(null);
            return;
        }
        this.trainingConfiguration.set(result.data);
        this.config.set({ ...result.data.defaults });
        this.resumeConfig.set({
            checkpoint_name: this.resumeConfig()?.checkpoint_name ?? '',
            additional_epochs: result.data.resume_defaults.additional_epochs,
        });
    }

    async checkStatus(): Promise<void> {
        const result = await getTrainingStatus();
        this.trainingStatusError.set(result.error);
        if (result.data) {
            this.trainingStatus.set(result.data);
        }
    }

    selectDataset(dataset: ProcessedDatasetInfo): void {
        this.config.update((config) => config ? ({
            ...config,
            dataset_label: dataset.dataset_label,
            dataset_hash: dataset.dataset_hash,
        }) : config);
    }

    selectCheckpoint(name: string | null): void {
        this.resumeConfig.update((config) => config ? ({ ...config, checkpoint_name: name ?? '' }) : config);
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
        const config = this.config();
        if (!config) {
            return { sessionId: '', message: this.trainingConfigurationError() || 'Training configuration is unavailable.', status: 'error' };
        }
        return startTraining(config);
    }

    async resumeTraining(): Promise<Awaited<ReturnType<typeof resumeTraining>>> {
        const config = this.resumeConfig();
        if (!config || !config.checkpoint_name) {
            return { sessionId: '', message: 'Select a checkpoint before resuming training.', status: 'error' };
        }
        return resumeTraining(config);
    }

    async stopTraining(): Promise<Awaited<ReturnType<typeof stopTraining>>> {
        return stopTraining();
    }

    async deleteProcessedDataset(label: string): Promise<{ success: boolean; message: string }> {
        return deleteDataset(label);
    }

    async fetchDatasetMetadata(label: string): Promise<{ data: DatasetFullInfo | null; error: string | null }> {
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
