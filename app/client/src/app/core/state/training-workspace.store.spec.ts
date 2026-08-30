import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TrainingConfig } from '../../models/training.model';
import { TrainingWorkspaceStore } from './training-workspace.store';

const TRAINING_CONFIG: TrainingConfig = {
    batch_size: 16,
    shuffle_dataset: true,
    max_buffer_size: 256,
    selected_model: 'SCADS Series',
    dropout_rate: 0.1,
    num_attention_heads: 2,
    num_encoders: 2,
    molecular_embedding_size: 64,
    epochs: 5,
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
};

describe('TrainingWorkspaceStore', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
        TestBed.resetTestingModule();
    });

    it('preserves the processed datasets returned by the ML service', async () => {
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/training/processed-datasets')) {
                return {
                    ok: true,
                    json: async () => ({
                        datasets: [
                            { dataset_label: ' zeolite_a ', dataset_hash: 'a1', train_samples: 10, validation_samples: 2 },
                            { dataset_label: 'zeolite_a', dataset_hash: 'a2', train_samples: 11, validation_samples: 3 },
                            { dataset_label: 'archived::old_set', dataset_hash: 'x', train_samples: 1, validation_samples: 1 },
                            { dataset_label: '', dataset_hash: 'z', train_samples: 1, validation_samples: 1 },
                            { dataset_label: 'nist_slice', dataset_hash: 'n1', train_samples: 20, validation_samples: 5 },
                        ],
                    }),
                };
            }
            if (url.endsWith('/training/checkpoints')) {
                return { ok: true, json: async () => ({ checkpoints: [] }) };
            }
            if (url.endsWith('/training/status')) {
                return {
                    ok: true,
                    json: async () => ({
                        is_training: false,
                        current_epoch: 0,
                        total_epochs: 0,
                        progress: 0,
                        metrics: {},
                        history: [],
                        log: [],
                    }),
                };
            }
            throw new Error(`Unhandled URL ${url}`);
        });

        const store = TestBed.inject(TrainingWorkspaceStore);
        await store.loadProcessedDatasets();

        expect(store.processedDatasets()).toEqual([
            { dataset_label: ' zeolite_a ', dataset_hash: 'a1', train_samples: 10, validation_samples: 2 },
            { dataset_label: 'zeolite_a', dataset_hash: 'a2', train_samples: 11, validation_samples: 3 },
            { dataset_label: 'archived::old_set', dataset_hash: 'x', train_samples: 1, validation_samples: 1 },
            { dataset_label: '', dataset_hash: 'z', train_samples: 1, validation_samples: 1 },
            { dataset_label: 'nist_slice', dataset_hash: 'n1', train_samples: 20, validation_samples: 5 },
        ]);
    });

    it('selects the matching dataset when opening the new training wizard', () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ checkpoints: [], datasets: [], is_training: false, current_epoch: 0, total_epochs: 0, progress: 0, metrics: {}, history: [], log: [] }),
        });

        const store = TestBed.inject(TrainingWorkspaceStore);
        store.setConfig(TRAINING_CONFIG);
        store.processedDatasets.set([
            { dataset_label: 'zeolite_curated_v2', dataset_hash: 'hash-1', train_samples: 100, validation_samples: 20 },
        ]);

        store.showNewTrainingWizardFor('zeolite_curated_v2');

        expect(store.showNewTrainingWizard()).toBe(true);
        expect(store.config()?.dataset_label).toBe('zeolite_curated_v2');
        expect(store.config()?.dataset_hash).toBe('hash-1');
    });

    it('resets the training log to the default ready message', () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ checkpoints: [], datasets: [], is_training: false, current_epoch: 0, total_epochs: 0, progress: 0, metrics: {}, history: [], log: [] }),
        });

        const store = TestBed.inject(TrainingWorkspaceStore);
        store.trainingStatus.set({
            is_training: false,
            current_epoch: 0,
            total_epochs: 0,
            progress: 0,
            metrics: {},
            history: [],
            log: ['previous', 'entries'],
        });

        store.clearTrainingLog();

        expect(store.trainingStatus().log).toEqual(['Ready to start training...']);
    });
});
