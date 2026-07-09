import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchCheckpoints, getTrainingStatus, startTraining } from './training.service';

describe('training.service', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('parses training status defensively and filters invalid history fields', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                is_training: true,
                current_epoch: 3,
                total_epochs: 10,
                progress: 30,
                poll_interval: 5,
                metrics: {
                    loss: 0.42,
                    val_loss: 0.51,
                    masked_r2: 0.89,
                    extra_metric: 999,
                },
                history: [
                    { epoch: 1, loss: 0.91, val_loss: 0.95, masked_r2: 0.33 },
                    { epoch: 2, val_accuracy: 0.78, ignored: 'x' },
                    { loss: 0.4 },
                    'bad row',
                ],
                log: ['step 1', 2, 'step 3'],
            }),
        });

        await expect(getTrainingStatus()).resolves.toEqual({
            is_training: true,
            current_epoch: 3,
            total_epochs: 10,
            progress: 30,
            poll_interval: 5,
            metrics: {
                loss: 0.42,
                val_loss: 0.51,
                masked_r2: 0.89,
            },
            history: [
                { epoch: 1, loss: 0.91, val_loss: 0.95, masked_r2: 0.33 },
                { epoch: 2, val_accuracy: 0.78 },
            ],
            log: ['step 1', 'step 3'],
            error: null,
        });
    });

    it('maps checkpoint responses into checkpoint info objects', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                checkpoints: [
                    { name: 'cp-1', epochs_trained: 4, final_loss: 0.12, is_compatible: true },
                    { final_accuracy: 0.88 },
                ],
            }),
        });

        await expect(fetchCheckpoints()).resolves.toEqual({
            checkpoints: [
                { name: 'cp-1', epochs_trained: 4, final_loss: 0.12, final_accuracy: null, is_compatible: true },
                { name: 'Unknown checkpoint', epochs_trained: null, final_loss: null, final_accuracy: 0.88, is_compatible: false },
            ],
            error: null,
        });
    });

    it('returns backend detail when training start fails', async () => {
        fetchMock.mockResolvedValue({
            ok: false,
            status: 400,
            json: async () => ({ detail: 'bad config' }),
        });

        await expect(startTraining({
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
        })).resolves.toEqual({
            sessionId: '',
            message: 'bad config',
            status: 'error',
        });
    });
});
