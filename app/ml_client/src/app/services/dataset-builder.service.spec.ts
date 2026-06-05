import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
    deleteDatasetSource,
    fetchProcessedDatasets,
    getTrainingDatasetInfo,
} from './dataset-builder.service';

describe('dataset-builder.service', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('loads processed datasets from the expected endpoint', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                datasets: [{ dataset_label: 'set-a', total_samples: 40 }],
            }),
        });

        await expect(fetchProcessedDatasets()).resolves.toEqual({
            datasets: [{ dataset_label: 'set-a', total_samples: 40 }],
            error: null,
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/training/processed-datasets',
            expect.objectContaining({ method: 'GET', signal: expect.any(AbortSignal) })
        );
    });

    it('encodes dataset labels when requesting dataset info', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                available: true,
                dataset_label: 'Dataset 1/alpha',
                total_samples: 32,
            }),
        });

        await expect(getTrainingDatasetInfo('Dataset 1/alpha')).resolves.toEqual({
            available: true,
            dataset_label: 'Dataset 1/alpha',
            created_at: undefined,
            sample_size: undefined,
            validation_size: undefined,
            min_measurements: undefined,
            max_measurements: undefined,
            smile_sequence_size: undefined,
            max_pressure: undefined,
            max_uptake: undefined,
            total_samples: 32,
            train_samples: undefined,
            validation_samples: undefined,
            smile_vocabulary_size: undefined,
            adsorbent_vocabulary_size: undefined,
            normalization_stats: undefined,
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/training/dataset-info?dataset_label=Dataset%201%2Falpha',
            expect.objectContaining({ method: 'GET', signal: expect.any(AbortSignal) })
        );
    });

    it('passes both source and dataset name when deleting a dataset source', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'success',
                message: 'Removed.',
            }),
        });

        await expect(deleteDatasetSource('nist source', 'dataset one')).resolves.toEqual({
            success: true,
            message: 'Removed.',
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/training/dataset-source?source=nist%20source&dataset_name=dataset%20one',
            expect.objectContaining({ method: 'DELETE', signal: expect.any(AbortSignal) })
        );
    });
});
