// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it, vi } from 'vitest';
import { buildCheckpointDetailsModalData, buildDatasetMetadataModalData } from './training-modal-data';
import { type CheckpointFullDetails, type DatasetFullInfo } from '../../../models/training.model';
import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { TrainingViewNavigationService } from './training-view-navigation.service';

// Training detail modal mapping

describe('training modal data mappers', () => {
    it('maps dataset metadata to stable modal labels', () => {
        const datasetInfo: DatasetFullInfo = {
            available: true,
            dataset_label: 'nist-sample',
            created_at: '2026-07-03T09:00:00',
            total_samples: 100,
            train_samples: 80,
            validation_samples: 20,
            sample_size: 0.5,
            validation_size: 0.2,
            min_measurements: 3,
            max_measurements: 40,
            smile_sequence_size: 128,
            max_pressure: 12.5,
            max_uptake: 7.25,
            smile_vocabulary_size: 32,
            adsorbent_vocabulary_size: 14,
            normalization_stats: { pressure: 1 },
        };

        expect(buildDatasetMetadataModalData(datasetInfo)).toEqual({
            'Dataset Label': 'nist-sample',
            'Created At': '2026-07-03T09:00:00',
            'Total Samples': 100,
            'Train Samples': 80,
            'Validation Samples': 20,
            'Sample Fraction': 0.5,
            'Validation Fraction': 0.2,
            'Min Measurements': 3,
            'Max Measurements': 40,
            'SMILES Length': 128,
            'Max Pressure': 12.5,
            'Max Uptake': 7.25,
            'SMILES Vocabulary': 32,
            'Adsorbents Count': 14,
            Normalization: { pressure: 1 },
        });
    });

    it('formats checkpoint details without changing modal labels', () => {
        const details: CheckpointFullDetails = {
            name: 'checkpoint-a',
            epochs_trained: 12,
            final_loss: 0.123456789,
            final_accuracy: null,
            is_compatible: true,
            created_at: '',
            configuration: null,
            metadata: null,
            history: null,
        };

        expect(buildCheckpointDetailsModalData(details)).toEqual({
            Name: 'checkpoint-a',
            'Epochs Trained': 12,
            'Final Loss': '0.123457',
            'Is Compatible': 'Yes',
            'Created At': 'Unknown',
        });
    });
});

// Training view navigation

describe('TrainingViewNavigationService', () => {
    it('navigates to the requested training view route', async () => {
        const router = {
            navigate: vi.fn().mockResolvedValue(true),
        };

        TestBed.configureTestingModule({
            providers: [
                TrainingViewNavigationService,
                { provide: Router, useValue: router },
            ],
        });

        const service = TestBed.inject(TrainingViewNavigationService);

        await expect(service.navigateTo('dashboard')).resolves.toBe(true);
        expect(router.navigate).toHaveBeenCalledWith(['/training', 'dashboard']);
    });
});
