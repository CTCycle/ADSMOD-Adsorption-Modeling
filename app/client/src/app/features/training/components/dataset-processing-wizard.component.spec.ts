// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';
import type { DatasetBuildConfig } from '../../../models/dataset-build.model';
import type { DatasetSourceInfo } from '../../../models/training.model';
import { DatasetProcessingWizardComponent } from './dataset-processing-wizard.component';

const SELECTED_DATASET: DatasetSourceInfo = {
    source: 'uploaded',
    dataset_name: 'dataset-alpha',
    display_name: 'Dataset Alpha',
    row_count: 21,
    dataset_id: 1,
};

const INITIAL_CONFIG: Partial<DatasetBuildConfig> = {
    sample_size: 0.8,
    validation_size: 0.2,
    smile_sequence_size: 32,
    min_measurements: 3,
    max_measurements: 48,
    max_pressure: 5000,
    max_uptake: 12,
};

describe('DatasetProcessingWizardComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [DatasetProcessingWizardComponent],
        }).compileComponents();
    });

    it('applies the supplied processing defaults after required inputs initialize', async () => {
        const fixture = TestBed.createComponent(DatasetProcessingWizardComponent);
        fixture.componentRef.setInput('selectedDatasets', [SELECTED_DATASET]);
        fixture.componentRef.setInput('initialConfig', INITIAL_CONFIG);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const values = Array.from(
            (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLInputElement>('.wizard-settings-grid input'),
        ).map((input) => Number(input.value));

        expect(values).toEqual([
            INITIAL_CONFIG.sample_size,
            INITIAL_CONFIG.validation_size,
            INITIAL_CONFIG.smile_sequence_size,
            INITIAL_CONFIG.min_measurements,
            INITIAL_CONFIG.max_measurements,
            INITIAL_CONFIG.max_pressure,
            INITIAL_CONFIG.max_uptake,
        ]);
    });

    it('shows updated processing values in the build summary', async () => {
        const fixture = TestBed.createComponent(DatasetProcessingWizardComponent);
        fixture.componentRef.setInput('selectedDatasets', [SELECTED_DATASET]);
        fixture.componentRef.setInput('initialConfig', INITIAL_CONFIG);
        fixture.detectChanges();
        await fixture.whenStable();

        const root = fixture.nativeElement as HTMLElement;
        const sampleSize = root.querySelector<HTMLInputElement>('.wizard-settings-grid input');
        expect(sampleSize).not.toBeNull();

        sampleSize!.value = '0.1';
        sampleSize!.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();

        const nextButton = Array.from(root.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Next →');
        expect(nextButton).toBeDefined();
        nextButton!.click();
        fixture.detectChanges();

        const values = Array.from(root.querySelectorAll('.wizard-summary-grid strong')).map((element) => element.textContent?.trim());
        expect(values).toEqual(['0.1', '0.2', '32', '3', '48', '5000', '12']);
    });
});
