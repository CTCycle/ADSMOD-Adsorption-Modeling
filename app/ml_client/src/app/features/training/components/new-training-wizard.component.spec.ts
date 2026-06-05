import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TrainingConfig } from '../../../models/training.model';
import { NewTrainingWizardComponent } from './new-training-wizard.component';

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
    custom_name: '',
};

describe('NewTrainingWizardComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [NewTrainingWizardComponent],
        }).compileComponents();
    });

    it('navigates through all wizard pages and confirms with the selected dataset label', async () => {
        const fixture = TestBed.createComponent(NewTrainingWizardComponent);
        const confirmSpy = vi.spyOn(fixture.componentInstance.confirmed, 'emit');
        fixture.componentRef.setInput('selectedDatasetLabel', 'dataset-alpha');
        fixture.componentRef.setInput('initialConfig', TRAINING_CONFIG);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        for (let index = 0; index < 4; index += 1) {
            root.querySelectorAll<HTMLButtonElement>('.wizard-footer button').item(root.querySelectorAll('.wizard-footer button').length - 1)?.click();
            fixture.detectChanges();
        }

        expect(root.textContent).toContain('Training Name');
        root.querySelectorAll<HTMLButtonElement>('.wizard-footer button').item(root.querySelectorAll('.wizard-footer button').length - 1)?.click();

        expect(confirmSpy).toHaveBeenCalledWith({
            ...TRAINING_CONFIG,
            dataset_label: 'dataset-alpha',
        });
    });

    it('disables wizard footer actions while loading', async () => {
        const fixture = TestBed.createComponent(NewTrainingWizardComponent);
        fixture.componentRef.setInput('selectedDatasetLabel', 'dataset-alpha');
        fixture.componentRef.setInput('initialConfig', TRAINING_CONFIG);
        fixture.componentRef.setInput('isLoading', true);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const buttons = Array.from((fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>('.wizard-footer button'));
        expect(buttons.every((button) => button.disabled)).toBe(true);
    });
});
