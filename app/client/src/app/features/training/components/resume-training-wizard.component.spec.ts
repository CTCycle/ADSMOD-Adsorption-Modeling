import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResumeTrainingWizardComponent } from './resume-training-wizard.component';

describe('ResumeTrainingWizardComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [ResumeTrainingWizardComponent],
        }).compileComponents();
    });

    it('shows checkpoint compatibility and emits the resume payload on confirmation', async () => {
        const fixture = TestBed.createComponent(ResumeTrainingWizardComponent);
        const confirmSpy = vi.spyOn(fixture.componentInstance.confirmed, 'emit');
        fixture.componentRef.setInput('checkpoints', [
            { name: 'cp-1', epochs_trained: 12, final_loss: 0.42, final_accuracy: 0.91, is_compatible: true },
        ]);
        fixture.componentRef.setInput('selectedCheckpointName', 'cp-1');
        fixture.componentRef.setInput('initialConfig', { checkpoint_name: 'cp-1', additional_epochs: 7 });
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.textContent).toContain('Compatible');
        expect(root.textContent).toContain('12');

        root.querySelectorAll<HTMLButtonElement>('.wizard-footer button').item(root.querySelectorAll('.wizard-footer button').length - 1)?.click();
        fixture.detectChanges();
        root.querySelectorAll<HTMLButtonElement>('.wizard-footer button').item(root.querySelectorAll('.wizard-footer button').length - 1)?.click();

        expect(confirmSpy).toHaveBeenCalledWith({
            checkpoint_name: 'cp-1',
            additional_epochs: 7,
        });
    });
});
