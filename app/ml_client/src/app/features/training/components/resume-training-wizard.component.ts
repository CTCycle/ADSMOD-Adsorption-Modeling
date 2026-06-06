import { Component, computed, input, output, signal } from '@angular/core';
import { FormControl, ReactiveFormsModule, Validators } from '@angular/forms';
import type { CheckpointInfo, ResumeTrainingConfig } from '../../../models/training.model';
import { NumberInputComponent } from '../../../shared/components/number-input/number-input.component';
import { WizardNavigationFooterComponent } from './wizard-navigation-footer.component';
import { WizardProgressIndicatorComponent } from './wizard-progress-indicator.component';

@Component({
    selector: 'adsmod-resume-training-wizard',
    standalone: true,
    imports: [ReactiveFormsModule, NumberInputComponent, WizardNavigationFooterComponent, WizardProgressIndicatorComponent],
    template: `
        <div class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="resume-training-wizard-title">
            <div class="wizard-modal">
                <div class="wizard-header">
                    <h4 id="resume-training-wizard-title">Resume Training Wizard</h4>
                    <p>Resuming from checkpoint: <strong>{{ selectedCheckpointName() }}</strong></p>
                    <adsmod-wizard-progress-indicator [currentPage]="currentPage()" [totalPages]="2" />
                </div>

                <div class="wizard-body">
                    @if (currentPage() === 0) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header"><span class="wizard-card-icon">CP</span><span>Configuration</span></div>
                                <p class="wizard-card-description">Set the number of additional epochs to train.</p>
                                <div class="wizard-card-body">
                                    <div class="wizard-settings-grid">
                                        <adsmod-number-input label="Additional Epochs" [value]="additionalEpochs()" [min]="1" [max]="100" [step]="1" [precision]="0" (valueChange)="additionalEpochsControl.setValue($event)" />
                                    </div>
                                    <div style="margin-top: 20px; padding: 15px; background-color: var(--slate-50); border-radius: 8px;">
                                        <strong>Checkpoint Details:</strong>
                                        <ul style="list-style: none; padding: 0; margin-top: 10px;">
                                            <li>Compatibility: {{ compatibilityLabel() }}</li>
                                            <li>Epochs Trained: {{ selectedCheckpoint()?.epochs_trained ?? '--' }}</li>
                                            <li>Final Loss: {{ selectedCheckpoint()?.final_loss?.toFixed(4) ?? '--' }}</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    } @else {
                        <div class="wizard-page">
                            <div class="wizard-summary">
                                <div class="wizard-summary-section">
                                    <h5>Resume Configuration</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Checkpoint</span><strong>{{ selectedCheckpointName() }}</strong>
                                        <span>Additional epochs</span><strong>{{ additionalEpochs() }}</strong>
                                        <span>Compatibility</span><strong>{{ compatibilityLabel() }}</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }
                </div>

                <adsmod-wizard-navigation-footer
                    [isLoading]="isLoading()"
                    [isFirstPage]="currentPage() === 0"
                    [isLastPage]="currentPage() === 1"
                    confirmIdleLabel="Confirm Resume"
                    confirmLoadingLabel="Resuming..."
                    (closed)="closed.emit()"
                    (next)="currentPage.set(1)"
                    (previous)="currentPage.set(0)"
                    (confirmed)="confirm()"
                />
            </div>
        </div>
    `,
})
export class ResumeTrainingWizardComponent {
    readonly checkpoints = input.required<CheckpointInfo[]>();
    readonly selectedCheckpointName = input.required<string>();
    readonly initialConfig = input.required<ResumeTrainingConfig>();
    readonly isLoading = input(false);
    readonly closed = output<void>();
    readonly confirmed = output<ResumeTrainingConfig>();
    protected readonly currentPage = signal(0);
    protected readonly additionalEpochsControl = new FormControl(10, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly additionalEpochs = computed(() => this.additionalEpochsControl.value);
    protected readonly selectedCheckpoint = computed(() => this.checkpoints().find((checkpoint) => checkpoint.name === this.selectedCheckpointName()) ?? null);
    protected readonly compatibilityLabel = computed(() => this.selectedCheckpoint()?.is_compatible ? 'Compatible' : this.selectedCheckpoint() ? 'Incompatible' : 'Unknown');

    constructor() {
        queueMicrotask(() => this.additionalEpochsControl.setValue(this.initialConfig().additional_epochs));
    }

    protected confirm(): void {
        this.confirmed.emit({
            checkpoint_name: this.selectedCheckpointName(),
            additional_epochs: this.additionalEpochsControl.value,
        });
    }
}
