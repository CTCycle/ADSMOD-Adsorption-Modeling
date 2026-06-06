import { Component, computed, input, output, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import type { DatasetBuildConfig, DatasetSelection } from '../../../models/dataset-build.model';
import type { DatasetSourceInfo } from '../../../models/training.model';
import { NumberInputComponent } from '../../../shared/components/number-input/number-input.component';
import { WizardProgressIndicatorComponent } from './wizard-progress-indicator.component';

const buildDatasetKey = (dataset: DatasetSourceInfo): string => `${dataset.source}:${dataset.dataset_name}`;

@Component({
    selector: 'adsmod-dataset-processing-wizard',
    standalone: true,
    imports: [ReactiveFormsModule, NumberInputComponent, WizardProgressIndicatorComponent],
    template: `
        <div class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="dataset-processing-wizard-title">
            <div class="wizard-modal">
                <div class="wizard-header">
                    <h4 id="dataset-processing-wizard-title">Dataset Processing Wizard</h4>
                    <p>Configure processing settings for your selected datasets.</p>
                    <adsmod-wizard-progress-indicator [currentPage]="currentPage()" [totalPages]="2" />
                </div>

                <div class="wizard-body">
                    @if (currentPage() === 0) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header">
                                    <span class="wizard-card-icon">⚙️</span>
                                    <span>Processing Settings</span>
                                </div>
                                <p class="wizard-card-description">
                                    Configure the parameters for dataset preprocessing. These settings control
                                    how raw adsorption data is filtered, sampled, and split for training.
                                </p>
                                <div class="wizard-card-body">
                                    <div class="wizard-settings-grid">
                                        <adsmod-number-input label="Sample Size" [value]="sampleSize()" [min]="0.01" [max]="1" [step]="0.01" [precision]="2" (valueChange)="sampleSizeControl.setValue($event)" />
                                        <adsmod-number-input label="Validation %" [value]="validationSize()" [min]="0.05" [max]="0.5" [step]="0.05" [precision]="2" (valueChange)="validationSizeControl.setValue($event)" />
                                        <adsmod-number-input label="SMILE Length" [value]="smileSequenceSize()" [min]="5" [max]="100" [step]="5" [precision]="0" (valueChange)="smileSequenceSizeControl.setValue($event)" />
                                        <adsmod-number-input label="Min Measurements" [value]="minMeasurements()" [min]="1" [max]="50" [step]="1" [precision]="0" (valueChange)="minMeasurementsControl.setValue($event)" />
                                        <adsmod-number-input label="Max Measurements" [value]="maxMeasurements()" [min]="5" [max]="500" [step]="5" [precision]="0" (valueChange)="maxMeasurementsControl.setValue($event)" />
                                        <adsmod-number-input label="Max Pressure (kPa)" [value]="maxPressure()" [min]="100" [max]="100000" [step]="1000" [precision]="0" (valueChange)="maxPressureControl.setValue($event)" />
                                        <adsmod-number-input label="Max Uptake (mol/g)" [value]="maxUptake()" [min]="1" [max]="1000" [step]="1" [precision]="1" (valueChange)="maxUptakeControl.setValue($event)" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    } @else {
                        <div class="wizard-page">
                            <div class="wizard-card" style="margin-bottom: 1rem; border: 1px solid var(--primary-200);">
                                <div class="wizard-card-header">
                                    <span class="wizard-card-icon">🏷️</span>
                                    <span>Dataset Name</span>
                                </div>
                                <div class="wizard-card-body">
                                    <div style="padding: 0.5rem 0;">
                                        <label class="field-label" style="margin-bottom: 0.5rem; display: block;">Custom Name</label>
                                        <input
                                            type="text"
                                            [formControl]="datasetNameControl"
                                            placeholder="e.g. my_dataset_v1"
                                            class="number-input-field"
                                            style="width: 100%; text-align: left; padding: 0.5rem 0.75rem; font-size: 0.95rem; border-radius: 8px; border: 1px solid var(--slate-300); height: auto;"
                                        />
                                    </div>
                                </div>
                            </div>

                            <div class="wizard-summary">
                                <div class="wizard-summary-section">
                                    <h5>Selected Datasets</h5>
                                    <ul>
                                        @for (dataset of selectedDatasets(); track datasetKey(dataset)) {
                                            <li>
                                                <strong>{{ dataset.display_name }}</strong>
                                                <span class="wizard-summary-meta">{{ dataset.source }} • {{ dataset.row_count }} rows</span>
                                            </li>
                                        }
                                    </ul>
                                </div>
                                <div class="wizard-summary-section">
                                    <h5>Processing Settings</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Sample size</span><strong>{{ sampleSize() }}</strong>
                                        <span>Validation split</span><strong>{{ validationSize() }}</strong>
                                        <span>SMILE length</span><strong>{{ smileSequenceSize() }}</strong>
                                        <span>Min measurements</span><strong>{{ minMeasurements() }}</strong>
                                        <span>Max measurements</span><strong>{{ maxMeasurements() }}</strong>
                                        <span>Max pressure (kPa)</span><strong>{{ maxPressure() }}</strong>
                                        <span>Max uptake (mol/g)</span><strong>{{ maxUptake() }}</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }
                </div>

                <div class="wizard-footer">
                    <button class="secondary" type="button" (click)="closed.emit()">Cancel</button>
                    @if (currentPage() === 0) {
                        <button class="primary" type="button" (click)="currentPage.set(1)">Next →</button>
                    } @else {
                        <button class="secondary" type="button" (click)="currentPage.set(0)">← Previous</button>
                        <button class="primary" type="button" [disabled]="form.invalid || selectedDatasets().length === 0" (click)="submit()">✓ Build Dataset</button>
                    }
                </div>
            </div>
        </div>
    `,
})
export class DatasetProcessingWizardComponent {
    readonly selectedDatasets = input.required<DatasetSourceInfo[]>();
    readonly closed = output<void>();
    readonly buildStarted = output<DatasetBuildConfig>();
    protected readonly currentPage = signal(0);

    protected readonly sampleSizeControl = new FormControl(1, { nonNullable: true, validators: [Validators.min(0.01), Validators.max(1)] });
    protected readonly validationSizeControl = new FormControl(0.2, { nonNullable: true, validators: [Validators.min(0.05), Validators.max(0.5)] });
    protected readonly minMeasurementsControl = new FormControl(1, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly maxMeasurementsControl = new FormControl(30, { nonNullable: true, validators: [Validators.min(5)] });
    protected readonly smileSequenceSizeControl = new FormControl(20, { nonNullable: true, validators: [Validators.min(5)] });
    protected readonly maxPressureControl = new FormControl(10000, { nonNullable: true, validators: [Validators.min(100)] });
    protected readonly maxUptakeControl = new FormControl(20, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly datasetNameControl = new FormControl(this.defaultDatasetName(), { nonNullable: true });
    protected readonly form = new FormGroup({
        sample_size: this.sampleSizeControl,
        validation_size: this.validationSizeControl,
        min_measurements: this.minMeasurementsControl,
        max_measurements: this.maxMeasurementsControl,
        smile_sequence_size: this.smileSequenceSizeControl,
        max_pressure: this.maxPressureControl,
        max_uptake: this.maxUptakeControl,
        dataset_label: this.datasetNameControl,
    });

    protected readonly sampleSize = computed(() => this.sampleSizeControl.value);
    protected readonly validationSize = computed(() => this.validationSizeControl.value);
    protected readonly minMeasurements = computed(() => this.minMeasurementsControl.value);
    protected readonly maxMeasurements = computed(() => this.maxMeasurementsControl.value);
    protected readonly smileSequenceSize = computed(() => this.smileSequenceSizeControl.value);
    protected readonly maxPressure = computed(() => this.maxPressureControl.value);
    protected readonly maxUptake = computed(() => this.maxUptakeControl.value);

    protected datasetKey(dataset: DatasetSourceInfo): string {
        return buildDatasetKey(dataset);
    }

    protected submit(): void {
        const datasets: DatasetSelection[] = this.selectedDatasets().map((dataset) => ({
            source: dataset.source,
            dataset_name: dataset.dataset_name,
        }));
        this.closed.emit();
        this.buildStarted.emit({
            sample_size: this.sampleSizeControl.value,
            validation_size: this.validationSizeControl.value,
            min_measurements: this.minMeasurementsControl.value,
            max_measurements: this.maxMeasurementsControl.value,
            smile_sequence_size: this.smileSequenceSizeControl.value,
            max_pressure: this.maxPressureControl.value,
            max_uptake: this.maxUptakeControl.value,
            datasets,
            dataset_label: this.datasetNameControl.value || undefined,
        });
    }

    private defaultDatasetName(): string {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        return `dataset_${timestamp}`;
    }
}
