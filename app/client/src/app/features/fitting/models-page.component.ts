import { Component, inject, signal } from '@angular/core';
import { ADSORPTION_MODELS } from '../../core/constants/adsorption-models';
import { CoreWorkspaceStore, OptimizationMethod } from '../../core/state/core-workspace.store';
import type { ModelParameters } from '../../models/fitting.model';
import { NumberInputComponent } from '../../shared/components/number-input/number-input.component';
import { ModelCardComponent } from './model-card.component';

interface OptimizationMethodOption {
    value: OptimizationMethod;
    label: string;
}

const OPTIMIZATION_METHOD_OPTIONS: readonly OptimizationMethodOption[] = [
    { value: 'LSS', label: 'Least Squares (LSS)' },
    { value: 'BFGS', label: 'BFGS' },
    { value: 'L-BFGS-B', label: 'L-BFGS-B' },
    { value: 'Nelder-Mead', label: 'Nelder-Mead' },
    { value: 'Powell', label: 'Powell' },
];

const parseOptimizationMethod = (value: string): OptimizationMethod | null => {
    const option = OPTIMIZATION_METHOD_OPTIONS.find((candidate) => candidate.value === value);
    return option?.value ?? null;
};

@Component({
    selector: 'adsmod-models-page',
    standalone: true,
    imports: [ModelCardComponent, NumberInputComponent],
    template: `
        <div class="models-page">
            <div class="fitting-config-panel">
                <div class="models-header-row">
                    <div class="models-title-block">
                        <h3>Fitting Configuration</h3>
                        <p>Configure the optimizer and run the fit.</p>
                    </div>
                </div>

                <div class="fitting-main-layout">
                    <div class="fitting-controls-column">
                        <div class="fitting-controls-row">
                            <div class="control-group">
                                <label class="field-label">Dataset</label>
                                <div class="fitting-dataset-row">
                                    <select
                                        [value]="store.selectedDataset() || ''"
                                        (change)="selectDataset($event)"
                                        class="select-input fitting-dataset-select"
                                    >
                                        <option value="">{{ store.availableDatasets().length === 0 ? 'No datasets available' : 'Select a dataset' }}</option>
                                        <option [value]="store.nistDatasetOption">NIST-A Collection</option>
                                        @for (datasetName of store.availableDatasets(); track datasetName) {
                                            <option [value]="datasetName">{{ datasetName }}</option>
                                        }
                                    </select>
                                </div>
                            </div>
                            <div class="control-group">
                                <adsmod-number-input
                                    label="Max iterations"
                                    [value]="store.maxIterations()"
                                    [min]="1"
                                    [max]="1000000"
                                    [step]="1"
                                    [precision]="0"
                                    (valueChange)="store.setMaxIterations($event)"
                                />
                            </div>
                            <div class="control-group">
                                <label class="field-label">Optimization method</label>
                                <select
                                    [value]="store.optimizationMethod()"
                                    (change)="selectOptimizer($event)"
                                    class="select-input"
                                >
                                    @for (option of optimizationOptions; track option.value) {
                                        <option [value]="option.value">{{ option.label }}</option>
                                    }
                                </select>
                            </div>
                            <div class="control-group">
                                <div class="fitting-action-buttons">
                                    <button class="primary fitting-action-primary" type="button" (click)="startFitting()">
                                        Start Fitting
                                    </button>
                                    <button class="secondary fitting-action-secondary" type="button" (click)="store.resetFittingStatus()">
                                        Reset Log
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="fitting-status-column">
                        <div class="fitting-status-box">
                            <div class="status-label">Fitting Log:</div>
                            <pre class="status-text">{{ store.fittingStatus() || 'Ready to start...' }}</pre>
                        </div>
                    </div>
                </div>
            </div>

            <hr class="section-separator" />

            <div class="models-grid-header">
                <h3>Select Adsorption Models</h3>
            </div>

            <div class="models-grid">
                @for (model of models; track model.id) {
                    <adsmod-model-card
                        [model]="model"
                        [isExpanded]="expandedModel() === model.id"
                        [isEnabled]="store.modelStates()[model.name].enabled"
                        [currentConfig]="store.modelStates()[model.name].config"
                        (toggle)="toggleExpanded($event)"
                        (enabledChange)="store.setModelEnabled(model.name, $event)"
                        (configChange)="updateModelConfig(model.name, $event)"
                    />
                }
            </div>
        </div>
    `,
})
export class ModelsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);
    protected readonly models = ADSORPTION_MODELS;
    protected readonly optimizationOptions = OPTIMIZATION_METHOD_OPTIONS;
    protected readonly expandedModel = signal<string | null>(null);

    protected toggleExpanded(modelId: string): void {
        this.expandedModel.set(this.expandedModel() === modelId ? null : modelId);
    }

    protected selectDataset(event: Event): void {
        const select = event.target as HTMLSelectElement;
        this.store.setSelectedDataset(select.value);
    }

    protected selectOptimizer(event: Event): void {
        const select = event.target as HTMLSelectElement;
        const method = parseOptimizationMethod(select.value);
        if (method) {
            this.store.setOptimizationMethod(method);
        }
    }

    protected updateModelConfig(modelName: string, config: ModelParameters): void {
        this.store.setModelParameters(modelName, config);
    }

    protected async startFitting(): Promise<void> {
        await this.store.startFitting();
    }
}
