import { Component, computed, inject, signal } from '@angular/core';
import { CoreWorkspaceStore, OptimizationMethod } from '../../core/state/core-workspace.store';
import { HeaderTabsComponent } from '../../layout/header-tabs.component';
import type { ModelParameters } from '../../models/fitting.model';
import { NumberInputComponent } from '../../shared/components/number-input/number-input.component';
import { ModelCardComponent } from './model-card.component';

interface OptimizationMethodOption {
    value: OptimizationMethod;
    label: string;
}

const OPTIMIZATION_METHOD_OPTIONS: readonly OptimizationMethodOption[] = [
    { value: 'trf', label: 'Trust Region Reflective (TRF)' },
    { value: 'dogbox', label: 'Dogbox' },
];

const parseOptimizationMethod = (value: string): OptimizationMethod | null => {
    const option = OPTIMIZATION_METHOD_OPTIONS.find((candidate) => candidate.value === value);
    return option?.value ?? null;
};

@Component({
    selector: 'adsmod-models-page',
    standalone: true,
    imports: [ModelCardComponent, NumberInputComponent, HeaderTabsComponent],
    template: `
        <div class="route-workspace route-workspace-fitting">
            <aside class="route-rail route-rail-fitting" aria-label="Fitting overview">
                <div class="route-rail-brand">
                    <div class="route-rail-logo" aria-hidden="true">AD</div>
                    <div class="route-rail-wordmark">ADSMOD</div>
                </div>
                <div class="route-rail-copy">
                    <h1>Fitting</h1>
                    <p>Configure the optimizer and run the fit.</p>
                </div>
            </aside>

            <section class="route-canvas route-canvas-fitting">
                <div class="route-tabs-row">
                    <adsmod-header-tabs />
                </div>

                <div class="models-page">
                    <div class="fitting-config-panel">
                        <div class="models-header-row">
                            <div class="models-title-block">
                                <h3>Fitting Configuration</h3>
                            </div>
                        </div>

                        <div class="fitting-main-layout">
                            <div class="fitting-controls-column">
                                <div class="fitting-controls-row">
                                    <div class="control-group">
                                        <label class="field-label" for="fitting-dataset-control">Dataset</label>
                                        <div class="fitting-dataset-row">
                                            <select
                                                id="fitting-dataset-control"
                                                [value]="store.selectedDatasetId() || ''"
                                                (change)="selectDataset($event)"
                                                class="select-input fitting-dataset-select"
                                            >
                                                <option value="">{{ store.datasets().length === 0 ? 'No datasets available' : 'Select a dataset' }}</option>
                                                @for (dataset of store.datasets(); track dataset.id) {
                                                    <option [value]="dataset.id">{{ dataset.name }}</option>
                                                }
                                            </select>
                                        </div>
                                    </div>
                                    <div class="control-group">
                                        <label class="field-label" for="fitting-experiment-control">Experiment / isotherm</label>
                                        <select
                                            id="fitting-experiment-control"
                                            class="select-input"
                                            [value]="store.selectedExperimentId() || ''"
                                            [disabled]="store.experimentsLoading() || !store.experiments().length"
                                            (change)="selectExperiment($event)"
                                        >
                                            <option value="" [selected]="store.selectedExperimentId() === null">
                                                {{ store.experimentsLoading() ? 'Loading experiments…' : store.experiments().length ? 'Select an experiment' : 'Select a dataset first' }}
                                            </option>
                                            @for (experiment of store.experiments(); track experiment.id) {
                                                <option [value]="experiment.id" [selected]="experiment.id === store.selectedExperimentId()">
                                                    {{ experiment.name }} · {{ experiment.observation_count }} points
                                                </option>
                                            }
                                        </select>
                                    </div>
                                    <div class="control-group">
                                        <label class="field-label" for="fitting-weighting-control">Weighting</label>
                                        <select id="fitting-weighting-control" class="select-input" [value]="store.weighting()" (change)="selectWeighting($event)">
                                            <option value="unweighted">Unweighted</option>
                                            <option value="inverse_sigma">Inverse sigma (complete uncertainties)</option>
                                        </select>
                                    </div>
                                    <div class="control-group">
                                        <adsmod-number-input
                                            label="Max iterations"
                                            [value]="store.maxEvaluations()"
                                            [min]="1"
                                            [max]="1000000"
                                            [step]="1"
                                            [precision]="0"
                                            (valueChange)="store.setMaxEvaluations($event)"
                                        />
                                    </div>
                                    <div class="control-group">
                                        <label class="field-label" for="fitting-optimization-control">Optimization method</label>
                                        <select
                                            id="fitting-optimization-control"
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
                        @for (model of models(); track model.id) {
                            <adsmod-model-card
                                [model]="model"
                                [isExpanded]="expandedModel() === model.id"
                                [isEnabled]="store.modelStates()[model.id]?.enabled ?? false"
                                [currentConfig]="store.modelStates()[model.id]?.config ?? {}"
                                (toggle)="toggleExpanded($event)"
                                (enabledChange)="store.setModelEnabled(model.id, $event)"
                                (configChange)="updateModelConfig(model.id, $event)"
                            />
                        }
                    </div>
                </div>
            </section>
        </div>
    `,
})
export class ModelsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);
    protected readonly models = computed(() => (this.store.modelCatalog()?.models ?? []).map((model) => ({ id: model.key, name: model.name, shortDescription: model.assumptions, equationLatex: model.equation_latex, parameterDefaults: Object.fromEntries(model.parameters.map((parameter) => [parameter.name, [parameter.lower, parameter.upper] as [number, number]])) })));
    protected readonly optimizationOptions = OPTIMIZATION_METHOD_OPTIONS;
    protected readonly expandedModel = signal<string | null>(null);

    protected toggleExpanded(modelId: string): void {
        this.expandedModel.set(this.expandedModel() === modelId ? null : modelId);
    }

    protected selectDataset(event: Event): void {
        const select = event.target as HTMLSelectElement;
        const id = Number(select.value);
        void this.store.selectDataset(Number.isFinite(id) && id > 0 ? id : null);
    }

    protected selectExperiment(event: Event): void {
        const id = Number((event.target as HTMLSelectElement).value);
        this.store.setSelectedExperiment(Number.isFinite(id) && id > 0 ? id : null);
    }

    protected selectOptimizer(event: Event): void {
        const select = event.target as HTMLSelectElement;
        const method = parseOptimizationMethod(select.value);
        if (method) {
            this.store.setOptimizationMethod(method);
        }
    }

    protected selectWeighting(event: Event): void {
        const value = (event.target as HTMLSelectElement).value;
        if (value === 'unweighted' || value === 'inverse_sigma') this.store.weighting.set(value);
    }

    protected updateModelConfig(modelName: string, config: ModelParameters): void {
        this.store.setModelParameters(modelName, config);
    }

    protected async startFitting(): Promise<void> {
        await this.store.startFitting();
    }
}
