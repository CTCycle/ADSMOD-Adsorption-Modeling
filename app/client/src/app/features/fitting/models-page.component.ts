import { Component, computed, inject, signal } from '@angular/core';
import { CoreWorkspaceStore, OptimizationMethod } from '../../core/state/core-workspace.store';
import { HeaderTabsComponent } from '../../layout/header-tabs.component';
import type { ModelParameters } from '../../models/fitting.model';
import { NumberInputComponent } from '../../shared/components/number-input/number-input.component';
import { ModelCardComponent } from './model-card.component';

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
                                                <option value="" [selected]="store.selectedDatasetId() === null">{{ store.datasets().length === 0 ? 'No datasets available' : 'Select a dataset' }}</option>
                                                @for (dataset of store.datasets(); track dataset.id) {
                                                    <option [value]="dataset.id" [selected]="dataset.id === store.selectedDatasetId()">{{ dataset.name }}</option>
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
                                        <select id="fitting-weighting-control" class="select-input" [value]="store.weighting() ?? ''" [disabled]="!store.fittingConfiguration()" (change)="selectWeighting($event)">
                                            @for (weighting of weightingOptions(); track weighting) {
                                                <option [value]="weighting">{{ weightingLabel(weighting) }}</option>
                                            }
                                        </select>
                                    </div>
                                    <div class="control-group">
                                        <adsmod-number-input
                                            label="Max iterations"
                                            [value]="store.maxEvaluations()"
                                            [min]="store.fittingConfiguration()?.max_evaluations_bounds?.minimum"
                                            [max]="store.fittingConfiguration()?.max_evaluations_bounds?.maximum"
                                            [step]="1"
                                            [precision]="0"
                                            [disabled]="!store.fittingConfiguration()"
                                            (valueChange)="store.setMaxEvaluations($event)"
                                        />
                                    </div>
                                    <div class="control-group">
                                        <label class="field-label" for="fitting-optimization-control">Optimization method</label>
                                        <select
                                            id="fitting-optimization-control"
                                            [value]="store.optimizationMethod() ?? ''"
                                            [disabled]="!store.fittingConfiguration()"
                                            (change)="selectOptimizer($event)"
                                            class="select-input"
                                        >
                                            @for (option of optimizationOptions(); track option.value) {
                                                <option [value]="option.value">{{ option.label }}</option>
                                            }
                                        </select>
                                    </div>
                                    <div class="control-group">
                                        <div class="fitting-action-buttons">
                                            <button class="primary fitting-action-primary" type="button" [disabled]="!store.fittingConfiguration() || store.fittingRunning()" (click)="startFitting()">
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
                        @if (store.fittingConfigurationError()) {
                            <p class="status-text">Fitting configuration unavailable: {{ store.fittingConfigurationError() }}</p>
                        }
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
    protected readonly optimizationOptions = computed(() => (this.store.fittingConfiguration()?.supported_optimizers ?? []).map((value) => ({
        value,
        label: value === 'trf' ? 'Trust Region Reflective (TRF)' : 'Dogbox',
    })));
    protected readonly weightingOptions = computed(() => this.store.fittingConfiguration()?.weighting_options ?? []);
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
        const supported = this.store.fittingConfiguration()?.supported_optimizers ?? [];
        if (supported.includes(select.value as OptimizationMethod)) {
            this.store.setOptimizationMethod(select.value as OptimizationMethod);
        }
    }

    protected selectWeighting(event: Event): void {
        const value = (event.target as HTMLSelectElement).value;
        const supported = this.store.fittingConfiguration()?.weighting_options ?? [];
        if (supported.includes(value as 'unweighted' | 'inverse_sigma')) this.store.setWeighting(value as 'unweighted' | 'inverse_sigma');
    }

    protected weightingLabel(value: 'unweighted' | 'inverse_sigma'): string {
        return value === 'inverse_sigma' ? 'Inverse sigma (complete uncertainties)' : 'Unweighted';
    }

    protected updateModelConfig(modelName: string, config: ModelParameters): void {
        this.store.setModelParameters(modelName, config);
    }

    protected async startFitting(): Promise<void> {
        await this.store.startFitting();
    }
}
