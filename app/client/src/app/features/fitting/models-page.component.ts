import { Component, computed, inject, signal } from '@angular/core';
import { CoreWorkspaceStore, OptimizationMethod } from '../../core/state/core-workspace.store';
import { HeaderTabsComponent } from '../../layout/header-tabs.component';
import type {
    FittingResponse,
    ModelFitResult,
    ModelParameters,
} from '../../models/fitting.model';
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

                        @if (store.fittingResult(); as result) {
                            <section class="fitting-result-panel" aria-live="polite" aria-labelledby="fitting-result-title">
                                <div class="fitting-result-header">
                                    <div>
                                        <p class="fitting-result-kicker">Completed fit</p>
                                        <h4 id="fitting-result-title">{{ result.dataset_name }} · {{ result.experiment_name }}</h4>
                                    </div>
                                    <span class="fitting-result-status" [class.warning]="result.status === 'warning'">{{ result.status }}</span>
                                </div>

                                <div class="fitting-result-summary">
                                    <div class="fitting-result-summary-card">
                                        <span>Best model</span>
                                        <strong>{{ bestModelLabel(result) }}</strong>
                                    </div>
                                    <div class="fitting-result-summary-card">
                                        <span>Models fitted</span>
                                        <strong>{{ fittedModelCount(result) }} / {{ result.results.length }}</strong>
                                    </div>
                                    <div class="fitting-result-summary-card">
                                        <span>Observations</span>
                                        <strong>{{ result.observation_count }}</strong>
                                    </div>
                                </div>

                                <div class="fitting-result-table-wrap">
                                    <table class="fitting-result-table">
                                        <caption class="sr-only">Fitting results for {{ result.experiment_name }}</caption>
                                        <thead>
                                            <tr>
                                                <th scope="col">Model</th>
                                                <th scope="col">Status</th>
                                                <th scope="col">RMSE</th>
                                                <th scope="col">R²</th>
                                                <th scope="col">AICc</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            @for (fit of result.results; track fit.model) {
                                                <tr [class.fitting-result-row-best]="fit.model === result.best_model">
                                                    <th scope="row">
                                                        {{ fit.name }}
                                                        @if (fit.model === result.best_model) {
                                                            <span class="fitting-best-label">Best</span>
                                                        }
                                                    </th>
                                                    <td><span class="fitting-result-status" [class.warning]="fit.status === 'warning'">{{ fit.status }}</span></td>
                                                    <td>{{ formatMetric(fit.metrics.rmse) }}</td>
                                                    <td>{{ formatMetric(fit.metrics.r_squared) }}</td>
                                                    <td>{{ formatMetric(fit.metrics.aicc) }}</td>
                                                </tr>
                                            }
                                        </tbody>
                                    </table>
                                </div>
                            </section>
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

    protected bestModelLabel(result: FittingResponse): string {
        return this.bestModelResult(result)?.name ?? result.best_model ?? '—';
    }

    protected fittedModelCount(result: FittingResponse): number {
        return result.results.filter((fit) => fit.status !== 'failed').length;
    }

    protected formatMetric(value: number | null): string {
        return value === null || !Number.isFinite(value)
            ? '—'
            : new Intl.NumberFormat('en-US', { maximumSignificantDigits: 6 }).format(value);
    }

    private bestModelResult(result: FittingResponse): ModelFitResult | null {
        return result.results.find((fit) => fit.model === result.best_model) ?? null;
    }

    protected updateModelConfig(modelName: string, config: ModelParameters): void {
        this.store.setModelParameters(modelName, config);
    }

    protected async startFitting(): Promise<void> {
        await this.store.startFitting();
    }
}
