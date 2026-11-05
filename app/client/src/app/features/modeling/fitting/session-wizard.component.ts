// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { CoreWorkspaceStore } from '../../../core/state/core-workspace.store';
import { ModelingStore, WizardStep, boundsError, catalogBounds } from '../../../core/state/modeling.store';
import type { FittingWeighting, ModelBounds, ModelCatalogEntry, OptimizationMethod } from '../../../models/fitting.model';
import { EquationRendererComponent } from '../../../shared/components/equation-renderer/equation-renderer.component';
import { WizardNavigationFooterComponent } from '../../../shared/components/wizard-navigation-footer/wizard-navigation-footer.component';
import { ModelConfigFormComponent } from '../models/model-config-form.component';
import { OPTIMIZER_LABELS, WEIGHTING_LABELS, formatNumber } from './session-format';

const STEPS = ['Dataset', 'Models', 'Fitting settings', 'Review'] as const;

/** Four-step creation of a dataset-level fitting session. */
@Component({
    selector: 'adsmod-session-wizard',
    standalone: true,
    imports: [EquationRendererComponent, ModelConfigFormComponent, RouterLink, WizardNavigationFooterComponent],
    templateUrl: './session-wizard.component.html',
    styleUrl: './session-wizard.component.css',
})
export class SessionWizardComponent implements OnInit {
    protected readonly store = inject(ModelingStore);
    protected readonly workspace = inject(CoreWorkspaceStore);
    private readonly router = inject(Router);
    protected readonly steps = STEPS;
    protected readonly optimizerLabels = OPTIMIZER_LABELS;
    protected readonly weightingLabels = WEIGHTING_LABELS;
    protected readonly formatNumber = formatNumber;
    protected readonly starting = signal(false);
    protected readonly startError = signal<string | null>(null);

    protected readonly draft = this.store.draft;
    protected readonly step = computed(() => this.draft()?.step ?? 0);
    protected readonly dataset = computed(() => this.workspace.datasets().find((item) => item.id === this.draft()?.datasetId) ?? null);
    protected readonly ineligible = computed(() => this.store.readiness()?.experiments.filter((item) => !item.eligible) ?? []);
    protected readonly incompatibility = computed(() => this.store.readiness()?.model_incompatibility ?? {});
    protected readonly enabledModels = this.store.enabledModels;
    protected readonly disabledCount = computed(() => this.store.catalog().length - this.enabledModels().length);
    protected readonly selectedModels = computed(() => {
        const selected = new Set(this.draft()?.models ?? []);
        return this.enabledModels().filter((model) => selected.has(model.key));
    });
    protected readonly maxBounds = computed(() => this.store.configuration()?.max_evaluations_bounds ?? { minimum: 10, maximum: 1_000_000_000 });

    protected readonly stepError = computed<string | null>(() => {
        const draft = this.draft();
        if (!draft) return 'Preparing the session…';
        switch (draft.step) {
            case 0: {
                if (!draft.name.trim()) return 'Enter a session name.';
                if (draft.datasetId === null) return 'Select a dataset.';
                if (this.store.readinessLoading()) return 'Checking the dataset…';
                const readiness = this.store.readiness();
                if (!readiness) return this.store.readinessError() ?? 'The dataset could not be checked.';
                if (!readiness.experiment_count) return 'The dataset has no experiments.';
                if (!readiness.ready) return 'Resolve the experiments that cannot be fitted before continuing.';
                return null;
            }
            case 1: {
                if (!this.selectedModels().length) return 'Select at least one adsorption model.';
                for (const model of this.selectedModels()) {
                    const invalid = draft.overrides[model.key] ? boundsError(draft.overrides[model.key]) : null;
                    if (invalid) return `${model.name} · ${invalid}`;
                }
                return null;
            }
            case 2: {
                const { minimum, maximum } = this.maxBounds();
                if (!Number.isInteger(draft.maxEvaluations) || draft.maxEvaluations < minimum || draft.maxEvaluations > maximum) {
                    return `Max iterations must be a whole number between ${minimum} and ${maximum}.`;
                }
                if (draft.weighting === 'inverse_sigma' && !this.store.readiness()?.inverse_sigma_available) {
                    return 'Inverse-sigma weighting needs uncertainties for every measurement.';
                }
                return null;
            }
            default:
                return null;
        }
    });

    async ngOnInit(): Promise<void> {
        if (!this.store.draft()) {
            await this.store.beginDraft();
        } else {
            await Promise.all([this.store.loadConfiguration(), this.store.loadCatalog()]);
            const datasetId = this.store.draft()?.datasetId ?? null;
            if (datasetId !== null && !this.store.readiness()) await this.store.selectDraftDataset(datasetId);
        }
        // Drop selections that the catalog no longer offers (disabled since the draft began).
        const enabled = new Set(this.enabledModels().map((model) => model.key));
        this.store.updateDraft({ models: (this.store.draft()?.models ?? []).filter((key) => enabled.has(key)) });
    }

    protected goTo(step: number): void {
        if (step < this.step()) this.store.updateDraft({ step: step as WizardStep });
    }

    protected next(): void {
        if (this.stepError() || this.step() >= 3) return;
        this.store.updateDraft({ step: (this.step() + 1) as WizardStep });
    }

    protected previous(): void {
        if (this.step() > 0) this.store.updateDraft({ step: (this.step() - 1) as WizardStep });
    }

    protected cancel(): void {
        this.store.discardDraft();
        void this.router.navigate(['/modeling/fitting']);
    }

    protected setName(value: string): void {
        this.store.updateDraft({ name: value });
    }

    protected setDataset(value: string): void {
        const id = Number(value);
        void this.store.selectDraftDataset(Number.isInteger(id) && id > 0 ? id : null);
    }

    protected isSelected(key: string): boolean {
        return this.draft()?.models.includes(key) ?? false;
    }

    protected toggleModel(key: string, selected: boolean): void {
        const models = this.draft()?.models ?? [];
        const order = this.enabledModels().map((model) => model.key);
        const next = selected ? [...new Set([...models, key])] : models.filter((item) => item !== key);
        this.store.updateDraft({ models: order.filter((item) => next.includes(item)) });
    }

    protected selectAll(selected: boolean): void {
        this.store.updateDraft({ models: selected ? this.enabledModels().filter((model) => !this.incompatibility()[model.key]).map((model) => model.key) : [] });
    }

    protected sessionBounds(model: ModelCatalogEntry): ModelBounds {
        return { ...catalogBounds(model), ...this.draft()?.overrides[model.key] };
    }

    protected isOverridden(key: string, parameter: string): boolean {
        return Boolean(this.draft()?.overrides[key]?.[parameter]);
    }

    protected defaults(model: ModelCatalogEntry): ModelBounds {
        return catalogBounds(model);
    }

    protected hasOverride(key: string): boolean {
        return Boolean(this.draft()?.overrides[key]);
    }

    protected setOverride(model: ModelCatalogEntry, bounds: ModelBounds): void {
        const draft = this.draft();
        if (!draft) return;
        const { [model.key]: _previous, ...rest } = draft.overrides;
        // Keep only the parameters that differ from the global defaults.
        const defaults = catalogBounds(model);
        const changed = Object.fromEntries(Object.entries(bounds).filter(([name, value]) => value.lower !== defaults[name]?.lower || value.upper !== defaults[name]?.upper));
        this.store.updateDraft({ overrides: Object.keys(changed).length ? { ...rest, [model.key]: changed } : rest });
    }

    protected clearOverride(model: ModelCatalogEntry): void {
        this.setOverride(model, catalogBounds(model));
    }

    protected setWeighting(value: string): void {
        this.store.updateDraft({ weighting: value as FittingWeighting });
    }

    protected setOptimizer(value: string): void {
        this.store.updateDraft({ optimizer: value as OptimizationMethod });
    }

    protected setMaxEvaluations(value: string): void {
        this.store.updateDraft({ maxEvaluations: value.trim() === '' ? Number.NaN : Number(value) });
    }

    protected async start(): Promise<void> {
        if (this.starting() || this.step() !== 3) return;
        this.starting.set(true);
        this.startError.set(null);
        const result = await this.store.startDraft();
        this.starting.set(false);
        if (result.sessionId === null) {
            this.startError.set(result.error);
            return;
        }
        void this.store.loadSessions();
        await this.router.navigate(['/modeling/fitting', result.sessionId]);
    }
}
