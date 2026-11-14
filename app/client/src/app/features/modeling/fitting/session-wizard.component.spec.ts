// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from '../../../core/state/core-workspace.store';
import { ModelingStore } from '../../../core/state/modeling.store';
import type { DatasetSummary } from '../../../models/dataset.model';
import type { FittingConfiguration, FittingDatasetReadiness, ModelCatalogEntry } from '../../../models/fitting.model';
import { SessionWizardComponent } from './session-wizard.component';

// Session wizard step gating

const catalog: ModelCatalogEntry[] = [{
    key: 'langmuir',
    name: 'Langmuir',
    equation_latex: 'q',
    assumptions: '',
    pressure_requirement: 'absolute',
    requires_temperature: false,
    reference: '',
    enabled: true,
    parameters: [{ name: 'k', label: 'k', lower: 1, upper: 10, canonical_lower: 1, canonical_upper: 10, unit: '1', customized: false }],
}];

const configuration: FittingConfiguration = {
    status: 'success',
    supported_optimizers: ['trf'],
    default_optimizer: 'trf',
    default_max_evaluations: 100,
    max_evaluations_bounds: { minimum: 10, maximum: 1000 },
    weighting_options: ['unweighted'],
    default_weighting: 'unweighted',
    display_units: { pressure: ['bar'], uptake: ['mmol/g'], default_pressure: 'bar', default_uptake: 'mmol/g' },
};

function readiness(ineligible: boolean): FittingDatasetReadiness {
    return {
        status: 'success',
        dataset_id: 1,
        dataset_name: 'ds',
        experiment_count: 2,
        eligible_count: ineligible ? 1 : 2,
        ready: !ineligible,
        inverse_sigma_available: false,
        model_incompatibility: {},
        experiments: [
            { isotherm_id: 1, name: 'good', eligible: true, reason: null },
            { isotherm_id: 2, name: 'bad-run', eligible: !ineligible, reason: ineligible ? 'Only one point' : null },
        ],
    };
}

describe('SessionWizardComponent', () => {
    beforeEach(async () => {
        TestBed.resetTestingModule();
        vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 500, json: async () => ({}) })));
        await TestBed.configureTestingModule({ imports: [SessionWizardComponent], providers: [provideRouter([])] }).compileComponents();
    });

    async function setup(ready: FittingDatasetReadiness, models: string[]) {
        const store = TestBed.inject(ModelingStore);
        TestBed.inject(CoreWorkspaceStore).datasets.set([{ id: 1, name: 'ds', experiment_count: 2 } as DatasetSummary]);
        store.catalog.set(catalog);
        store.configuration.set(configuration);
        store.draft.set({ step: 0, name: 'Run', datasetId: 1, models, overrides: {}, optimizer: 'trf', weighting: 'unweighted', maxEvaluations: 100, skipIneligible: false });
        store.readiness.set(ready);
        const fixture = TestBed.createComponent(SessionWizardComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;
        const next = () => Array.from(root.querySelectorAll('button')).find((item) => item.textContent?.trim() === 'Next') as HTMLButtonElement;
        return { fixture, root, store, next };
    }

    it('blocks Next and lists the experiments that cannot be fitted', async () => {
        const { fixture, root, store, next } = await setup(readiness(true), ['langmuir']);

        expect(next().disabled).toBe(true);
        expect(root.textContent).toContain('bad-run');
        expect(root.textContent).toContain('Exclude the experiments that cannot be fitted');

        store.readiness.set(readiness(false));
        fixture.detectChanges();
        expect(next().disabled).toBe(false);
    });

    it('lets Next proceed once the ineligible experiments are explicitly excluded', async () => {
        const { fixture, root, store, next } = await setup(readiness(true), ['langmuir']);
        const checkbox = root.querySelector('.skip-option input') as HTMLInputElement;
        checkbox.click();
        fixture.detectChanges();

        expect(store.draft()?.skipIneligible).toBe(true);
        expect(next().disabled).toBe(false);
    });

    it('blocks Next on the models step until at least one model is selected', async () => {
        const { fixture, root, store, next } = await setup(readiness(false), []);
        store.updateDraft({ step: 1 });
        fixture.detectChanges();

        expect(next().disabled).toBe(true);
        expect(root.textContent).toContain('Select at least one adsorption model.');

        store.updateDraft({ models: ['langmuir'] });
        fixture.detectChanges();
        expect(next().disabled).toBe(false);
    });
});
