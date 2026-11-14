// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { FittingConfiguration, ModelCatalogEntry } from '../../models/fitting.model';
import { ModelingStore, boundsError } from './modeling.store';

// Modeling store

const fetchMock = vi.fn();

function response(body: unknown, ok = true) {
    return { ok, status: ok ? 200 : 400, json: async () => body };
}

function model(key: string, enabled: boolean): ModelCatalogEntry {
    return {
        key,
        name: key,
        equation_latex: 'q',
        assumptions: '',
        pressure_requirement: 'absolute',
        requires_temperature: false,
        reference: '',
        enabled,
        parameters: [{ name: 'k', label: 'k', lower: 1, upper: 10, canonical_lower: 1, canonical_upper: 10, unit: '1', customized: false }],
    };
}

const configuration: FittingConfiguration = {
    status: 'success',
    supported_optimizers: ['trf', 'dogbox'],
    default_optimizer: 'dogbox',
    default_max_evaluations: 500,
    max_evaluations_bounds: { minimum: 10, maximum: 1000 },
    weighting_options: ['unweighted'],
    default_weighting: 'unweighted',
    display_units: { pressure: ['bar'], uptake: ['mmol/g'], default_pressure: 'bar', default_uptake: 'mmol/g' },
};

function requestedUrls(): URL[] {
    return fetchMock.mock.calls.map(([input]) => new URL(String(input), 'http://localhost'));
}

describe('boundsError', () => {
    it.each([
        [{ k: { lower: Number.NaN, upper: 1 } }, 'enter numeric bounds'],
        [{ k: { lower: 0, upper: 1 } }, 'greater than zero'],
        [{ k: { lower: -1, upper: 1 } }, 'greater than zero'],
        [{ k: { lower: 5, upper: 5 } }, 'smaller than the upper bound'],
        [{ k: { lower: 9, upper: 2 } }, 'smaller than the upper bound'],
    ])('rejects %j', (bounds, message) => {
        expect(boundsError(bounds)).toContain(message);
    });

    it('accepts strictly positive ascending bounds', () => {
        expect(boundsError({ k: { lower: 1e-6, upper: 1e3 }, q: { lower: 1, upper: 2 } })).toBeNull();
    });
});

describe('ModelingStore', () => {
    beforeEach(() => {
        TestBed.resetTestingModule();
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('resets to page 1 and requests the filtered, sorted page when the session query changes', async () => {
        fetchMock.mockResolvedValue(response({ items: [], total: 0, page: 1, page_size: 20 }));
        const store = TestBed.inject(ModelingStore);
        store.sessionQuery.update((query) => ({ ...query, page: 4 }));

        store.updateSessionQuery({ query: ' co2 ', statuses: ['failed', 'warning'], datasetId: 7, sort: 'name', direction: 'asc' });
        await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());

        expect(store.sessionQuery().page).toBe(1);
        const url = requestedUrls()[0];
        expect(url.pathname).toMatch(/\/fitting\/sessions$/);
        expect(url.searchParams.get('page')).toBe('1');
        expect(url.searchParams.get('query')).toBe('co2');
        expect(url.searchParams.getAll('status')).toEqual(['failed', 'warning']);
        expect(url.searchParams.get('dataset_id')).toBe('7');
        expect(url.searchParams.get('sort')).toBe('name');
        expect(url.searchParams.get('direction')).toBe('asc');
    });

    it('keeps an explicit page in the patch instead of resetting it', async () => {
        fetchMock.mockResolvedValue(response({ items: [], total: 100, page: 3, page_size: 20 }));
        const store = TestBed.inject(ModelingStore);

        store.updateSessionQuery({ page: 3 });
        await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());

        expect(requestedUrls()[0].searchParams.get('page')).toBe('3');
    });

    it('starts a session with only the overridden bounds of selected models and the default display units', async () => {
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/system/configuration')) return response(configuration);
            if (url.includes('/fitting/models')) return response({ status: 'success', models: [model('langmuir', true), model('toth', true), model('off', false)] });
            if (url.endsWith('/fitting/sessions')) return response({ session_id: 12 });
            throw new Error(`Unhandled URL ${url}`);
        });
        const store = TestBed.inject(ModelingStore);
        await store.beginDraft();

        expect(store.draft()).toMatchObject({ models: ['langmuir', 'toth'], optimizer: 'dogbox', maxEvaluations: 500 });

        store.updateDraft({
            name: '  Screening ',
            datasetId: 3,
            models: ['langmuir'],
            overrides: { langmuir: { k: { lower: 2, upper: 5 } }, toth: { k: { lower: 3, upper: 4 } } },
        });
        const result = await store.startDraft();

        expect(result).toEqual({ sessionId: 12, error: null });
        const post = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/fitting/sessions'));
        expect(JSON.parse(post?.[1].body)).toEqual({
            name: 'Screening',
            dataset_id: 3,
            models: ['langmuir'],
            optimizer: 'dogbox',
            weighting: 'unweighted',
            max_evaluations: 500,
            skip_ineligible: false,
            parameter_overrides: { langmuir: { k: { lower: 2, upper: 5 } } },
            display_units: { pressure: 'bar', uptake: 'mmol/g' },
        });
        expect(store.draft()).toBeNull();
    });
});
