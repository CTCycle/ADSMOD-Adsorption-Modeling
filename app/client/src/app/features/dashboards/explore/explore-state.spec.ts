// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest';
import type { ExploreState } from './explore-state';
import { chartFilter, DEFAULT_EXPLORE_STATE, stateFromQueryParams, stateToQueryParams } from './explore-state';

describe('explore state URL serialisation', () => {
    it('round-trips a configured view through query parameters', () => {
        const state: ExploreState = {
            ...DEFAULT_EXPLORE_STATE,
            filter: {
                ...DEFAULT_EXPLORE_STATE.filter,
                dataset_sources: ['nist'],
                adsorbent_ids: [4, 9],
                adsorbate_ids: [7],
                temperature_min_k: 273.15,
                pressure_basis: 'relative',
                include_mixtures: true,
            },
            selectedIds: [11, 12],
            mode: 'uptake',
            colorBy: 'material',
            logPressure: true,
            uptakeUnit: 'mmol/g',
            targetPressure: 0.3,
            method: 'nearest',
            tolerance: 0.1,
        };

        const params = stateToQueryParams(state);
        const encoded = Object.fromEntries(Object.entries(params).filter(([, value]) => value !== null).map(([key, value]) => [key, String(value)]));

        expect(stateFromQueryParams(encoded)).toEqual(state);
        expect(params['punit']).toBeNull();
    });

    it('falls back to defaults for missing or invalid parameters', () => {
        const state = stateFromQueryParams({ materials: '3,abc,-1', basis: 'furlong', bins: '500', tol: '5', qunit: 'stone' });

        expect(state.filter.adsorbent_ids).toEqual([3]);
        expect(state.filter.pressure_basis).toBe('absolute');
        expect(state.bins).toBe(100);
        expect(state.tolerance).toBe(1);
        expect(state.uptakeUnit).toBe('mol/kg');
    });

    it('narrows chart queries to explicitly selected isotherms', () => {
        const state = { ...DEFAULT_EXPLORE_STATE, selectedIds: [5] };

        expect(chartFilter(state).isotherm_ids).toEqual([5]);
        expect(state.filter.isotherm_ids).toEqual([]);
    });
});
