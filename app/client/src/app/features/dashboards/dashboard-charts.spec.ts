// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest';
import { type IsothermSeries, type UptakeAtPressureRow } from '../../models/analytics.model';
import { buildIsothermOption, buildUptakeOption, TEMPERATURE_RAMP } from './dashboard-charts';
import { type ExploreState, chartFilter, DEFAULT_EXPLORE_STATE, stateFromQueryParams, stateToQueryParams } from './explore/explore-state';

// Dashboard chart behavior

const meta = {
    component_id: 1,
    isotherm: 'iso',
    dataset_id: 1,
    dataset: 'Lab',
    dataset_source: 'uploaded' as const,
    material_id: 1,
    material: 'Material A',
    adsorbate_id: 1,
    adsorbate: 'carbon dioxide',
    mole_fraction: null,
    uptake_unit: 'mol/kg',
};

function series(isothermId: number, temperature: number, points: [number, number][]): IsothermSeries {
    return {
        ...meta,
        isotherm_id: isothermId,
        component_id: isothermId,
        temperature_k: temperature,
        points: points.map(([pressure, uptake], index) => ({ observation_id: index + 1, pressure, uptake, uptake_stddev: null })),
    };
}

type SeriesOption = { color: string; data: unknown[] };

describe('dashboard chart options', () => {
    it('colours isotherms by temperature and drops non-positive pressures on a log axis', () => {
        const option = buildIsothermOption(
            [series(1, 273, [[0, 0], [1, 2]]), series(2, 323, [[1, 3]])],
            { colorBy: 'temperature', logPressure: true, pressureUnit: 'bar' },
        ) as { xAxis: { type: string; name: string }; series: SeriesOption[] };

        expect(option.xAxis.type).toBe('log');
        expect(option.xAxis.name).toBe('Pressure (bar)');
        expect(option.series.map((item) => item.color)).toEqual([TEMPERATURE_RAMP[0], TEMPERATURE_RAMP[TEMPERATURE_RAMP.length - 1]]);
        expect(option.series[0].data).toEqual([[1, 2]]);
    });

    it('marks interpolated uptake values with hollow markers and omits rows without a value', () => {
        const row = (id: number, method: UptakeAtPressureRow['method'], status: UptakeAtPressureRow['status']): UptakeAtPressureRow => ({
            ...meta,
            isotherm_id: id,
            component_id: id,
            temperature_k: 270 + id,
            value: status === 'ok' ? id : null,
            method,
            status,
            sources: [],
        });
        const option = buildUptakeOption(
            [row(1, 'measured', 'ok'), row(2, 'interpolated', 'ok'), row(3, null, 'out_of_range')],
            'mol/kg',
        ) as { series: { data: { itemStyle: { color: string; borderWidth?: number } }[] }[] };

        const points = option.series[0].data;
        expect(points).toHaveLength(2);
        expect(points[0].itemStyle.borderWidth).toBeUndefined();
        expect(points[1].itemStyle).toMatchObject({ color: 'transparent', borderWidth: 2 });
    });
});

// Explore state URL behavior

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
