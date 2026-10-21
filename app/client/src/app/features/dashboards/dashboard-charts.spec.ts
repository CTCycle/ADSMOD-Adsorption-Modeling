// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest';
import type { IsothermSeries, UptakeAtPressureRow } from '../../models/analytics.model';
import { buildIsothermOption, buildUptakeOption, TEMPERATURE_RAMP } from './dashboard-charts';

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
            'material',
            'mol/kg',
        ) as { series: { data: { itemStyle: { color: string; borderWidth?: number } }[] }[] };

        const points = option.series[0].data;
        expect(points).toHaveLength(2);
        expect(points[0].itemStyle.borderWidth).toBeUndefined();
        expect(points[1].itemStyle).toMatchObject({ color: '#ffffff', borderWidth: 2 });
    });
});
