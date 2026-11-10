// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest';
import type { ExperimentComponent, ObservationRow } from '../../models/dataset.model';
import { buildPlotSeries } from './plot-series';

const components: ExperimentComponent[] = [
    { id: 1, position: 1, adsorbate_id: 10, name: 'CO2', mole_fraction: null },
    { id: 2, position: 2, adsorbate_id: 11, name: 'N2', mole_fraction: null },
];

function row(component_id: number, pressure: number, uptake: number): ObservationRow {
    return {
        id: pressure,
        component_id,
        sequence_index: 0,
        source_row: null,
        pressure_original: pressure,
        pressure_original_unit: 'Pa',
        pressure_canonical: pressure,
        pressure_canonical_unit: 'Pa',
        uptake_original: uptake,
        uptake_original_unit: 'mol/kg',
        uptake_mol_kg: uptake,
        uptake_stddev_mol_kg: null,
    };
}

describe('buildPlotSeries', () => {
    it('keeps a single-species experiment as one series', () => {
        expect(buildPlotSeries([row(1, 1, 0.5), row(1, 2, 0.9)], components.slice(0, 1))).toEqual([
            { name: 'CO2', points: [[1, 0.5], [2, 0.9]] },
        ]);
    });

    it('splits a multi-species experiment into one series per species', () => {
        const series = buildPlotSeries([row(1, 1, 0.5), row(2, 1, 0.1), row(1, 2, 0.9), row(2, 2, 0.2)], components);
        expect(series.map((item) => item.name)).toEqual(['CO2', 'N2']);
        expect(series[1].points).toEqual([[1, 0.1], [2, 0.2]]);
    });
});
