// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { ExperimentComponent, ObservationRow } from '../../models/dataset.model';

export interface PlotSeries {
    name: string;
    points: [number, number][];
}

/** One series per species so multi-species experiments are not drawn as a single line. */
export function buildPlotSeries(rows: readonly ObservationRow[], components: readonly ExperimentComponent[]): PlotSeries[] {
    const byComponent = new Map<number, [number, number][]>();
    for (const row of rows) {
        const points = byComponent.get(row.component_id) ?? [];
        points.push([row.pressure_canonical, row.uptake_mol_kg]);
        byComponent.set(row.component_id, points);
    }
    return [...byComponent].map(([id, points]) => ({
        name: components.find((component) => component.id === id)?.name ?? 'Measurements',
        points,
    }));
}
