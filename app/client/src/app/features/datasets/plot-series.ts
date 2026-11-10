// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { ExperimentComponent, ObservationBranch, ObservationRow } from '../../models/dataset.model';

export interface PlotSeries {
    name: string;
    branch: ObservationBranch;
    points: [number, number][];
}

/**
 * One series per species and branch, so multi-species experiments are not
 * drawn as one line and a desorption branch is never joined to its
 * adsorption branch. Series follow species position, adsorption first.
 */
export function buildPlotSeries(rows: readonly ObservationRow[], components: readonly ExperimentComponent[]): PlotSeries[] {
    const grouped = new Map<string, { adsorbateId: number; branch: ObservationBranch; points: [number, number][] }>();
    for (const row of rows) {
        const key = `${row.adsorbate_id}:${row.branch}`;
        const group = grouped.get(key) ?? { adsorbateId: row.adsorbate_id, branch: row.branch, points: [] };
        group.points.push([row.pressure_canonical, row.uptake_mol_kg]);
        grouped.set(key, group);
    }
    const position = (id: number) => components.find((component) => component.adsorbate_id === id)?.position ?? Number.MAX_SAFE_INTEGER;
    return [...grouped.values()]
        .sort((left, right) => position(left.adsorbateId) - position(right.adsorbateId) || left.branch.localeCompare(right.branch))
        .map((group) => {
            const species = components.find((component) => component.adsorbate_id === group.adsorbateId)?.name ?? 'Measurements';
            return {
                name: group.branch === 'desorption' ? `${species} · desorption` : species,
                branch: group.branch,
                points: group.points,
            };
        });
}
