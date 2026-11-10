// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { unsavedChangesGuard, type HasUnsavedChanges } from '../../core/guards/unsaved-changes.guard';
import type { DatasetSummary, ExperimentSummary, ObservationRow } from '../../models/dataset.model';
import { DatasetEditSession } from './dataset-edit-session';

const dataset = { id: 1, name: 'Screening', description: '', tags: [] } as unknown as DatasetSummary;
const experiment = {
    id: 7,
    name: 'A',
    adsorbent_id: 3,
    components: [{ position: 1, adsorbate_id: 5, name: 'CO2' }],
} as unknown as ExperimentSummary;
const at = { order: 'source', adsorbateId: null, index: 0 } as const;

function row(id: number, pressure: number): ObservationRow {
    return {
        id,
        adsorbate_id: 5,
        source_row: id,
        pressure_original: pressure,
        pressure_original_unit: 'bar',
        pressure_canonical: pressure * 1e5,
        pressure_canonical_unit: 'Pa',
        uptake_original: 1.5,
        uptake_original_unit: 'mol/kg',
        uptake_mol_kg: 1.5,
        uptake_stddev_mol_kg: null,
        gas_mole_fraction: null,
    };
}

describe('DatasetEditSession', () => {
    it('keeps only real changes and builds one change set across experiments and pages', () => {
        const session = new DatasetEditSession();
        session.start();
        const first = row(101, 0.1);

        session.setObservationValue(experiment.id, first, 'pressure_original', 0.2, at);
        session.setObservationValue(experiment.id, row(250, 2), 'uptake_stddev_mol_kg', 0.05, at);
        session.setExperimentValue(experiment, 'temperature_original_unit', '°C', 'K');
        session.setSpeciesValue(experiment, 1, 9, 5);
        session.setDatasetValue('name', 'Curated', dataset);

        expect(session.observationValue(first, 'pressure_original')).toBe(0.2);
        expect(session.editedExperimentIds()).toEqual(new Set([7]));
        expect(session.toRequest()).toEqual({
            name: 'Curated',
            experiments: [{ id: 7, temperature_original_unit: '°C', components: [{ position: 1, adsorbate_id: 9 }] }],
            observations: [
                { id: 101, pressure_original: 0.2 },
                { id: 250, uptake_stddev_mol_kg: 0.05 },
            ],
        });

        // Typing the stored value back drops the change instead of sending a no-op.
        session.setObservationValue(experiment.id, first, 'pressure_original', 0.1, at);
        session.setDatasetValue('name', 'Screening', dataset);
        expect(session.toRequest().observations).toEqual([{ id: 250, uptake_stddev_mol_kg: 0.05 }]);
        expect(session.toRequest().name).toBeUndefined();
    });

    it('cancel discards every pending change and leaves edit mode', () => {
        const session = new DatasetEditSession();
        session.start();
        session.setObservationValue(experiment.id, row(101, 0.1), 'pressure_original', 3, at);
        expect(session.dirty()).toBe(true);

        session.cancel();

        expect(session.active()).toBe(false);
        expect(session.dirty()).toBe(false);
        expect(session.toRequest()).toEqual({});
    });

    it('flags invalid values locally and clears an issue once the cell is corrected', () => {
        const session = new DatasetEditSession();
        session.start();
        const first = row(101, 0.1);
        session.setObservationValue(experiment.id, first, 'pressure_original', -1, at);
        session.setObservationValue(experiment.id, row(102, 0.2), 'uptake_original', Number.NaN, at);
        session.setObservationValue(experiment.id, row(103, 0.3), 'uptake_stddev_mol_kg', 0, at);
        session.setObservationValue(experiment.id, row(104, 0.4), 'gas_mole_fraction', 1.2, at);
        session.setObservationValue(experiment.id, row(105, 0.5), 'gas_mole_fraction', 0, at);

        session.issues.set(session.validateLocally());

        expect(session.issues().map((issue) => [issue.observation_id, issue.field])).toEqual([
            [101, 'pressure_original'],
            [102, 'uptake_original'],
            [103, 'uptake_stddev_mol_kg'],
            [104, 'gas_mole_fraction'],
        ]);
        session.setObservationValue(experiment.id, first, 'pressure_original', 0.4, at);
        expect(session.issueFor('observation', 101, 'pressure_original')).toBeUndefined();
        expect(session.issues()).toHaveLength(3);
    });
});

describe('unsavedChangesGuard', () => {
    afterEach(() => vi.restoreAllMocks());

    function run(component: HasUnsavedChanges): unknown {
        return unsavedChangesGuard(component, {} as ActivatedRouteSnapshot, {} as RouterStateSnapshot, {} as RouterStateSnapshot);
    }

    it('asks before leaving with unsaved changes and stays when declined', () => {
        const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);

        expect(run({ hasUnsavedChanges: () => true })).toBe(false);
        expect(confirm).toHaveBeenCalledOnce();
        expect(run({ hasUnsavedChanges: () => false })).toBe(true);
        expect(confirm).toHaveBeenCalledOnce();
    });
});
