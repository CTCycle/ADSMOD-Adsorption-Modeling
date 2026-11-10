// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { computed, signal } from '@angular/core';
import type {
    ComponentUpdate,
    DatasetEditIssue,
    DatasetSummary,
    DatasetUpdateRequest,
    ExperimentSummary,
    ExperimentUpdate,
    ObservationBranch,
    ObservationRow,
    ObservationUpdate,
} from '../../models/dataset.model';

export type DatasetField = 'name' | 'description' | 'tags';
export type ExperimentField = Exclude<keyof ExperimentUpdate, 'id' | 'components'>;
/** Numeric measurement fields; the branch is a label edited on its own. */
export type ObservationField = Exclude<keyof ObservationUpdate, 'id' | 'branch'>;

type Value = string | number | null | string[];
type ObservationDraftFields = Partial<Record<ObservationField, number | null>> & { branch?: ObservationBranch };

/** Where an edited measurement sits in the worksheet view it was edited in. */
export interface WorksheetLocation {
    order: 'source' | 'pressure';
    adsorbateId: number | null;
    branch: ObservationBranch | null;
    index: number;
}

interface ExperimentDraft {
    label: string;
    fields: Partial<Record<ExperimentField, Value>>;
    /** Species position -> replacement adsorbate id. */
    species: Map<number, ComponentUpdate['adsorbate_id']>;
}

interface ObservationDraft {
    experimentId: number;
    location: WorksheetLocation;
    fields: ObservationDraftFields;
}

const NONNEGATIVE_FIELDS: readonly ObservationField[] = ['pressure_original', 'uptake_original'];

function sameValue(left: Value | undefined, right: Value | undefined): boolean {
    return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

function observationProblem(field: ObservationField, value: number | null): string | null {
    if (NONNEGATIVE_FIELDS.includes(field)) {
        return value === null || !Number.isFinite(value) || value < 0 ? 'Enter a finite value of zero or more.' : null;
    }
    if (value === null) return null;
    if (field === 'gas_mole_fraction') {
        return Number.isFinite(value) && value >= 0 && value <= 1 ? null : 'Gas mole fraction must be between 0 and 1 or empty.';
    }
    return Number.isFinite(value) && value > 0 ? null : 'Uncertainty must be a positive number or empty.';
}

/**
 * Local, unsaved edits for one dataset.
 *
 * Changes are keyed by record id so they survive paging and experiment
 * switches. A change equal to the stored value is dropped, so `dirty` means
 * "something would actually be written".
 */
export class DatasetEditSession {
    readonly active = signal(false);
    readonly saving = signal(false);
    readonly issues = signal<DatasetEditIssue[]>([]);

    private readonly dataset = signal<Partial<Record<DatasetField, Value>>>({});
    private readonly experiments = signal<Map<number, ExperimentDraft>>(new Map());
    private readonly observations = signal<Map<number, ObservationDraft>>(new Map());

    readonly changeCount = computed(() => {
        let count = Object.keys(this.dataset()).length;
        for (const draft of this.experiments().values()) {
            count += Object.keys(draft.fields).length + draft.species.size;
        }
        for (const draft of this.observations().values()) count += Object.keys(draft.fields).length;
        return count;
    });
    readonly dirty = computed(() => this.changeCount() > 0);
    readonly editedExperimentIds = computed(() => {
        const ids = new Set(this.experiments().keys());
        for (const draft of this.observations().values()) ids.add(draft.experimentId);
        return ids;
    });

    start(): void {
        this.discard();
        this.active.set(true);
    }

    /** Drop every pending change and leave edit mode. */
    cancel(): void {
        this.discard();
        this.active.set(false);
    }

    private discard(): void {
        this.dataset.set({});
        this.experiments.set(new Map());
        this.observations.set(new Map());
        this.issues.set([]);
    }

    // Dataset ---------------------------------------------------------------
    datasetValue<T extends Value>(field: DatasetField, stored: DatasetSummary): T {
        const pending = this.dataset()[field];
        return (pending !== undefined ? pending : stored[field]) as T;
    }

    setDatasetValue(field: DatasetField, value: Value, stored: DatasetSummary): void {
        this.dataset.update((current) => {
            const next = { ...current };
            if (sameValue(value, stored[field])) delete next[field];
            else next[field] = value;
            return next;
        });
        this.clearIssue('dataset', null, field);
    }

    // Experiments -----------------------------------------------------------
    experimentValue<T extends Value>(experiment: ExperimentSummary, field: ExperimentField, stored: Value): T {
        const pending = this.experiments().get(experiment.id)?.fields[field];
        return (pending !== undefined ? pending : stored) as T;
    }

    isExperimentChanged(experimentId: number, field: ExperimentField): boolean {
        return this.experiments().get(experimentId)?.fields[field] !== undefined;
    }

    setExperimentValue(experiment: ExperimentSummary, field: ExperimentField, value: Value, stored: Value): void {
        this.updateExperiment(experiment, (draft) => {
            if (sameValue(value, stored)) delete draft.fields[field];
            else draft.fields[field] = value;
        });
        this.clearIssue('experiment', experiment.id, field);
    }

    speciesValue(experiment: ExperimentSummary, position: number, stored: number): number {
        return this.experiments().get(experiment.id)?.species.get(position) ?? stored;
    }

    setSpeciesValue(experiment: ExperimentSummary, position: number, adsorbateId: number, stored: number): void {
        this.updateExperiment(experiment, (draft) => {
            if (adsorbateId === stored) draft.species.delete(position);
            else draft.species.set(position, adsorbateId);
        });
        this.clearIssue('experiment', experiment.id, 'components');
    }

    private updateExperiment(experiment: ExperimentSummary, change: (draft: ExperimentDraft) => void): void {
        this.experiments.update((current) => {
            const next = new Map(current);
            const previous = next.get(experiment.id);
            const draft: ExperimentDraft = {
                label: experiment.name,
                fields: { ...(previous?.fields ?? {}) },
                species: new Map(previous?.species ?? []),
            };
            change(draft);
            if (Object.keys(draft.fields).length || draft.species.size) next.set(experiment.id, draft);
            else next.delete(experiment.id);
            return next;
        });
    }

    // Observations ----------------------------------------------------------
    observationValue(row: ObservationRow, field: ObservationField): number | null {
        const pending = this.observations().get(row.id)?.fields[field];
        return pending !== undefined ? pending : row[field];
    }

    isObservationChanged(rowId: number, field: ObservationField | 'branch'): boolean {
        return this.observations().get(rowId)?.fields[field] !== undefined;
    }

    setObservationValue(
        experimentId: number,
        row: ObservationRow,
        field: ObservationField,
        value: number | null,
        location: WorksheetLocation,
    ): void {
        this.updateObservation(experimentId, row, location, (fields) => {
            if (sameValue(value, row[field])) delete fields[field];
            else fields[field] = value;
        });
        this.clearIssue('observation', row.id, field);
    }

    observationBranch(row: ObservationRow): ObservationBranch {
        return this.observations().get(row.id)?.fields.branch ?? row.branch;
    }

    setObservationBranch(experimentId: number, row: ObservationRow, branch: ObservationBranch, location: WorksheetLocation): void {
        this.updateObservation(experimentId, row, location, (fields) => {
            if (branch === row.branch) delete fields.branch;
            else fields.branch = branch;
        });
        this.clearIssue('observation', row.id, 'branch');
    }

    private updateObservation(
        experimentId: number,
        row: ObservationRow,
        location: WorksheetLocation,
        change: (fields: ObservationDraftFields) => void,
    ): void {
        this.observations.update((current) => {
            const next = new Map(current);
            const fields = { ...(next.get(row.id)?.fields ?? {}) };
            change(fields);
            if (Object.keys(fields).length) {
                next.set(row.id, { experimentId, location, fields });
            } else {
                next.delete(row.id);
            }
            return next;
        });
    }

    /** Worksheet position of an edited measurement, used to bring it back into view. */
    observationLocation(observationId: number): { experimentId: number; location: WorksheetLocation } | null {
        const draft = this.observations().get(observationId);
        return draft ? { experimentId: draft.experimentId, location: draft.location } : null;
    }

    experimentLabel(experimentId: number): string | null {
        return this.experiments().get(experimentId)?.label ?? null;
    }

    // Validation ------------------------------------------------------------
    issueFor(scope: DatasetEditIssue['scope'], id: number | null, field: string): DatasetEditIssue | undefined {
        return this.issues().find(
            (issue) =>
                issue.scope === scope &&
                issue.field === field &&
                (scope === 'observation' ? issue.observation_id === id : scope === 'experiment' ? issue.experiment_id === id : true),
        );
    }

    private clearIssue(scope: DatasetEditIssue['scope'], id: number | null, field: string): void {
        const resolved = this.issueFor(scope, id, field);
        if (!resolved) return;
        this.issues.update((issues) => issues.filter((issue) => issue !== resolved));
    }

    /** Checks that need no server round trip; the backend re-validates everything. */
    validateLocally(): DatasetEditIssue[] {
        const issues: DatasetEditIssue[] = [];
        const name = this.dataset()['name'];
        if (typeof name === 'string' && !name.trim()) {
            issues.push({ scope: 'dataset', field: 'name', message: 'The dataset name cannot be empty.', experiment_id: null, observation_id: null });
        }
        for (const [experimentId, draft] of this.experiments()) {
            const label = draft.fields['name'];
            if (typeof label === 'string' && !label.trim()) {
                issues.push({ scope: 'experiment', field: 'name', message: 'The experiment name cannot be empty.', experiment_id: experimentId, observation_id: null });
            }
            const temperature = draft.fields['temperature_original'];
            if (temperature !== undefined && (typeof temperature !== 'number' || !Number.isFinite(temperature))) {
                issues.push({ scope: 'experiment', field: 'temperature_original', message: 'Enter a numeric temperature.', experiment_id: experimentId, observation_id: null });
            }
        }
        for (const [observationId, draft] of this.observations()) {
            const numeric = Object.entries(draft.fields).filter(([field]) => field !== 'branch') as [ObservationField, number | null][];
            for (const [field, value] of numeric) {
                const message = observationProblem(field, value);
                if (message) {
                    issues.push({ scope: 'observation', field, message, experiment_id: draft.experimentId, observation_id: observationId });
                }
            }
        }
        return issues;
    }

    toRequest(): DatasetUpdateRequest {
        const request: DatasetUpdateRequest = {};
        const dataset = this.dataset();
        if (dataset.name !== undefined) request.name = String(dataset.name);
        if (dataset.description !== undefined) request.description = String(dataset.description);
        if (dataset.tags !== undefined) request.tags = dataset.tags as string[];
        const experiments: ExperimentUpdate[] = [...this.experiments()].map(([id, draft]) => {
            const update = { id, ...draft.fields } as ExperimentUpdate;
            if (draft.species.size) {
                update.components = [...draft.species].map(([position, adsorbate_id]): ComponentUpdate => ({ position, adsorbate_id }));
            }
            return update;
        });
        if (experiments.length) request.experiments = experiments;
        const observations: ObservationUpdate[] = [...this.observations()].map(([id, draft]) => ({ id, ...draft.fields }));
        if (observations.length) request.observations = observations;
        return request;
    }
}
