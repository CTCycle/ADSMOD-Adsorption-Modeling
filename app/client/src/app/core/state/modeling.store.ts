// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { computed, Injectable, signal } from '@angular/core';
import type {
    FittingConfiguration,
    FittingDatasetReadiness,
    FittingSessionQuery,
    FittingSessionSummary,
    FittingWeighting,
    ModelBounds,
    ModelCatalogEntry,
    OptimizationMethod,
} from '../../models/fitting.model';
import {
    deleteFittingSession,
    fetchDatasetReadiness,
    fetchFittingConfiguration,
    fetchFittingSessions,
    fetchModelCatalog,
    restoreModelDefaults,
    saveModelSettings,
    startFittingSession,
} from '../../services/fitting.service';

export const DEFAULT_SESSION_QUERY: FittingSessionQuery = {
    query: '',
    statuses: [],
    datasetId: null,
    model: null,
    createdFrom: '',
    createdTo: '',
    sort: 'created_at',
    direction: 'desc',
    page: 1,
    pageSize: 20,
};

export type WizardStep = 0 | 1 | 2 | 3;

export interface FittingSessionDraft {
    step: WizardStep;
    name: string;
    datasetId: number | null;
    models: string[];
    /** Session-only bound overrides; never written back to the global catalog. */
    overrides: Record<string, ModelBounds>;
    optimizer: OptimizationMethod;
    weighting: FittingWeighting;
    maxEvaluations: number;
}

/** Bounds as effectively applied: canonical spec bounds overlaid by saved global defaults. */
export const catalogBounds = (model: ModelCatalogEntry): ModelBounds =>
    Object.fromEntries(model.parameters.map((parameter) => [parameter.name, { lower: parameter.lower, upper: parameter.upper }]));

export function boundsError(bounds: ModelBounds): string | null {
    for (const [name, value] of Object.entries(bounds)) {
        if (!Number.isFinite(value.lower) || !Number.isFinite(value.upper)) return `${name}: enter numeric bounds.`;
        if (value.lower <= 0) return `${name}: the lower bound must be greater than zero.`;
        if (value.lower >= value.upper) return `${name}: the lower bound must be smaller than the upper bound.`;
    }
    return null;
}

/** Shared state for the Modeling workspace: global model catalog, session list, and the new-session draft. */
@Injectable({ providedIn: 'root' })
export class ModelingStore {
    readonly configuration = signal<FittingConfiguration | null>(null);
    readonly configurationError = signal<string | null>(null);

    readonly catalog = signal<ModelCatalogEntry[]>([]);
    readonly catalogLoaded = signal(false);
    readonly catalogError = signal<string | null>(null);
    readonly enabledModels = computed(() => this.catalog().filter((model) => model.enabled));

    readonly sessionQuery = signal<FittingSessionQuery>(DEFAULT_SESSION_QUERY);
    readonly sessions = signal<FittingSessionSummary[]>([]);
    readonly sessionTotal = signal(0);
    readonly sessionsLoading = signal(false);
    readonly sessionsError = signal<string | null>(null);
    readonly filtersActive = computed(() => {
        const query = this.sessionQuery();
        return Boolean(query.query.trim() || query.statuses.length || query.datasetId !== null || query.model || query.createdFrom || query.createdTo);
    });

    readonly draft = signal<FittingSessionDraft | null>(null);
    readonly readiness = signal<FittingDatasetReadiness | null>(null);
    readonly readinessLoading = signal(false);
    readonly readinessError = signal<string | null>(null);

    private sessionsRevision = 0;
    private readinessRevision = 0;

    async loadConfiguration(): Promise<FittingConfiguration | null> {
        if (this.configuration()) return this.configuration();
        const result = await fetchFittingConfiguration();
        this.configurationError.set(result.error);
        this.configuration.set(result.data);
        return result.data;
    }

    async loadCatalog(): Promise<void> {
        const result = await fetchModelCatalog();
        this.catalogError.set(result.error);
        if (result.data) this.catalog.set(result.data.models);
        this.catalogLoaded.set(true);
    }

    async setModelEnabled(key: string, enabled: boolean): Promise<string | null> {
        const model = this.catalog().find((item) => item.key === key);
        if (!model) return 'Unknown model.';
        return this.applyEntry(await saveModelSettings(key, { enabled, parameters: catalogBounds(model) }));
    }

    async saveModelDefaults(key: string, bounds: ModelBounds): Promise<string | null> {
        const model = this.catalog().find((item) => item.key === key);
        if (!model) return 'Unknown model.';
        const invalid = boundsError(bounds);
        if (invalid) return invalid;
        return this.applyEntry(await saveModelSettings(key, { enabled: model.enabled, parameters: bounds }));
    }

    async restoreModelDefaults(key: string): Promise<string | null> {
        return this.applyEntry(await restoreModelDefaults(key));
    }

    private applyEntry(result: { data: ModelCatalogEntry | null; error: string | null }): string | null {
        const entry = result.data;
        if (!entry) return result.error ?? 'The model could not be saved.';
        this.catalog.update((models) => models.map((model) => (model.key === entry.key ? entry : model)));
        return null;
    }

    updateSessionQuery(patch: Partial<FittingSessionQuery>): void {
        this.sessionQuery.update((query) => ({ ...query, page: 1, ...patch }));
        void this.loadSessions();
    }

    clearSessionFilters(): void {
        const { sort, direction, pageSize } = this.sessionQuery();
        this.sessionQuery.set({ ...DEFAULT_SESSION_QUERY, sort, direction, pageSize });
        void this.loadSessions();
    }

    async loadSessions(): Promise<void> {
        const revision = ++this.sessionsRevision;
        this.sessionsLoading.set(true);
        const result = await fetchFittingSessions(this.sessionQuery());
        if (revision !== this.sessionsRevision) return;
        this.sessionsLoading.set(false);
        this.sessionsError.set(result.error);
        if (!result.data) return;
        const lastPage = Math.max(1, Math.ceil(result.data.total / result.data.page_size));
        if (result.data.items.length === 0 && result.data.page > lastPage) {
            this.sessionQuery.update((query) => ({ ...query, page: lastPage }));
            await this.loadSessions();
            return;
        }
        this.sessions.set(result.data.items);
        this.sessionTotal.set(result.data.total);
    }

    async deleteSession(sessionId: number): Promise<string | null> {
        const result = await deleteFittingSession(sessionId);
        if (result.error) return result.error;
        await this.loadSessions();
        return null;
    }

    /** Start a fresh draft from the current fitting defaults and enabled models. */
    async beginDraft(): Promise<void> {
        const [configuration] = await Promise.all([this.loadConfiguration(), this.loadCatalog()]);
        this.readiness.set(null);
        this.readinessError.set(null);
        this.draft.set({
            step: 0,
            name: '',
            datasetId: null,
            models: this.enabledModels().map((model) => model.key),
            overrides: {},
            optimizer: configuration?.default_optimizer ?? 'trf',
            weighting: configuration?.default_weighting ?? 'unweighted',
            maxEvaluations: configuration?.default_max_evaluations ?? 1000,
        });
    }

    updateDraft(patch: Partial<FittingSessionDraft>): void {
        this.draft.update((draft) => (draft ? { ...draft, ...patch } : draft));
    }

    discardDraft(): void {
        this.draft.set(null);
        this.readiness.set(null);
    }

    async selectDraftDataset(datasetId: number | null): Promise<void> {
        const revision = ++this.readinessRevision;
        this.updateDraft({ datasetId });
        this.readiness.set(null);
        this.readinessError.set(null);
        if (datasetId === null) return;
        this.readinessLoading.set(true);
        const result = await fetchDatasetReadiness(datasetId);
        if (revision !== this.readinessRevision) return;
        this.readinessLoading.set(false);
        this.readiness.set(result.data);
        this.readinessError.set(result.error);
        const readiness = result.data;
        if (!readiness) return;
        this.draft.update((draft) => draft && ({
            ...draft,
            models: draft.models.filter((key) => !readiness.model_incompatibility[key]),
            weighting: readiness.inverse_sigma_available ? draft.weighting : 'unweighted',
        }));
    }

    async startDraft(): Promise<{ sessionId: number | null; error: string | null }> {
        const draft = this.draft();
        const configuration = this.configuration();
        if (!draft || draft.datasetId === null) return { sessionId: null, error: 'Select a dataset.' };
        const result = await startFittingSession({
            name: draft.name.trim(),
            dataset_id: draft.datasetId,
            models: draft.models,
            optimizer: draft.optimizer,
            weighting: draft.weighting,
            max_evaluations: draft.maxEvaluations,
            parameter_overrides: Object.fromEntries(Object.entries(draft.overrides).filter(([key]) => draft.models.includes(key))),
            display_units: {
                pressure: configuration?.display_units.default_pressure ?? 'bar',
                uptake: configuration?.display_units.default_uptake ?? 'mmol/g',
            },
        });
        if (!result.data) return { sessionId: null, error: result.error ?? 'The fitting session could not be started.' };
        this.discardDraft();
        return { sessionId: result.data.session_id, error: null };
    }
}
