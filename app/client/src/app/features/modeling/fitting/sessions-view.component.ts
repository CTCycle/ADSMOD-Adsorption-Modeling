// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import { Component, DestroyRef, OnInit, computed, effect, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CoreWorkspaceStore } from '../../../core/state/core-workspace.store';
import { ModelingStore } from '../../../core/state/modeling.store';
import type { FittingSessionSort, FittingSessionStatus, FittingSessionSummary } from '../../../models/fitting.model';
import { ConfirmDialogComponent } from '../../../shared/components/confirm-dialog/confirm-dialog.component';
import { DataPaginationComponent } from '../../../shared/components/data-pagination/data-pagination.component';
import { IconComponent } from '../../../shared/components/icon/icon.component';
import { OPTIMIZER_LABELS, SESSION_STATUS_LABELS, WEIGHTING_LABELS, formatDuration } from './session-format';

const SORT_OPTIONS: { value: string; label: string; sort: FittingSessionSort; direction: 'asc' | 'desc' }[] = [
    { value: 'created_at:desc', label: 'Newest first', sort: 'created_at', direction: 'desc' },
    { value: 'created_at:asc', label: 'Oldest first', sort: 'created_at', direction: 'asc' },
    { value: 'name:asc', label: 'Name A–Z', sort: 'name', direction: 'asc' },
    { value: 'name:desc', label: 'Name Z–A', sort: 'name', direction: 'desc' },
    { value: 'status:asc', label: 'Status', sort: 'status', direction: 'asc' },
];
const LIVE_REFRESH_MS = 3000;

/** Curve Fitting landing view: persisted fitting sessions with filtering and pagination. */
@Component({
    selector: 'adsmod-sessions-view',
    standalone: true,
    imports: [ConfirmDialogComponent, DataPaginationComponent, DatePipe, IconComponent, RouterLink],
    template: `
        <section class="sessions" aria-labelledby="sessions-title">
            <header class="sessions-head">
                <h2 id="sessions-title">Fitting sessions <span class="sessions-count">{{ store.sessionTotal() }}</span></h2>
            </header>

            <div class="sessions-tools" role="search">
                <input type="search" aria-label="Search sessions by name" placeholder="Search sessions" [value]="query().query" (input)="onSearch($any($event.target).value)" />
                <select class="select-input" aria-label="Filter by status" [value]="statusValue()" (change)="setStatus($any($event.target).value)">
                    <option value="">All statuses</option>
                    @for (status of statuses; track status) {
                        <option [value]="status">{{ statusLabels[status] }}</option>
                    }
                </select>
                <select class="select-input" aria-label="Sort sessions" [value]="sortValue()" (change)="setSort($any($event.target).value)">
                    @for (option of sortOptions; track option.value) {
                        <option [value]="option.value">{{ option.label }}</option>
                    }
                </select>
                <button class="button secondary filters-toggle" type="button" aria-controls="session-filters" [attr.aria-expanded]="filtersOpen()" (click)="filtersOpen.set(!filtersOpen())">
                    <adsmod-icon name="filter" /> Filters
                    @if (advancedCount()) { <span class="filter-count">{{ advancedCount() }}</span> }
                </button>
                <a class="button primary" routerLink="/modeling/fitting/new"><adsmod-icon name="plus" /> New session</a>
            </div>

            @if (filtersOpen()) {
                <div class="sessions-filters" id="session-filters">
                    <label>
                        <span>Dataset</span>
                        <select class="select-input" [value]="query().datasetId ?? ''" (change)="setDataset($any($event.target).value)">
                            <option value="">Any dataset</option>
                            @for (dataset of workspace.datasets(); track dataset.id) {
                                <option [value]="dataset.id" [selected]="dataset.id === query().datasetId">{{ dataset.name }}</option>
                            }
                        </select>
                    </label>
                    <label>
                        <span>Model</span>
                        <select class="select-input" [value]="query().model ?? ''" (change)="store.updateSessionQuery({ model: $any($event.target).value || null })">
                            <option value="">Any model</option>
                            @for (model of store.catalog(); track model.key) {
                                <option [value]="model.key" [selected]="model.key === query().model">{{ model.name }}</option>
                            }
                        </select>
                    </label>
                    <label>
                        <span>Created from</span>
                        <input type="date" [value]="query().createdFrom" [max]="query().createdTo || null" (change)="store.updateSessionQuery({ createdFrom: $any($event.target).value })" />
                    </label>
                    <label>
                        <span>Created to</span>
                        <input type="date" [value]="query().createdTo" [min]="query().createdFrom || null" (change)="store.updateSessionQuery({ createdTo: $any($event.target).value })" />
                    </label>
                    <button class="button quiet" type="button" [disabled]="!store.filtersActive()" (click)="clearFilters()">Clear filters</button>
                </div>
            }

            @if (store.sessionsError()) {
                <div class="inline-error" role="alert">
                    <span>Fitting sessions could not be loaded: {{ store.sessionsError() }}</span>
                    <button class="button secondary" type="button" (click)="store.loadSessions()">Retry</button>
                </div>
            } @else if (store.sessionsLoading() && !store.sessions().length) {
                <p class="sessions-empty" role="status">Loading fitting sessions…</p>
            } @else if (!store.sessions().length) {
                @if (store.filtersActive()) {
                    <div class="sessions-empty" role="status">
                        <strong>No sessions match the current filters.</strong>
                        <button class="button quiet" type="button" (click)="clearFilters()">Clear filters</button>
                    </div>
                } @else {
                    <div class="sessions-empty">
                        <strong>No fitting sessions yet</strong>
                        <span>Create a session to fit adsorption models to every experiment of a dataset.</span>
                    </div>
                }
            } @else {
                <div class="sessions-table-wrap">
                    <table class="sessions-table" [attr.aria-busy]="store.sessionsLoading()">
                        <caption class="sr-only">Fitting sessions</caption>
                        <thead>
                            <tr>
                                <th scope="col" class="col-toggle"><span class="sr-only">Details</span></th>
                                <th scope="col">Name</th>
                                <th scope="col">Created</th>
                                <th scope="col" class="num">Models</th>
                                <th scope="col" class="num">Experiments</th>
                                <th scope="col">Status</th>
                                <th scope="col" class="col-actions"><span class="sr-only">Actions</span></th>
                            </tr>
                        </thead>
                        <tbody>
                            @for (session of store.sessions(); track session.id) {
                                <tr class="session-row" [class.open]="expanded() === session.id">
                                    <td class="col-toggle">
                                        <button class="button quiet icon-button" type="button" [attr.aria-expanded]="expanded() === session.id" [attr.aria-controls]="'session-detail-' + session.id" [attr.aria-label]="'Details for ' + session.name" (click)="toggle(session.id)">
                                            <span class="row-chevron" [class.open]="expanded() === session.id"><adsmod-icon name="chevron" /></span>
                                        </button>
                                    </td>
                                    <td data-label="Name"><a class="session-name" [routerLink]="['/modeling/fitting', session.id]" [title]="session.name">{{ session.name }}</a><span class="session-dataset">{{ session.dataset_name }}</span></td>
                                    <td data-label="Created" class="nowrap">{{ session.created_at | date: 'medium' }}</td>
                                    <td data-label="Models" class="num">{{ session.models.length }}</td>
                                    <td data-label="Experiments" class="num">{{ session.experiment_count }}</td>
                                    <td data-label="Status"><span class="session-status" [attr.data-status]="session.status">{{ statusLabels[session.status] }}</span>@if (session.stale) { <span class="session-stale" title="The dataset changed after this session was fitted">Stale</span> }</td>
                                    <td class="col-actions">
                                        <a class="button quiet icon-button" [routerLink]="['/modeling/fitting', session.id]" title="Open session" [attr.aria-label]="'Open ' + session.name"><adsmod-icon name="eye" /></a>
                                        <button class="button quiet danger icon-button" type="button" title="Delete session" [disabled]="session.status === 'running'" [attr.aria-label]="'Delete ' + session.name" (click)="deleteTarget.set(session)"><adsmod-icon name="trash" /></button>
                                    </td>
                                </tr>
                                @if (expanded() === session.id) {
                                    <tr class="session-detail-row">
                                        <td colspan="7" [id]="'session-detail-' + session.id">
                                            <dl class="session-detail">
                                                <div><dt>Dataset</dt><dd>{{ session.dataset_name }} <span class="muted">#{{ session.dataset_id }}</span></dd></div>
                                                <div><dt>Models</dt><dd>{{ modelNames(session) }}</dd></div>
                                                <div><dt>Optimizer</dt><dd>{{ optimizerLabels[session.optimizer] }}</dd></div>
                                                <div><dt>Weighting</dt><dd>{{ weightingLabels[session.weighting] }}</dd></div>
                                                <div><dt>Max iterations</dt><dd>{{ session.max_evaluations }}</dd></div>
                                                <div><dt>Duration</dt><dd>{{ formatDuration(session.duration_seconds) }}</dd></div>
                                                <div class="wide"><dt>Outcome</dt><dd>{{ session.message || statusLabels[session.status] }}</dd></div>
                                                @if (bestModels(session).length) {
                                                    <div class="wide"><dt>Best model per experiment</dt>
                                                        <dd><ul class="best-list">
                                                            @for (entry of bestModels(session); track entry[0]) {
                                                                <li><span>{{ entry[0] }}</span> {{ entry[1] ? modelName(entry[1]) : 'No valid fit' }}</li>
                                                            }
                                                        </ul></dd>
                                                    </div>
                                                }
                                            </dl>
                                            <a class="button secondary" [routerLink]="['/modeling/fitting', session.id]">{{ session.status === 'running' ? 'View progress' : 'Open results' }}</a>
                                        </td>
                                    </tr>
                                }
                            }
                        </tbody>
                    </table>
                </div>
                <adsmod-data-pagination
                    [page]="query().page"
                    [pageSize]="query().pageSize"
                    [pageSizes]="pageSizes"
                    [total]="store.sessionTotal()"
                    itemLabel="session"
                    (pageChange)="store.updateSessionQuery({ page: $event })"
                    (pageSizeChange)="store.updateSessionQuery({ pageSize: $event })"
                />
            }
        </section>

        <adsmod-confirm-dialog
            [open]="deleteTarget() !== null"
            title="Delete fitting session"
            [message]="'Delete “' + (deleteTarget()?.name ?? '') + '” and its fitting results? The dataset, its experiments, and the model catalog are kept.'"
            confirmLabel="Delete session"
            (confirmed)="confirmDelete()"
            (closed)="deleteTarget.set(null)"
        />
        @if (deleteError()) {
            <p class="inline-error" role="alert">{{ deleteError() }}</p>
        }
    `,
    styleUrl: './sessions-view.component.css',
})
export class SessionsViewComponent implements OnInit {
    protected readonly store = inject(ModelingStore);
    protected readonly workspace = inject(CoreWorkspaceStore);
    protected readonly statusLabels = SESSION_STATUS_LABELS;
    protected readonly optimizerLabels = OPTIMIZER_LABELS;
    protected readonly weightingLabels = WEIGHTING_LABELS;
    protected readonly formatDuration = formatDuration;
    protected readonly statuses = Object.keys(SESSION_STATUS_LABELS) as FittingSessionStatus[];
    protected readonly sortOptions = SORT_OPTIONS;
    protected readonly pageSizes = [10, 20, 50];
    protected readonly query = this.store.sessionQuery;
    protected readonly expanded = signal<number | null>(null);
    protected readonly filtersOpen = signal(false);
    protected readonly deleteTarget = signal<FittingSessionSummary | null>(null);
    protected readonly deleteError = signal<string | null>(null);
    protected readonly statusValue = computed(() => this.query().statuses.at(0) ?? '');
    protected readonly sortValue = computed(() => `${this.query().sort}:${this.query().direction}`);
    protected readonly advancedCount = computed(() => {
        const query = this.query();
        return [query.datasetId !== null, Boolean(query.model), Boolean(query.createdFrom), Boolean(query.createdTo)].filter(Boolean).length;
    });
    private searchTimer: ReturnType<typeof setTimeout> | null = null;
    private refreshTimer: ReturnType<typeof setTimeout> | null = null;

    constructor() {
        const destroyRef = inject(DestroyRef);
        // Keep running sessions current without loading more than the visible page.
        effect(() => {
            const running = this.store.sessions().some((session) => session.status === 'running');
            if (this.refreshTimer) clearTimeout(this.refreshTimer);
            this.refreshTimer = running ? setTimeout(() => void this.store.loadSessions(), LIVE_REFRESH_MS) : null;
        });
        destroyRef.onDestroy(() => {
            if (this.refreshTimer) clearTimeout(this.refreshTimer);
            if (this.searchTimer) clearTimeout(this.searchTimer);
        });
    }

    ngOnInit(): void {
        if (this.advancedCount()) this.filtersOpen.set(true);
        void this.store.loadSessions();
        if (!this.store.catalogLoaded()) void this.store.loadCatalog();
    }

    protected onSearch(value: string): void {
        this.query.update((query) => ({ ...query, query: value }));
        if (this.searchTimer) clearTimeout(this.searchTimer);
        this.searchTimer = setTimeout(() => this.store.updateSessionQuery({ query: value }), 250);
    }

    protected setStatus(value: string): void {
        this.store.updateSessionQuery({ statuses: value ? [value as FittingSessionStatus] : [] });
    }

    protected setSort(value: string): void {
        const option = SORT_OPTIONS.find((item) => item.value === value);
        if (option) this.store.updateSessionQuery({ sort: option.sort, direction: option.direction });
    }

    protected setDataset(value: string): void {
        const id = Number(value);
        this.store.updateSessionQuery({ datasetId: Number.isInteger(id) && id > 0 ? id : null });
    }

    protected clearFilters(): void {
        this.store.clearSessionFilters();
    }

    protected toggle(id: number): void {
        this.expanded.set(this.expanded() === id ? null : id);
    }

    protected modelName(key: string): string {
        return this.store.catalog().find((model) => model.key === key)?.name ?? key;
    }

    protected modelNames(session: FittingSessionSummary): string {
        return session.models.map((key) => this.modelName(key)).join(', ');
    }

    protected bestModels(session: FittingSessionSummary): [string, string | null][] {
        return Object.entries(session.best_models);
    }

    protected async confirmDelete(): Promise<void> {
        const target = this.deleteTarget();
        this.deleteTarget.set(null);
        if (!target) return;
        const error = await this.store.deleteSession(target.id);
        this.deleteError.set(error ? `“${target.name}” could not be deleted: ${error}` : null);
        if (!error && this.expanded() === target.id) this.expanded.set(null);
    }
}
