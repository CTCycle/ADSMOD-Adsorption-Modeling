// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import { Component, ElementRef, computed, effect, inject, input, output, viewChild } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { DatasetListState, DatasetSort } from '../../core/state/core-workspace.store';
import type { DatasetSummary } from '../../models/dataset.model';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import { IconComponent } from '../../shared/components/icon/icon.component';
import { DATASET_SOURCES } from '../../shared/utils/dataset-source';

const SORTERS: Record<DatasetSort, (left: DatasetSummary, right: DatasetSummary) => number> = {
    last_update: (left, right) => right.last_update.localeCompare(left.last_update),
    created_at: (left, right) => right.created_at.localeCompare(left.created_at),
    name: (left, right) => left.name.localeCompare(right.name, undefined, { sensitivity: 'base' }),
    measurements: (left, right) => right.observation_count - left.observation_count,
};

@Component({
    selector: 'adsmod-dataset-list',
    standalone: true,
    imports: [DataPaginationComponent, DatePipe, IconComponent, RouterLink],
    template: `
        <section class="dataset-list" aria-labelledby="dataset-list-title">
            <header class="dataset-list-head">
                <h2 id="dataset-list-title">Datasets <span class="dataset-list-count">{{ filtered().length }}</span></h2>
            </header>

            <div class="dataset-list-tools" role="search">
                <input
                    type="search"
                    aria-label="Search datasets"
                    placeholder="Search datasets"
                    [value]="state().query"
                    (input)="update({ query: $any($event.target).value, page: 1 })"
                />
                <select class="select-input" aria-label="Filter by origin" [value]="state().source" (change)="update({ source: $any($event.target).value, page: 1 })">
                    <option value="all">All origins</option>
                    <option value="uploaded">Uploaded</option>
                    <option value="nist">NIST ISODB</option>
                </select>
                <select class="select-input" aria-label="Sort datasets" [value]="state().sort" (change)="update({ sort: $any($event.target).value })">
                    <option value="last_update">Recently updated</option>
                    <option value="created_at">Recently created</option>
                    <option value="name">Name</option>
                    <option value="measurements">Most measurements</option>
                </select>
                <button #addButton class="button primary icon-button" type="button" title="Upload dataset" aria-label="Upload dataset" (click)="addRequested.emit()">
                    <adsmod-icon name="upload" />
                </button>
            </div>

            @if (loading() && !datasets().length) {
                <p class="dataset-list-empty" role="status">Loading datasets…</p>
            } @else if (!datasets().length) {
                <div class="dataset-list-empty">
                    <strong>No datasets yet</strong>
                    <span>Upload a CSV or Excel file, or acquire NIST isotherms from Sources.</span>
                </div>
            } @else if (!filtered().length) {
                <p class="dataset-list-empty" role="status">No datasets match the current search.</p>
            } @else {
                <ul class="dataset-rows" aria-label="Datasets">
                    @for (dataset of visible(); track dataset.id) {
                        <li class="dataset-row" [class.highlighted]="dataset.id === highlightedId()" [attr.data-dataset-id]="dataset.id">
                            <span class="dataset-row-source" role="img" [attr.aria-label]="'Source: ' + source(dataset).label" [title]="source(dataset).label">
                                <adsmod-icon [name]="source(dataset).icon" />
                            </span>
                            <div class="dataset-row-main">
                                <div class="dataset-row-title">
                                    <a class="dataset-row-name" [routerLink]="['/datasets', dataset.id]" [title]="dataset.name">{{ dataset.name }}</a>
                                    @if (dataset.edited_experiment_count) {
                                        <span class="edited-badge" [title]="dataset.edited_experiment_count + ' experiment(s) edited locally'">Edited</span>
                                    }
                                </div>
                                <p class="dataset-row-meta">
                                    {{ dataset.experiment_count }} {{ dataset.experiment_count === 1 ? 'experiment' : 'experiments' }} ·
                                    {{ dataset.observation_count }} {{ dataset.observation_count === 1 ? 'measurement' : 'measurements' }} ·
                                    {{ dataset.material_count }} {{ dataset.material_count === 1 ? 'material' : 'materials' }} ·
                                    {{ dataset.species_count }} species
                                </p>
                                <p class="dataset-row-updated">Updated {{ dataset.last_update | date: 'mediumDate' }}</p>
                            </div>
                            <div class="dataset-row-actions">
                                <a class="button quiet icon-button" [routerLink]="['/datasets', dataset.id]" title="Explore dataset" [attr.aria-label]="'Explore ' + dataset.name">
                                    <adsmod-icon name="eye" />
                                </a>
                                <button class="button quiet danger icon-button" type="button" title="Delete dataset" [attr.aria-label]="'Delete ' + dataset.name" (click)="deleteRequested.emit(dataset)">
                                    <adsmod-icon name="trash" />
                                </button>
                            </div>
                        </li>
                    }
                </ul>
            }
            @if (filtered().length) {
                <adsmod-data-pagination
                    [page]="state().page"
                    [pageSize]="state().pageSize"
                    [pageSizes]="pageSizes"
                    [total]="filtered().length"
                    itemLabel="dataset"
                    (pageChange)="update({ page: $event })"
                    (pageSizeChange)="update({ pageSize: $event, page: 1 })"
                />
            }
        </section>
    `,
    styles: [
        `
            :host { display: flex; flex-direction: column; min-width: 0; }
            .dataset-list { display: flex; flex: 1; flex-direction: column; gap: var(--spacing-sm); min-height: 0; }
            .dataset-list-head h2 { margin: 0; font-size: 1.125rem; font-weight: 600; }
            .dataset-list-count { margin-left: 0.25rem; padding: 0 7px; border-radius: 999px; background: var(--slate-100); color: var(--text-secondary); font-size: 0.75rem; font-weight: 600; vertical-align: middle; }
            .dataset-list-tools { display: grid; grid-template-columns: minmax(0, 1fr) auto auto auto; gap: var(--spacing-xs); align-items: center; }
            .dataset-list-tools .select-input { width: auto; min-width: 9.5rem; }
            .dataset-rows { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--border-color); }
            .dataset-row {
                display: grid;
                grid-template-columns: 1.5rem minmax(0, 1fr) auto;
                gap: var(--spacing-sm);
                align-items: center;
                padding: 0.5rem var(--spacing-xs);
                border-bottom: 1px solid var(--border-color);
                transition: background-color 600ms ease;
            }
            .dataset-row.highlighted { background: color-mix(in srgb, var(--primary-600) 8%, transparent); }
            .dataset-row-source { display: inline-flex; align-items: center; justify-content: center; color: var(--text-secondary); }
            .dataset-row-main { min-width: 0; }
            .dataset-row-title { display: flex; align-items: center; gap: 0.5rem; min-width: 0; }
            .dataset-row-name {
                overflow: hidden;
                color: var(--text-primary);
                font-size: 0.95rem;
                font-weight: 600;
                text-decoration: none;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-row-name:hover { color: var(--link); text-decoration: underline; }
            .edited-badge {
                flex: none;
                padding: 1px 7px;
                border-radius: 999px;
                background: color-mix(in srgb, #d97706 14%, transparent);
                color: #b45309;
                font-size: 0.6875rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .dataset-row-meta,
            .dataset-row-updated { margin: 0.1rem 0 0; font-size: 0.8125rem; font-variant-numeric: tabular-nums; }
            .dataset-row-meta { color: var(--text-secondary); }
            .dataset-row-updated { color: var(--text-tertiary); font-size: 0.75rem; }
            .dataset-row-actions { display: flex; align-items: center; gap: 0.125rem; }
            .dataset-list adsmod-data-pagination { margin-top: auto; }
            .dataset-list-empty { display: grid; gap: 0.25rem; margin: 0; padding: var(--spacing-lg) 0; color: var(--text-secondary); font-size: 0.85rem; }
            .dataset-list-empty strong { color: var(--text-primary); }
            @media (max-width: 640px) {
                .dataset-list-tools { grid-template-columns: repeat(4, minmax(0, 1fr)); }
                .dataset-list-tools input { grid-column: span 3; }
                .dataset-list-tools .icon-button { grid-column: 4; grid-row: 1; justify-self: end; }
                .dataset-list-tools .select-input { grid-column: span 2; width: 100%; min-width: 0; }
            }
        `,
    ],
})
export class DatasetListComponent {
    readonly datasets = input<DatasetSummary[]>([]);
    readonly state = input.required<DatasetListState>();
    readonly highlightedId = input<number | null>(null);
    readonly loading = input(false);
    readonly addRequested = output<void>();
    readonly deleteRequested = output<DatasetSummary>();
    readonly stateChanged = output<DatasetListState>();
    protected readonly pageSizes = [12, 25] as const;
    private readonly addButton = viewChild<ElementRef<HTMLButtonElement>>('addButton');
    private readonly host: ElementRef<HTMLElement> = inject(ElementRef);

    protected readonly filtered = computed(() => {
        const { query, source, sort } = this.state();
        const needle = query.trim().toLocaleLowerCase();
        return this.datasets()
            .filter((dataset) => source === 'all' || dataset.source === source)
            .filter(
                (dataset) =>
                    !needle ||
                    [dataset.name, dataset.description, ...dataset.material_names, ...dataset.species_names, ...dataset.tags]
                        .join(' ')
                        .toLocaleLowerCase()
                        .includes(needle),
            )
            .sort(SORTERS[sort]);
    });

    protected readonly visible = computed(() => {
        const { page, pageSize } = this.state();
        const start = (page - 1) * pageSize;
        return this.filtered().slice(start, start + pageSize);
    });

    constructor() {
        effect(() => {
            const id = this.highlightedId();
            if (id === null) return;
            queueMicrotask(() =>
                this.host.nativeElement
                    .querySelector(`[data-dataset-id="${id}"]`)
                    ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }),
            );
        });
        effect(() => {
            const { page, pageSize } = this.state();
            const pages = Math.max(1, Math.ceil(this.filtered().length / pageSize));
            if (page > pages) {
                this.stateChanged.emit({ ...this.state(), page: pages });
            }
        });
    }

    focusAddButton(): void {
        this.addButton()?.nativeElement.focus();
    }

    /** Move to the page holding a dataset, so a fresh import is visible. */
    revealDataset(id: number): void {
        const index = this.filtered().findIndex((dataset) => dataset.id === id);
        if (index < 0) {
            this.stateChanged.emit({ ...this.state(), query: '', source: 'all', page: 1 });
            return;
        }
        this.update({ page: Math.floor(index / this.state().pageSize) + 1 });
    }

    protected update(changes: Partial<DatasetListState>): void {
        this.stateChanged.emit({ ...this.state(), ...changes });
    }

    protected source(dataset: DatasetSummary): (typeof DATASET_SOURCES)[DatasetSummary['source']] {
        return DATASET_SOURCES[dataset.source];
    }
}
