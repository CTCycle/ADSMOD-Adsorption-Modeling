// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import { Component, ElementRef, computed, effect, inject, input, output, viewChild } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { DatasetListState, DatasetSort } from '../../core/state/core-workspace.store';
import type { DatasetSummary } from '../../models/dataset.model';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';

const SORTERS: Record<DatasetSort, (left: DatasetSummary, right: DatasetSummary) => number> = {
    last_update: (left, right) => right.last_update.localeCompare(left.last_update),
    created_at: (left, right) => right.created_at.localeCompare(left.created_at),
    name: (left, right) => left.name.localeCompare(right.name, undefined, { sensitivity: 'base' }),
    measurements: (left, right) => right.observation_count - left.observation_count,
};

export const SOURCE_LABELS: Record<DatasetSummary['source'], string> = {
    uploaded: 'Uploaded',
    nist: 'NIST ISODB',
};

@Component({
    selector: 'adsmod-dataset-list',
    standalone: true,
    imports: [DataPaginationComponent, DatePipe, RouterLink],
    template: `
        <section class="dataset-list" aria-labelledby="dataset-list-title">
            <header class="dataset-list-head">
                <h2 id="dataset-list-title">Datasets <span class="dataset-list-count">{{ filtered().length }}</span></h2>
                <button #addButton class="button primary" type="button" (click)="addRequested.emit()">Add dataset</button>
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
            </div>

            @if (loading() && !datasets().length) {
                <p class="dataset-list-empty" role="status">Loading datasets…</p>
            } @else if (!datasets().length) {
                <div class="dataset-list-empty">
                    <strong>No datasets yet</strong>
                    <span>Add a CSV or Excel file, or acquire NIST isotherms from Sources.</span>
                </div>
            } @else if (!filtered().length) {
                <p class="dataset-list-empty" role="status">No datasets match the current search.</p>
            } @else {
                <ul class="dataset-rows" aria-label="Datasets">
                    @for (dataset of visible(); track dataset.id) {
                        <li class="dataset-row" [class.highlighted]="dataset.id === highlightedId()" [attr.data-dataset-id]="dataset.id">
                            <div class="dataset-row-main">
                                <div class="dataset-row-title">
                                    <a class="dataset-row-name" [routerLink]="['/datasets', dataset.id]" [title]="dataset.name">{{ dataset.name }}</a>
                                    <span class="origin-badge" [class.nist]="dataset.source === 'nist'">{{ sourceLabel(dataset) }}</span>
                                    @if (dataset.edited_experiment_count) {
                                        <span class="edited-badge" [title]="dataset.edited_experiment_count + ' experiment(s) edited locally'">Edited</span>
                                    }
                                </div>
                                <p class="dataset-row-meta">
                                    {{ dataset.experiment_count }} {{ dataset.experiment_count === 1 ? 'experiment' : 'experiments' }} ·
                                    {{ dataset.observation_count }} measurements · Updated {{ dataset.last_update | date: 'mediumDate' }}
                                </p>
                                <p class="dataset-row-names" [title]="detailTitle(dataset)">{{ detailLabel(dataset) }}</p>
                            </div>
                            <div class="dataset-row-actions">
                                <a class="button secondary" [routerLink]="['/datasets', dataset.id]" [attr.aria-label]="'Explore ' + dataset.name">Explore</a>
                                <button class="button quiet danger" type="button" [attr.aria-label]="'Delete ' + dataset.name" (click)="deleteRequested.emit(dataset)">Delete</button>
                            </div>
                        </li>
                    }
                </ul>
                <adsmod-data-pagination
                    [page]="state().page"
                    [pageSize]="state().pageSize"
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
            :host { display: block; min-width: 0; }
            .dataset-list { display: grid; gap: var(--spacing-md); }
            .dataset-list-head { display: flex; align-items: center; justify-content: space-between; gap: var(--spacing-md); }
            .dataset-list-head h2 { margin: 0; font-size: 1.125rem; font-weight: 600; }
            .dataset-list-count { margin-left: 0.25rem; padding: 0 7px; border-radius: 999px; background: var(--slate-100); color: var(--text-secondary); font-size: 0.75rem; font-weight: 600; vertical-align: middle; }
            .dataset-list-tools { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: var(--spacing-xs); }
            .dataset-list-tools .select-input { width: auto; min-width: 9.5rem; }
            .dataset-rows { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--border-color); }
            .dataset-row {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: var(--spacing-md);
                align-items: center;
                padding: var(--spacing-md) var(--spacing-xs);
                border-bottom: 1px solid var(--border-color);
                transition: background-color 600ms ease;
            }
            .dataset-row:last-child { border-bottom: 0; }
            .dataset-row.highlighted { background: color-mix(in srgb, var(--primary-600) 8%, transparent); }
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
            .origin-badge, .edited-badge {
                flex: none;
                padding: 1px 7px;
                border-radius: 999px;
                background: var(--slate-100);
                color: var(--text-secondary);
                font-size: 0.6875rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .origin-badge.nist { color: var(--link); }
            .edited-badge { background: color-mix(in srgb, #d97706 14%, transparent); color: #b45309; }
            .dataset-row-meta,
            .dataset-row-names {
                margin: 0.2rem 0 0;
                overflow: hidden;
                font-size: 0.8125rem;
                font-variant-numeric: tabular-nums;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-row-meta { color: var(--text-secondary); }
            .dataset-row-names { color: var(--text-tertiary); }
            .dataset-row-actions { display: flex; align-items: center; gap: var(--spacing-xs); }
            .dataset-list-empty { display: grid; gap: 0.25rem; margin: 0; padding: var(--spacing-lg) 0; color: var(--text-secondary); font-size: 0.85rem; }
            .dataset-list-empty strong { color: var(--text-primary); }
            @media (max-width: 640px) {
                .dataset-list-tools { grid-template-columns: 1fr 1fr; }
                .dataset-list-tools input { grid-column: span 2; }
                .dataset-list-tools .select-input { width: 100%; min-width: 0; }
                .dataset-row { grid-template-columns: 1fr; }
                .dataset-row-actions .button { flex: 1; }
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

    protected sourceLabel(dataset: DatasetSummary): string {
        return SOURCE_LABELS[dataset.source];
    }

    protected detailLabel(dataset: DatasetSummary): string {
        return [
            this.namesLabel(dataset.material_names, dataset.material_count),
            this.namesLabel(dataset.species_names, dataset.species_count),
        ]
            .filter(Boolean)
            .join(' · ');
    }

    protected detailTitle(dataset: DatasetSummary): string {
        return [...dataset.material_names, ...dataset.species_names].join(', ');
    }

    private namesLabel(names: string[], total: number): string {
        if (!total) return '';
        const shown = names.slice(0, 2);
        const extra = total - shown.length;
        return extra > 0 ? `${shown.join(', ')} +${extra}` : shown.join(', ');
    }
}
