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
                <div>
                    <h2 id="dataset-list-title">Adsorption datasets</h2>
                    <p>Each dataset groups one or more experiments, and each experiment holds its measurements.</p>
                </div>
                <button #addButton class="button primary" type="button" (click)="addRequested.emit()">Add dataset</button>
            </header>

            <div class="dataset-list-tools" role="search">
                <input
                    type="search"
                    aria-label="Search datasets"
                    placeholder="Search name, description, material, or species"
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
                                @if (dataset.description) {
                                    <p class="dataset-row-description" [title]="dataset.description">{{ dataset.description }}</p>
                                }
                                <dl class="dataset-row-facts">
                                    <div><dt>Experiments</dt><dd>{{ dataset.experiment_count }}</dd></div>
                                    <div><dt>Measurements</dt><dd>{{ dataset.observation_count }}</dd></div>
                                    <div class="wide"><dt>Materials</dt><dd [title]="dataset.material_names.join(', ')">{{ namesLabel(dataset.material_names, dataset.material_count) }}</dd></div>
                                    <div class="wide"><dt>Species</dt><dd [title]="dataset.species_names.join(', ')">{{ namesLabel(dataset.species_names, dataset.species_count) }}</dd></div>
                                    <div><dt>Created</dt><dd>{{ dataset.created_at | date: 'mediumDate' }}</dd></div>
                                    <div><dt>Updated</dt><dd [title]="dataset.last_update | date: 'medium'">{{ dataset.last_update | date: 'mediumDate' }}</dd></div>
                                </dl>
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
            .dataset-list { display: grid; gap: var(--spacing-sm); }
            .dataset-list-head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--spacing-md); }
            .dataset-list-head h2 { margin: 0; font-size: 1.05rem; }
            .dataset-list-head p { margin: 2px 0 0; color: var(--text-secondary); font-size: 0.8rem; }
            .dataset-list-tools { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: var(--spacing-xs); }
            .dataset-list-tools .select-input { width: auto; min-width: 9.5rem; }
            .dataset-rows { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--border-color); }
            .dataset-row {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: var(--spacing-md);
                align-items: center;
                padding: 0.7rem 0.25rem;
                border-bottom: 1px solid var(--border-color);
                transition: background-color 600ms ease;
            }
            .dataset-row.highlighted { background: color-mix(in srgb, var(--primary-600) 8%, transparent); }
            .dataset-row-main { min-width: 0; }
            .dataset-row-title { display: flex; align-items: center; gap: 0.5rem; min-width: 0; }
            .dataset-row-name {
                overflow: hidden;
                color: var(--text-primary);
                font-size: 0.9rem;
                font-weight: 700;
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
                font-size: 0.64rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .origin-badge.nist { color: var(--link); }
            .edited-badge { background: color-mix(in srgb, #d97706 14%, transparent); color: #b45309; }
            .dataset-row-description {
                margin: 2px 0 0;
                overflow: hidden;
                color: var(--text-secondary);
                font-size: 0.76rem;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-row-facts {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(6.5rem, 1fr));
                gap: 0.25rem var(--spacing-md);
                margin: 0.45rem 0 0;
            }
            .dataset-row-facts div { min-width: 0; }
            .dataset-row-facts .wide { grid-column: span 2; }
            .dataset-row-facts dt { color: var(--text-tertiary); font-size: 0.62rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }
            .dataset-row-facts dd { margin: 0; overflow: hidden; font-size: 0.78rem; font-variant-numeric: tabular-nums; text-overflow: ellipsis; white-space: nowrap; }
            .dataset-row-actions { display: flex; gap: 0.35rem; }
            .dataset-list-empty { display: grid; gap: 0.25rem; margin: 0; padding: var(--spacing-lg) 0; color: var(--text-secondary); font-size: 0.85rem; }
            .dataset-list-empty strong { color: var(--text-primary); }
            @media (max-width: 640px) {
                .dataset-list-head { flex-direction: column; }
                .dataset-list-tools { grid-template-columns: 1fr 1fr; }
                .dataset-list-tools input { grid-column: span 2; }
                .dataset-list-tools .select-input { width: 100%; min-width: 0; }
                .dataset-row { grid-template-columns: 1fr; }
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

    protected namesLabel(names: string[], total: number): string {
        if (!total) return '—';
        const extra = total - names.length;
        return extra > 0 ? `${names.join(', ')} +${extra}` : names.join(', ');
    }
}
