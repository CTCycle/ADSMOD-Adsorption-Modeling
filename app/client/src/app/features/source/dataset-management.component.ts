// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import {
    Component,
    ElementRef,
    EventEmitter,
    Input,
    OnChanges,
    OnInit,
    Output,
    ViewChild,
    signal,
    viewChild,
} from '@angular/core';
import { RouterLink } from '@angular/router';
import type { DatasetMetadata, DatasetSummary } from '../../models/dataset.model';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import {
    DatasetActionDialogsComponent,
    type DatasetRenameEvent,
} from './dataset-action-dialogs.component';

export type { DatasetRenameEvent };

export interface DatasetListState {
    page: number;
    pageSize: number;
}

@Component({
    selector: 'adsmod-dataset-management',
    standalone: true,
    imports: [DataPaginationComponent, DatasetActionDialogsComponent, RouterLink],
    template: `
        <section class="console-card workspace-datasets-card" aria-labelledby="workspace-datasets-title">
            <div class="card-title-row workspace-datasets-head">
                <div>
                    <h2 id="workspace-datasets-title">Workspace datasets</h2>
                    <p>Open a dataset to inspect its experiments, observations and data preview.</p>
                </div>
                <button #addDatasetButton class="button primary" type="button" (click)="addRequested.emit()">Add dataset</button>
            </div>

            @if (datasets.length) {
                <div class="dataset-table-frame">
                    <table class="data-table dataset-table">
                        <caption class="sr-only">Workspace datasets</caption>
                        <thead>
                            <tr>
                                <th class="col-dataset" scope="col">Dataset</th>
                                <th class="col-origin" scope="col">Origin</th>
                                <th class="col-count num" scope="col">Experiments</th>
                                <th class="col-count num" scope="col">Observations</th>
                                <th class="col-tags" scope="col">Tags</th>
                                <th class="col-actions" scope="col">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            @for (dataset of visibleDatasets; track dataset.id) {
                                <tr>
                                    <td class="col-dataset">
                                        <a
                                            class="dataset-name"
                                            [routerLink]="['/datasets', dataset.id]"
                                            [title]="dataset.name"
                                        >
                                            {{ dataset.name }}
                                        </a>
                                        <span
                                            class="dataset-description"
                                            [title]="dataset.description || 'No description added yet.'"
                                        >
                                            {{ dataset.description || 'No description added yet.' }}
                                        </span>
                                    </td>
                                    <td class="col-origin" data-label="Origin">
                                        <span class="dataset-origin">{{ originLabel(dataset) }}</span>
                                    </td>
                                    <td class="col-count num" data-label="Experiments">{{ dataset.experiment_count }}</td>
                                    <td class="col-count num" data-label="Observations">{{ dataset.observation_count }}</td>
                                    <td class="col-tags" data-label="Tags">
                                        <span class="dataset-tags" [title]="tagsLabel(dataset)">{{ tagsLabel(dataset) }}</span>
                                    </td>
                                    <td class="col-actions">
                                        <div class="table-actions">
                                            <a
                                                class="table-action"
                                                [routerLink]="['/datasets', dataset.id]"
                                                [attr.aria-label]="'Open ' + dataset.name"
                                            >
                                                Open
                                            </a>
                                            <button
                                                class="table-action"
                                                type="button"
                                                [title]="'Edit metadata for ' + dataset.name"
                                                [attr.aria-label]="'Edit metadata for ' + dataset.name"
                                                (click)="dialogs().editMetadata(dataset)"
                                            >
                                                Edit
                                            </button>
                                            <button
                                                class="table-action"
                                                type="button"
                                                [title]="'Rename ' + dataset.name"
                                                [attr.aria-label]="'Rename ' + dataset.name"
                                                (click)="dialogs().rename(dataset)"
                                            >
                                                Rename
                                            </button>
                                            <button
                                                class="table-action danger"
                                                type="button"
                                                [title]="'Delete ' + dataset.name"
                                                [attr.aria-label]="'Delete ' + dataset.name"
                                                (click)="dialogs().remove(dataset)"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            }
                        </tbody>
                    </table>
                </div>
                <adsmod-data-pagination
                    class="dataset-pagination"
                    [page]="page()"
                    [pageSize]="pageSize()"
                    [total]="datasets.length"
                    itemLabel="dataset"
                    (pageChange)="setPage($event)"
                    (pageSizeChange)="setPageSize($event)"
                />
            } @else {
                <p class="dataset-empty">No workspace datasets yet. Add a CSV or Excel file to begin.</p>
            }
        </section>

        <adsmod-dataset-action-dialogs
            (renamed)="renamed.emit($event)"
            (metadataSaved)="metadataSaved.emit($event)"
            (deleted)="deleted.emit($event)"
        />
    `,
    styles: [
        `
            :host {
                display: block;
            }
            .workspace-datasets-card {
                padding: 0;
            }
            .workspace-datasets-head {
                align-items: flex-start;
                padding: var(--spacing-md) var(--spacing-lg) 0;
            }
            .workspace-datasets-head h2 {
                margin: 0;
                font-size: 1.125rem;
            }
            .workspace-datasets-head p {
                margin: 4px 0 0;
                color: var(--console-text-muted, #657386);
                font-size: 0.82rem;
            }
            .dataset-table-frame {
                margin: var(--spacing-sm) var(--spacing-lg) 0;
                border: 1px solid var(--slate-200);
                border-radius: 10px;
            }
            .dataset-table {
                width: 100%;
                table-layout: fixed;
                border-collapse: collapse;
            }
            .dataset-table .col-dataset {
                width: auto;
            }
            .dataset-table .col-origin {
                width: 110px;
            }
            .dataset-table .col-count {
                width: 108px;
            }
            .dataset-table .col-tags {
                width: 180px;
            }
            .dataset-table .col-actions {
                width: 262px;
            }
            .dataset-name {
                display: block;
                max-width: 100%;
                padding: 0;
                overflow: hidden;
                color: var(--console-text, #1c2a45);
                font: inherit;
                font-size: 0.84rem;
                font-weight: 700;
                text-align: left;
                text-decoration: none;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-name:hover {
                color: var(--primary-600, #2563eb);
                text-decoration: underline;
            }
            .dataset-name:focus-visible {
                outline: 2px solid var(--primary-300, #93b4f5);
                outline-offset: 2px;
            }
            .dataset-description,
            .dataset-tags {
                display: block;
                max-width: 100%;
                overflow: hidden;
                color: var(--console-text-muted, #657386);
                font-size: 0.72rem;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-description {
                margin-top: 1px;
            }
            .dataset-origin {
                display: inline-flex;
                padding: 2px 7px;
                border-radius: 999px;
                background: var(--slate-50);
                color: var(--link);
                font-size: 0.66rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .dataset-empty {
                margin: 0;
                padding: var(--spacing-md) var(--spacing-lg) var(--spacing-lg);
                color: var(--console-text-muted, #657386);
                font-size: 0.85rem;
            }
            .dataset-pagination {
                display: block;
                padding: 0 var(--spacing-lg) var(--spacing-md);
            }
            @media (max-width: 960px) {
                .dataset-table thead {
                    position: absolute;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                    clip-path: inset(50%);
                }
                .dataset-table,
                .dataset-table tbody,
                .dataset-table tr,
                .dataset-table td {
                    display: block;
                    width: auto !important;
                }
                .dataset-table tr {
                    padding: 0.6rem 0.2rem;
                    border-bottom: 1px solid var(--slate-200);
                }
                .dataset-table tr:last-child {
                    border-bottom: 0;
                }
                .dataset-table td {
                    padding: 0.2rem 0.75rem;
                    border: 0;
                    text-align: left;
                }
                .dataset-table td[data-label] {
                    display: grid;
                    grid-template-columns: 7.5rem minmax(0, 1fr);
                    gap: 0.5rem;
                }
                .dataset-table td[data-label]::before {
                    content: attr(data-label);
                    color: var(--slate-500);
                    font-size: 0.72rem;
                    font-weight: 700;
                }
                .dataset-table td.num {
                    text-align: left;
                }
            }
            @media (max-width: 760px) {
                .workspace-datasets-head {
                    padding: var(--spacing-md) var(--spacing-md) 0;
                    flex-direction: column;
                    align-items: stretch;
                    gap: var(--spacing-sm);
                }
                .workspace-datasets-head > button {
                    align-self: flex-start;
                }
                .dataset-table-frame {
                    margin-inline: var(--spacing-md);
                }
                .dataset-pagination {
                    padding-inline: var(--spacing-md);
                }
            }
        `,
    ],
})
export class DatasetManagementComponent implements OnInit, OnChanges {
    @Input() datasets: DatasetSummary[] = [];
    @Input() initialState: DatasetListState = { page: 1, pageSize: 10 };
    @Output() readonly addRequested = new EventEmitter<void>();
    @Output() readonly deleted = new EventEmitter<number>();
    @Output() readonly renamed = new EventEmitter<DatasetRenameEvent>();
    @Output() readonly metadataSaved = new EventEmitter<{ id: number; metadata: DatasetMetadata }>();
    @Output() readonly stateChanged = new EventEmitter<DatasetListState>();
    @ViewChild('addDatasetButton') private addDatasetButton?: ElementRef<HTMLButtonElement>;
    protected readonly dialogs = viewChild.required(DatasetActionDialogsComponent);
    readonly page = signal(1);
    readonly pageSize = signal(10);

    protected get visibleDatasets(): DatasetSummary[] {
        const start = (this.page() - 1) * this.pageSize();
        return this.datasets.slice(start, start + this.pageSize());
    }

    ngOnInit(): void {
        this.pageSize.set(this.initialState.pageSize);
        this.page.set(this.initialState.page);
        this.clampPage();
    }

    ngOnChanges(): void {
        this.clampPage();
    }

    focusAddButton(): void {
        this.addDatasetButton?.nativeElement.focus();
    }

    protected originLabel(dataset: DatasetSummary): string {
        return dataset.source === 'nist' ? 'NIST' : 'Workspace';
    }

    protected tagsLabel(dataset: DatasetSummary): string {
        return dataset.tags.length ? dataset.tags.join(', ') : 'No tags';
    }

    protected setPage(page: number): void {
        const max = Math.max(1, Math.ceil(this.datasets.length / this.pageSize()));
        this.page.set(Math.min(Math.max(1, page), max));
        this.emitState();
    }

    protected setPageSize(size: number): void {
        if (size === this.pageSize()) return;
        this.pageSize.set(size);
        this.page.set(1);
        this.emitState();
    }

    private clampPage(): void {
        if (this.datasets.length === 0) return;
        const max = Math.max(1, Math.ceil(this.datasets.length / this.pageSize()));
        if (this.page() > max) {
            this.page.set(max);
            this.emitState();
        }
    }

    private emitState(): void {
        this.stateChanged.emit({ page: this.page(), pageSize: this.pageSize() });
    }
}
