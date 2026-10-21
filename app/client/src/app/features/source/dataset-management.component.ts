// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, EventEmitter, Input, OnChanges, Output, ViewChild, signal } from '@angular/core';
import type { DatasetMetadata, DatasetSummary } from '../../models/dataset.model';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';

export interface DatasetRenameEvent { id: number; newName: string; }

@Component({
    selector: 'adsmod-dataset-management',
    standalone: true,
    imports: [ConfirmDialogComponent, DataPaginationComponent],
    template: `
        <section class="console-card workspace-datasets-card" aria-labelledby="workspace-datasets-title">
            <div class="card-title-row workspace-datasets-head">
                <div>
                    <h2 id="workspace-datasets-title">Workspace datasets</h2>
                    <p>Select a dataset to inspect its experiments.</p>
                </div>
                <button #addDatasetButton class="button primary" type="button" (click)="addRequested.emit()">Add dataset</button>
            </div>

            @if (datasets.length) {
                <div class="data-table-frame dataset-table-frame">
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
                                <tr [class.selected]="dataset.id === selected">
                                    <td class="col-dataset">
                                        <button
                                            class="dataset-name"
                                            type="button"
                                            [title]="dataset.name"
                                            (click)="opened.emit(dataset.id)"
                                        >
                                            {{ dataset.name }}
                                        </button>
                                        <span
                                            class="dataset-description"
                                            [title]="dataset.description || 'No description added yet.'"
                                        >
                                            {{ dataset.description || 'No description added yet.' }}
                                        </span>
                                    </td>
                                    <td class="col-origin">
                                        <span class="dataset-origin">{{ originLabel(dataset) }}</span>
                                    </td>
                                    <td class="col-count num">{{ dataset.experiment_count }}</td>
                                    <td class="col-count num">{{ dataset.observation_count }}</td>
                                    <td class="col-tags">
                                        <span class="dataset-tags" [title]="tagsLabel(dataset)">{{ tagsLabel(dataset) }}</span>
                                    </td>
                                    <td class="col-actions">
                                        <div class="table-actions">
                                            <button
                                                class="table-action"
                                                type="button"
                                                [attr.aria-label]="'Open ' + dataset.name"
                                                (click)="opened.emit(dataset.id)"
                                            >
                                                Open
                                            </button>
                                            <button
                                                class="table-action"
                                                type="button"
                                                [title]="'Edit metadata for ' + dataset.name"
                                                [attr.aria-label]="'Edit metadata for ' + dataset.name"
                                                (click)="editMetadata(dataset.id)"
                                            >
                                                Edit
                                            </button>
                                            <button
                                                class="table-action"
                                                type="button"
                                                [title]="'Rename ' + dataset.name"
                                                [attr.aria-label]="'Rename ' + dataset.name"
                                                (click)="beginRename(dataset)"
                                            >
                                                Rename
                                            </button>
                                            <button
                                                class="table-action danger"
                                                type="button"
                                                [title]="'Delete ' + dataset.name"
                                                [attr.aria-label]="'Delete ' + dataset.name"
                                                (click)="requestDelete(dataset)"
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

        @if (metadataEditing()) {
            <div class="dataset-modal-backdrop" (click)="metadataEditing.set(false)">
                <section class="dataset-metadata-editor" role="dialog" aria-modal="true" aria-labelledby="dataset-metadata-title" (click)="$event.stopPropagation()">
                    <div class="dataset-metadata-header">
                        <div>
                            <h2 id="dataset-metadata-title">Edit metadata</h2>
                        </div>
                        <div class="dataset-metadata-actions">
                            <button class="button primary" type="button" (click)="saveMetadata()">Save metadata</button>
                            <button class="button quiet" type="button" aria-label="Close" title="Close" (click)="metadataEditing.set(false)">×</button>
                        </div>
                    </div>
                    <div class="dataset-metadata-body">
                        <label class="metadata-field">
                            <span>Tags</span>
                            <input class="metadata-input" [value]="tags()" (input)="tags.set(read($event))" />
                        </label>
                        <label class="metadata-field">
                            <span>Description</span>
                            <textarea class="metadata-input metadata-textarea" [value]="description()" (input)="description.set(read($event))"></textarea>
                        </label>
                    </div>
                </section>
            </div>
        }

        @if (renameTarget(); as dataset) {
            <div class="dataset-modal-backdrop" (click)="cancelRename()">
                <form
                    class="dataset-metadata-editor"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby="dataset-rename-title"
                    (submit)="saveRename($event)"
                    (click)="$event.stopPropagation()"
                >
                    <div class="dataset-metadata-header">
                        <div>
                            <h2 id="dataset-rename-title">Rename dataset</h2>
                        </div>
                        <div class="dataset-metadata-actions">
                            <button class="button secondary" type="button" (click)="cancelRename()">Cancel</button>
                            <button class="button primary" type="submit">Save name</button>
                            <button class="button quiet" type="button" aria-label="Close" title="Close" (click)="cancelRename()">×</button>
                        </div>
                    </div>
                    <div class="dataset-metadata-body">
                        <label class="metadata-field">
                            <span>Dataset name</span>
                            <input
                                class="metadata-input"
                                type="text"
                                required
                                autofocus
                                [value]="renameName()"
                                (input)="renameName.set(read($event))"
                            />
                        </label>
                    </div>
                </form>
            </div>
        }

        <adsmod-confirm-dialog
            [open]="pendingDelete() !== null"
            title="Delete dataset?"
            [message]="deleteMessage()"
            confirmLabel="Delete dataset"
            (closed)="pendingDelete.set(null)"
            (confirmed)="confirmDelete()"
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
                max-height: min(46vh, 28rem);
                --data-table-min-width: 900px;
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
                width: 200px;
            }
            .dataset-table .col-actions {
                width: 272px;
            }
            .dataset-name {
                display: block;
                max-width: 100%;
                padding: 0;
                overflow: hidden;
                border: 0;
                background: transparent;
                color: var(--console-text, #1c2a45);
                font: inherit;
                font-size: 0.84rem;
                font-weight: 700;
                text-align: left;
                text-overflow: ellipsis;
                white-space: nowrap;
                cursor: pointer;
            }
            .dataset-name:hover {
                color: var(--primary-600, #2563eb);
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
                background: #eef4ff;
                color: #3768ad;
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
export class DatasetManagementComponent implements OnChanges {
    @Input() datasets: DatasetSummary[] = [];
    @Input() selected: number | null = null;
    @Output() readonly opened = new EventEmitter<number>();
    @Output() readonly addRequested = new EventEmitter<void>();
    @Output() readonly deleted = new EventEmitter<number>();
    @Output() readonly renamed = new EventEmitter<DatasetRenameEvent>();
    @Output() readonly metadataSaved = new EventEmitter<{ id: number; metadata: DatasetMetadata }>();
    @ViewChild('addDatasetButton') private addDatasetButton?: ElementRef<HTMLButtonElement>;
    readonly tags = signal('');
    readonly description = signal('');
    readonly metadataEditing = signal(false);
    readonly renameTarget = signal<DatasetSummary | null>(null);
    readonly renameName = signal('');
    readonly pendingDelete = signal<DatasetSummary | null>(null);
    readonly page = signal(1);
    readonly pageSize = signal(10);
    private editingId: number | null = null;

    protected get visibleDatasets(): DatasetSummary[] {
        const start = (this.page() - 1) * this.pageSize();
        return this.datasets.slice(start, start + this.pageSize());
    }

    ngOnChanges(): void {
        const dataset = this.datasets.find((item) => item.id === this.selected);
        this.tags.set(dataset?.tags.join(', ') ?? '');
        this.description.set(dataset?.description ?? '');
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
    }

    protected setPageSize(size: number): void {
        if (size === this.pageSize()) return;
        this.pageSize.set(size);
        this.page.set(1);
    }

    protected read(event: Event): string {
        const target = event.target;
        return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ? target.value : '';
    }

    protected editMetadata(id: number): void {
        const dataset = this.datasets.find((item) => item.id === id);
        this.editingId = id;
        this.tags.set(dataset?.tags.join(', ') ?? '');
        this.description.set(dataset?.description ?? '');
        this.metadataEditing.set(true);
    }

    protected saveMetadata(): void {
        if (this.editingId !== null) {
            this.metadataSaved.emit({
                id: this.editingId,
                metadata: {
                    tags: this.tags().split(',').map((tag) => tag.trim()).filter(Boolean),
                    description: this.description().trim(),
                },
            });
        }
        this.metadataEditing.set(false);
    }

    protected beginRename(dataset: DatasetSummary): void {
        this.renameName.set(dataset.name);
        this.renameTarget.set(dataset);
    }

    protected cancelRename(): void {
        this.renameTarget.set(null);
    }

    protected saveRename(event: Event): void {
        event.preventDefault();
        const dataset = this.renameTarget();
        const newName = this.renameName().trim();
        if (dataset && newName && newName !== dataset.name) {
            this.renamed.emit({ id: dataset.id, newName });
        }
        this.renameTarget.set(null);
    }

    protected requestDelete(dataset: DatasetSummary): void {
        this.pendingDelete.set(dataset);
    }

    protected deleteMessage(): string {
        const name = this.pendingDelete()?.name;
        return name
            ? `This will permanently delete “${name}”. This action cannot be undone.`
            : 'This action cannot be undone.';
    }

    protected confirmDelete(): void {
        const datasetId = this.pendingDelete()?.id;
        this.pendingDelete.set(null);
        if (datasetId !== undefined) {
            this.deleted.emit(datasetId);
        }
    }

    private clampPage(): void {
        const max = Math.max(1, Math.ceil(this.datasets.length / this.pageSize()));
        if (this.page() > max) this.page.set(max);
    }
}
