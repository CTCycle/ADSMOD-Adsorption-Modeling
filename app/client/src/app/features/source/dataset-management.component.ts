import { Component, EventEmitter, Input, OnChanges, Output, signal } from '@angular/core';
import type { DatasetMetadata, DatasetSummary } from '../../models/dataset.model';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';

export interface DatasetRenameEvent { id: number; newName: string; }

@Component({
    selector: 'adsmod-dataset-management',
    standalone: true,
    imports: [ConfirmDialogComponent],
    template: `
        <section class="console-card user-dataset-list-card" aria-labelledby="workspace-datasets-title">
            <div class="card-title-row">
                <div>
                    <h2 id="workspace-datasets-title">Workspace datasets</h2>
                    <p>Select a dataset to inspect its experiments.</p>
                </div>
            </div>
            <div class="dataset-add-row">
                <button class="button primary add-dataset-button" type="button" (click)="addRequested.emit()">Add dataset</button>
                <div>
                    <h3>{{ datasets.length ? 'Add another dataset' : 'Add your first dataset' }}</h3>
                    <p>Import CSV or Excel observations. One observation per row is recommended.</p>
                </div>
            </div>
            @if (!datasets.length) {
                <p class="empty-state-copy">No datasets have been imported yet. Add a dataset to get started.</p>
            } @else {
                <div class="dataset-record-list" role="list" aria-label="Workspace datasets">
                    @for (dataset of datasets; track dataset.id) {
                        <article class="dataset-record" role="listitem" [class.selected]="dataset.id === selected" [class.renaming]="renameTarget()?.id === dataset.id">
                            @if (renameTarget()?.id === dataset.id) {
                                <form class="dataset-rename-inline" (submit)="saveRename($event)">
                                    <h3>Rename dataset</h3>
                                    <label [for]="'dataset-name-' + dataset.id">
                                        Dataset name
                                        <input
                                            [id]="'dataset-name-' + dataset.id"
                                            type="text"
                                            required
                                            autofocus
                                            [value]="renameName()"
                                            (input)="renameName.set(read($event))"
                                        />
                                    </label>
                                    <div class="dataset-rename-actions">
                                        <button class="button secondary" type="button" (click)="cancelRename()">Cancel</button>
                                        <button class="button primary" type="submit">Save name</button>
                                    </div>
                                </form>
                            } @else {
                                <div class="dataset-record-copy">
                                    <div class="dataset-record-heading">
                                        <h3>{{ dataset.name }}</h3>
                                        <span class="dataset-source">{{ dataset.source }}</span>
                                    </div>
                                    <p>{{ dataset.description || 'No description added yet.' }}</p>
                                </div>
                                <div class="dataset-record-stats">
                                    <span>{{ dataset.experiment_count }} experiments</span>
                                    <span>{{ dataset.observation_count }} observations</span>
                                    <span>{{ dataset.tags.join(', ') || 'No tags' }}</span>
                                </div>
                                <div class="dataset-record-actions">
                                    <button class="button primary" type="button" (click)="opened.emit(dataset.id)">Select</button>
                                    <button class="button secondary" type="button" (click)="editMetadata(dataset.id)">Edit metadata</button>
                                    <button class="button quiet" type="button" (click)="beginRename(dataset)">Rename</button>
                                    <button class="button quiet danger" type="button" (click)="requestDelete(dataset)">Delete</button>
                                </div>
                            }
                        </article>
                    }
                </div>
            }
        </section>

        @if (metadataEditing()) {
            <div class="dataset-modal-backdrop" (click)="metadataEditing.set(false)">
                <section class="dataset-metadata-editor" role="dialog" aria-modal="true" aria-labelledby="dataset-metadata-title" (click)="$event.stopPropagation()">
                    <div class="dataset-metadata-header">
                        <div>
                            <p class="eyebrow">Dataset actions</p>
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

        <adsmod-confirm-dialog
            [open]="pendingDelete() !== null"
            title="Delete dataset?"
            [message]="deleteMessage()"
            confirmLabel="Delete dataset"
            (closed)="pendingDelete.set(null)"
            (confirmed)="confirmDelete()"
        />
    `,
})
export class DatasetManagementComponent implements OnChanges {
    @Input() datasets: DatasetSummary[] = [];
    @Input() selected: number | null = null;
    @Output() readonly opened = new EventEmitter<number>();
    @Output() readonly addRequested = new EventEmitter<void>();
    @Output() readonly deleted = new EventEmitter<number>();
    @Output() readonly renamed = new EventEmitter<DatasetRenameEvent>();
    @Output() readonly metadataSaved = new EventEmitter<{ id: number; metadata: DatasetMetadata }>();
    readonly tags = signal('');
    readonly description = signal('');
    readonly metadataEditing = signal(false);
    readonly renameTarget = signal<DatasetSummary | null>(null);
    readonly renameName = signal('');
    readonly pendingDelete = signal<DatasetSummary | null>(null);
    private editingId: number | null = null;

    ngOnChanges(): void {
        const dataset = this.datasets.find((item) => item.id === this.selected);
        this.tags.set(dataset?.tags.join(', ') ?? '');
        this.description.set(dataset?.description ?? '');
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
}
