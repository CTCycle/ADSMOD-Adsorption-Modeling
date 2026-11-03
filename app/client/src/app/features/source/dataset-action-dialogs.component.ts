// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, output, signal } from '@angular/core';
import type {
    DatasetMetadata,
    DatasetSummary,
} from '../../models/dataset.model';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';

export interface DatasetRenameEvent {
    id: number;
    newName: string;
}

/** Edit-metadata, rename and delete dialogs shared by the list and details pages. */
@Component({
    selector: 'adsmod-dataset-action-dialogs',
    standalone: true,
    imports: [ConfirmDialogComponent],
    template: `
        @if (metadataTarget(); as dataset) {
            <div class="dataset-modal-backdrop" (click)="metadataTarget.set(null)">
                <section class="dataset-metadata-editor" role="dialog" aria-modal="true" aria-labelledby="dataset-metadata-title" (click)="$event.stopPropagation()">
                    <div class="dataset-metadata-header">
                        <div>
                            <h2 id="dataset-metadata-title">Edit metadata</h2>
                        </div>
                        <button class="button quiet dataset-metadata-close" type="button" aria-label="Close" title="Close" (click)="metadataTarget.set(null)">×</button>
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
                    <div class="dataset-metadata-footer">
                        <button class="button secondary" type="button" (click)="metadataTarget.set(null)">Cancel</button>
                        <button class="button primary" type="button" (click)="saveMetadata(dataset)">Save metadata</button>
                    </div>
                </section>
            </div>
        }

        @if (renameTarget(); as dataset) {
            <div class="dataset-modal-backdrop" (click)="renameTarget.set(null)">
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
                        <button class="button quiet dataset-metadata-close" type="button" aria-label="Close" title="Close" (click)="renameTarget.set(null)">×</button>
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
                    <div class="dataset-metadata-footer">
                        <button class="button secondary" type="button" (click)="renameTarget.set(null)">Cancel</button>
                        <button class="button primary" type="submit">Save name</button>
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
})
export class DatasetActionDialogsComponent {
    readonly renamed = output<DatasetRenameEvent>();
    readonly metadataSaved = output<{ id: number; metadata: DatasetMetadata }>();
    readonly deleted = output<number>();

    protected readonly tags = signal('');
    protected readonly description = signal('');
    protected readonly renameName = signal('');
    protected readonly metadataTarget = signal<DatasetSummary | null>(null);
    protected readonly renameTarget = signal<DatasetSummary | null>(null);
    protected readonly pendingDelete = signal<DatasetSummary | null>(null);

    editMetadata(dataset: DatasetSummary): void {
        this.tags.set(dataset.tags.join(', '));
        this.description.set(dataset.description);
        this.metadataTarget.set(dataset);
    }

    rename(dataset: DatasetSummary): void {
        this.renameName.set(dataset.name);
        this.renameTarget.set(dataset);
    }

    remove(dataset: DatasetSummary): void {
        this.pendingDelete.set(dataset);
    }

    protected read(event: Event): string {
        const target = event.target;
        return target instanceof HTMLInputElement ||
            target instanceof HTMLTextAreaElement
            ? target.value
            : '';
    }

    protected saveMetadata(dataset: DatasetSummary): void {
        this.metadataSaved.emit({
            id: dataset.id,
            metadata: {
                tags: this.tags()
                    .split(',')
                    .map((tag) => tag.trim())
                    .filter(Boolean),
                description: this.description().trim(),
            },
        });
        this.metadataTarget.set(null);
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
