import { Component, EventEmitter, Input, OnChanges, Output, signal } from '@angular/core';
import type { DatasetMetadata, DatasetSummary } from '../../models/dataset.model';
export interface DatasetRenameEvent {
    name: string;
    newName: string;
}

@Component({ selector: 'adsmod-dataset-management', standalone: true, template: `
<section class="console-card user-dataset-list-card"><div class="card-title-row"><div><h2>Workspace datasets</h2></div></div>
<div class="dataset-add-row"><button class="empty-state-icon add-dataset-button" type="button" aria-label="Add dataset" (click)="addRequested.emit()">＋</button><div>@if (!datasets.length) { <h3>Your workspace is ready</h3><p>Use + to add a CSV, TXT, XLSX, or JSON dataset.</p> } @else { <h3>Add another dataset</h3><p>Use + to add another CSV, TXT, XLSX, or JSON dataset.</p> }</div></div>
@if (datasets.length) { <div class="dataset-record-list" role="list" aria-label="Workspace datasets">@for (dataset of datasets; track dataset.name) { <article class="dataset-record" role="listitem" [class.selected]="dataset.name === selected"><div class="dataset-record-copy"><div class="dataset-record-heading"><h3>{{ dataset.name }}</h3><span class="dataset-source">{{ dataset.source }}</span></div><p>{{ dataset.description || 'No description added yet.' }}</p></div><div class="dataset-record-info"><div class="dataset-record-stats"><span>{{ dataset.row_count }} rows</span><span>{{ dataset.column_count }} columns</span></div><div class="dataset-record-meta"><span>{{ dataset.tags.join(', ') || 'No tags' }}</span></div></div><div class="dataset-record-actions"><button class="button secondary" type="button" (click)="opened.emit(dataset.name)" [attr.aria-label]="'Open spreadsheet for ' + dataset.name">Open spreadsheet</button><button class="button secondary" type="button" (click)="editMetadata(dataset.name)" [attr.aria-label]="'Edit metadata for ' + dataset.name">Edit metadata</button><button class="button quiet" type="button" (click)="rename(dataset.name)" [attr.aria-label]="'Rename dataset ' + dataset.name">Rename</button><button class="button quiet danger" type="button" (click)="deleted.emit(dataset.name)" [attr.aria-label]="'Delete dataset ' + dataset.name">Delete</button></div></article> }</div> }
</section>
@if (metadataEditing()) { <div class="dataset-modal-backdrop"><section class="dataset-metadata-editor" role="dialog" aria-modal="true" aria-labelledby="dataset-metadata-title"><div class="dataset-metadata-header"><div><p class="eyebrow">Dataset actions</p><h2 id="dataset-metadata-title">Edit metadata</h2><p class="dataset-metadata-subtitle">Refine the tags and description for this dataset.</p></div><div class="dataset-metadata-actions"><button class="button primary dataset-metadata-save" type="button" (click)="saveMetadata()">Save metadata</button><button class="button quiet dataset-metadata-close" type="button" aria-label="Close metadata editor" title="Close metadata editor" (click)="metadataEditing.set(false)"><span aria-hidden="true">×</span></button></div></div><div class="dataset-metadata-body"><label class="metadata-field"><span>Tags</span><small>Separate multiple tags with commas.</small><input class="metadata-input" [value]="tags()" placeholder="e.g. zeolite, benchmark" (input)="tags.set(read($event))" /></label><label class="metadata-field"><span>Description</span><small>Add context that will help identify this dataset later.</small><textarea class="metadata-input metadata-textarea" [value]="description()" placeholder="Describe the source, conditions, or intended use." (input)="description.set(read($event))"></textarea></label></div></section></div> }` })
export class DatasetManagementComponent implements OnChanges {
    @Input() datasets: DatasetSummary[] = [];
    @Input() selected: string | null = null;

    @Output() readonly opened = new EventEmitter<string>();
    @Output() readonly addRequested = new EventEmitter<void>();
    @Output() readonly metadataRequested = new EventEmitter<string>();
    @Output() readonly deleted = new EventEmitter<string>();
    @Output() readonly renamed = new EventEmitter<DatasetRenameEvent>();
    @Output() readonly metadataSaved = new EventEmitter<DatasetMetadata>();

    readonly tags = signal('');
    readonly description = signal('');
    readonly metadataEditing = signal(false);

    ngOnChanges(): void {
        const dataset = this.datasets.find((item) => item.name === this.selected);
        this.tags.set(dataset?.tags.join(', ') ?? '');
        this.description.set(dataset?.description ?? '');
    }

    protected read(event: Event): string {
        const target = event.target;
        return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ? target.value : '';
    }

    protected parsedTags(): string[] {
        return this.tags().split(',').map((tag) => tag.trim()).filter(Boolean);
    }

    protected editMetadata(name: string): void {
        const dataset = this.datasets.find((item) => item.name === name);
        this.tags.set(dataset?.tags.join(', ') ?? '');
        this.description.set(dataset?.description ?? '');
        this.metadataEditing.set(true);
        this.metadataRequested.emit(name);
    }

    protected saveMetadata(): void {
        this.metadataSaved.emit({ tags: this.parsedTags(), description: this.description() });
        this.metadataEditing.set(false);
    }

    protected rename(name: string): void {
        const newName = window.prompt('New dataset name', name);
        if (newName && newName !== name) {
            this.renamed.emit({ name, newName });
        }
    }
}
