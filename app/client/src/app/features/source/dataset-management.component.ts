import { Component, EventEmitter, Input, OnChanges, Output, signal } from '@angular/core';
import type { DatasetMetadata, DatasetSummary } from '../../models/dataset.model';

export interface DatasetRenameEvent { id: number; newName: string; }

@Component({
    selector: 'adsmod-dataset-management',
    standalone: true,
    template: `
        <section class="console-card user-dataset-list-card">
            <div class="card-title-row"><div><h2>Workspace datasets</h2><p>Select a canonical dataset to inspect its experiments.</p></div></div>
            <div class="dataset-add-row"><button class="empty-state-icon add-dataset-button" type="button" aria-label="Add dataset" (click)="addRequested.emit()">＋</button><div><h3>{{ datasets.length ? 'Add another dataset' : 'Your workspace is ready' }}</h3><p>Atomic one-observation-per-row files are recommended.</p></div></div>
            @if (!datasets.length) { <p class="empty-state-copy">No datasets have been imported yet.</p> }
            @else { <div class="dataset-record-list" role="list" aria-label="Workspace datasets">@for (dataset of datasets; track dataset.id) {
                <article class="dataset-record" role="listitem" [class.selected]="dataset.id === selected">
                    <div class="dataset-record-copy"><div class="dataset-record-heading"><h3>{{ dataset.name }}</h3><span class="dataset-source">{{ dataset.source }}</span></div><p>{{ dataset.description || 'No description added yet.' }}</p></div>
                    <div class="dataset-record-info"><span>{{ dataset.experiment_count }} experiments</span><span>{{ dataset.observation_count }} observations</span><span>{{ dataset.tags.join(', ') || 'No tags' }}</span></div>
                    <div class="dataset-record-actions"><button class="button primary" type="button" (click)="opened.emit(dataset.id)">Select</button><button class="button secondary" type="button" (click)="editMetadata(dataset.id)">Edit metadata</button><button class="button quiet" type="button" (click)="rename(dataset)">Rename</button><button class="button quiet danger" type="button" (click)="deleted.emit(dataset.id)">Delete</button></div>
                </article>
            }</div> }
        </section>
        @if (metadataEditing()) { <div class="dataset-modal-backdrop"><section class="dataset-metadata-editor" role="dialog" aria-modal="true"><div class="dataset-metadata-header"><div><p class="eyebrow">Dataset actions</p><h2>Edit metadata</h2></div><div class="dataset-metadata-actions"><button class="button primary" type="button" (click)="saveMetadata()">Save metadata</button><button class="button quiet" type="button" aria-label="Close" (click)="metadataEditing.set(false)">×</button></div></div><div class="dataset-metadata-body"><label class="metadata-field"><span>Tags</span><input class="metadata-input" [value]="tags()" (input)="tags.set(read($event))" /></label><label class="metadata-field"><span>Description</span><textarea class="metadata-input metadata-textarea" [value]="description()" (input)="description.set(read($event))"></textarea></label></div></section></div> }
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
    readonly tags = signal(''); readonly description = signal(''); readonly metadataEditing = signal(false); private editingId: number | null = null;
    ngOnChanges(): void { const d = this.datasets.find((item) => item.id === this.selected); this.tags.set(d?.tags.join(', ') ?? ''); this.description.set(d?.description ?? ''); }
    protected read(event: Event): string { const target = event.target; return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ? target.value : ''; }
    protected editMetadata(id: number): void { const d = this.datasets.find((item) => item.id === id); this.editingId = id; this.tags.set(d?.tags.join(', ') ?? ''); this.description.set(d?.description ?? ''); this.metadataEditing.set(true); }
    protected saveMetadata(): void { if (this.editingId !== null) this.metadataSaved.emit({ id: this.editingId, metadata: { tags: this.tags().split(',').map((tag) => tag.trim()).filter(Boolean), description: this.description().trim() } }); this.metadataEditing.set(false); }
    protected rename(dataset: DatasetSummary): void { const newName = window.prompt('New dataset name', dataset.name)?.trim(); if (newName && newName !== dataset.name) this.renamed.emit({ id: dataset.id, newName }); }
}
