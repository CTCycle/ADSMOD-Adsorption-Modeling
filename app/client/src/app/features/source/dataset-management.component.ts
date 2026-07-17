import { Component, EventEmitter, Input, OnChanges, Output, signal } from '@angular/core';
import type { DatasetMetadata, DatasetRow, DatasetRowMutation, DatasetRowsPage, DatasetSummary } from '../../models/dataset.model';
export interface DatasetRenameEvent {
    name: string;
    newName: string;
}

@Component({ selector: 'adsmod-dataset-management', standalone: true, template: `
<section class="console-card user-dataset-list-card"><div class="card-title-row"><div><p class="eyebrow">Step 2</p><h2>Workspace datasets</h2></div><button class="button secondary" type="button" (click)="refreshed.emit()">Refresh</button></div>
@if (!datasets.length) { <div class="dataset-empty-state"><span class="empty-state-icon" aria-hidden="true">＋</span><div><h3>Your workspace is ready</h3><p>Upload a CSV, TXT, XLSX, or JSON file above to start exploring your adsorption data.</p></div></div> } @else { <table class="dataset-management-table"><caption class="sr-only">Workspace datasets</caption><thead><tr><th scope="col">Name</th><th scope="col">Rows</th><th scope="col">Tags</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead><tbody>@for (dataset of datasets; track dataset.name) { <tr [class.selected]="dataset.name === selected"><td><button type="button" (click)="opened.emit(dataset.name)" [attr.aria-label]="'Open dataset ' + dataset.name">{{ dataset.name }}</button><small>{{ dataset.description }}</small></td><td>{{ dataset.row_count }}</td><td>{{ dataset.tags.join(', ') || '—' }}</td><td class="table-actions"><button type="button" (click)="rename(dataset.name)" [attr.aria-label]="'Rename dataset ' + dataset.name">Rename</button><button type="button" (click)="deleted.emit(dataset.name)" [attr.aria-label]="'Delete dataset ' + dataset.name">Delete</button></td></tr> }</tbody></table> }
</section>
@if (selected) { <section class="console-card dataset-metadata-editor"><div class="card-title-row"><h2>Metadata</h2></div><label>Tags <input [value]="tags()" (input)="tags.set(read($event))" /></label><label>Description <textarea [value]="description()" (input)="description.set(read($event))"></textarea></label><button class="button primary" type="button" (click)="metadataSaved.emit({ tags: parsedTags(), description: description() })">Save metadata</button></section> } @else { <section class="console-card compact-placeholder-card"><h2>Metadata</h2><p>Select a dataset to edit tags and description.</p></section> }
<section class="console-card user-dataset-editor-card"><h2>Spreadsheet editor</h2>@if (!selected || !page) { <div class="compact-placeholder"><span class="empty-state-icon" aria-hidden="true">▤</span><p>{{ selected ? 'Loading rows for the selected dataset…' : 'Select a dataset to edit its rows.' }}</p></div> } @else { <div class="spreadsheet-scroll"><table class="spreadsheet-table"><caption class="sr-only">Editable rows for {{ selected }}</caption><thead><tr>@for (column of page.columns; track column) { <th scope="col">{{ column }}</th> }<th scope="col">Actions</th></tr></thead><tbody>@for (row of rows(); track row.row_id; let rowIndex = $index) { <tr>@for (column of dataColumns(); track column) { <td><input [value]="value(row, column)" [attr.aria-label]="column + ' for row ' + (rowIndex + 1)" (input)="change(row.row_id, column, $event)" /></td> }<td class="table-actions"><button type="button" (click)="remove(row.row_id)" [attr.aria-label]="'Delete row ' + (rowIndex + 1)">Delete</button></td></tr> }</tbody></table></div><div class="editor-actions"><button type="button" (click)="addRow()">Add row</button><button type="button" [disabled]="!mutations().length" (click)="saved.emit(mutations())">Save changes</button><button type="button" [disabled]="page.offset === 0" (click)="paged.emit(page.offset - page.limit)">Previous</button><button type="button" [disabled]="page.offset + page.limit >= page.total_rows" (click)="paged.emit(page.offset + page.limit)">Next</button></div> }</section>` })
export class DatasetManagementComponent implements OnChanges {
    @Input() datasets: DatasetSummary[] = [];
    @Input() selected: string | null = null;
    @Input() page: DatasetRowsPage | null = null;

    @Output() readonly opened = new EventEmitter<string>();
    @Output() readonly deleted = new EventEmitter<string>();
    @Output() readonly refreshed = new EventEmitter<void>();
    @Output() readonly renamed = new EventEmitter<DatasetRenameEvent>();
    @Output() readonly metadataSaved = new EventEmitter<DatasetMetadata>();
    @Output() readonly paged = new EventEmitter<number>();
    @Output() readonly saved = new EventEmitter<DatasetRowMutation[]>();

    readonly tags = signal('');
    readonly description = signal('');
    readonly rows = signal<DatasetRow[]>([]);
    readonly mutations = signal<DatasetRowMutation[]>([]);

    ngOnChanges(): void {
        this.rows.set(this.page?.rows ?? []);
        const dataset = this.datasets.find((item) => item.name === this.selected);
        this.tags.set(dataset?.tags.join(', ') ?? '');
        this.description.set(dataset?.description ?? '');
        this.mutations.set([]);
    }

    protected read(event: Event): string {
        const target = event.target;
        return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ? target.value : '';
    }

    protected parsedTags(): string[] {
        return this.tags().split(',').map((tag) => tag.trim()).filter(Boolean);
    }

    protected dataColumns(): string[] {
        return (this.page?.columns ?? []).filter((column) => column !== 'row_id');
    }

    protected value(row: DatasetRow, column: string): string {
        return String(row[column] ?? '');
    }

    protected change(rowId: number, column: string, event: Event): void {
        const value = this.read(event);
        const row = this.rows().find((item) => item.row_id === rowId);
        if (!row) {
            return;
        }

        this.rows.update((rows) => rows.map((item) => item.row_id === rowId ? { ...item, [column]: value } : item));
        this.mutations.update((items) => [
            ...items.filter((item) => !(item.operation === 'update' && item.row_id === rowId)),
            { operation: 'update', row_id: rowId, values: { ...row, [column]: value } },
        ]);
    }

    protected remove(rowId: number): void {
        this.rows.update((rows) => rows.filter((row) => row.row_id !== rowId));
        this.mutations.update((items) => [...items, { operation: 'delete', row_id: rowId }]);
    }

    protected addRow(): void {
        const values = Object.fromEntries(this.dataColumns().map((column) => [column, '']));
        this.rows.update((rows) => [...rows, { row_id: -Date.now(), ...values }]);
        this.mutations.update((items) => [...items, { operation: 'insert', values }]);
    }

    protected rename(name: string): void {
        const newName = window.prompt('New dataset name', name);
        if (newName && newName !== name) {
            this.renamed.emit({ name, newName });
        }
    }
}
