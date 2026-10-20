// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, EventEmitter, Input, Output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import type { MaterialRecordView } from '../../models/public-data.model';
import { fetchPublicMaterials } from '../../services/public-data.service';

/** Search stored materials by name and pick one for an explicit link. */
@Component({
    selector: 'adsmod-material-picker',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="material-picker">
            <label>{{ label }}
                <input
                    [(ngModel)]="query"
                    placeholder="Search stored materials"
                    [disabled]="disabled"
                    (keyup.enter)="search()"
                />
            </label>
            <button class="button secondary" type="button" [disabled]="disabled || searching() || !query.trim()" (click)="search()">
                {{ searching() ? 'Searching…' : 'Find' }}
            </button>
            @if (results().length) {
                <label>Material
                    <select [ngModel]="selectedId()" (ngModelChange)="choose($event)" [disabled]="disabled">
                        <option [ngValue]="null">Select a material</option>
                        @for (item of results(); track item.id) {
                            <option [ngValue]="item.id">{{ item.name }}{{ item.formula ? ' · ' + item.formula : '' }}</option>
                        }
                    </select>
                </label>
            } @else if (searched()) {
                <span class="material-picker-empty" role="status">No stored material matches “{{ lastQuery() }}”.</span>
            }
        </div>
    `,
    styles: [`
        .material-picker {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: end;
        }
        .material-picker label {
            display: grid;
            gap: 5px;
            min-width: 180px;
            color: var(--text-secondary, #667085);
            font-size: 0.72rem;
            font-weight: 600;
        }
        .material-picker input,
        .material-picker select {
            min-height: 38px;
            padding: 0 0.6rem;
            border: 1px solid #dbe4f4;
            border-radius: 10px;
            background: #fff;
            color: var(--text-primary, #18202c);
            font: inherit;
            font-size: 0.82rem;
        }
        .material-picker-empty {
            align-self: center;
            color: var(--text-secondary, #667085);
            font-size: 0.74rem;
        }
    `],
})
export class MaterialPickerComponent {
    @Input() label = 'Link to material';
    @Input() disabled = false;
    @Output() readonly selected = new EventEmitter<MaterialRecordView | null>();

    protected query = '';
    protected readonly results = signal<MaterialRecordView[]>([]);
    protected readonly selectedId = signal<number | null>(null);
    protected readonly searching = signal(false);
    protected readonly searched = signal(false);
    protected readonly lastQuery = signal('');

    protected async search(): Promise<void> {
        const query = this.query.trim();
        if (!query) return;
        this.searching.set(true);
        try {
            const result = await fetchPublicMaterials({ page: 1, page_size: 25, q: query });
            this.results.set(result.data?.items ?? []);
            this.lastQuery.set(query);
            this.searched.set(true);
            this.choose(null);
        } finally {
            this.searching.set(false);
        }
    }

    protected choose(id: number | null): void {
        this.selectedId.set(id);
        this.selected.emit(this.results().find((item) => item.id === id) ?? null);
    }
}
