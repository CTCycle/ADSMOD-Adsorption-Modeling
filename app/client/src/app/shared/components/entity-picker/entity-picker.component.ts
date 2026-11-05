// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { fetchMaterials, fetchSpecies } from '../../../services/materials.service';

export type PickerKind = 'material' | 'species';

export interface PickedEntity {
    id: number;
    name: string;
}

interface PickerOption extends PickedEntity {
    formula: string | null;
}

const NOUNS: Record<PickerKind, string> = { material: 'material', species: 'species' };

/** Search stored materials or species by name and pick one existing record. */
@Component({
    selector: 'adsmod-entity-picker',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="entity-picker" [class.compact]="compact()">
            <label>{{ label() }}
                <input
                    type="search"
                    [(ngModel)]="query"
                    [placeholder]="'Search stored ' + (kind() === 'material' ? 'materials' : 'species')"
                    [disabled]="disabled()"
                    (keydown.enter)="$event.preventDefault(); search()"
                />
            </label>
            <button class="button secondary" type="button" [disabled]="disabled() || searching() || !query.trim()" (click)="search()">
                {{ searching() ? 'Searching…' : 'Find' }}
            </button>
            @if (results().length) {
                <label>{{ noun() === 'material' ? 'Material' : 'Species' }}
                    <select [ngModel]="selectedId()" (ngModelChange)="choose($event)" [disabled]="disabled()">
                        <option [ngValue]="null">Select a {{ noun() }}</option>
                        @for (item of results(); track item.id) {
                            <option [ngValue]="item.id">{{ item.name }}{{ item.formula ? ' · ' + item.formula : '' }}</option>
                        }
                    </select>
                </label>
            } @else if (searched()) {
                <span class="entity-picker-empty" role="status">No stored {{ noun() }} matches “{{ lastQuery() }}”.</span>
            }
        </div>
    `,
    styles: [`
        .entity-picker {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: end;
        }
        .entity-picker label {
            display: grid;
            gap: 5px;
            min-width: 180px;
            color: var(--text-secondary, #667085);
            font-size: 0.72rem;
            font-weight: 600;
        }
        .entity-picker.compact label {
            min-width: 9rem;
        }
        .entity-picker input,
        .entity-picker select {
            min-height: 34px;
            padding: 0 0.6rem;
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-sm);
            background: var(--surface);
            color: var(--text-primary, #18202c);
            font: inherit;
            font-size: 0.8rem;
        }
        .entity-picker-empty {
            align-self: center;
            color: var(--text-secondary, #667085);
            font-size: 0.74rem;
        }
    `],
})
export class EntityPickerComponent {
    readonly kind = input<PickerKind>('material');
    readonly label = input('Link to material');
    readonly disabled = input(false);
    readonly compact = input(false);
    readonly selected = output<PickedEntity | null>();

    protected query = '';
    protected readonly results = signal<PickerOption[]>([]);
    protected readonly selectedId = signal<number | null>(null);
    protected readonly searching = signal(false);
    protected readonly searched = signal(false);
    protected readonly lastQuery = signal('');

    protected noun(): string {
        return NOUNS[this.kind()];
    }

    protected async search(): Promise<void> {
        const query = this.query.trim();
        if (!query) return;
        this.searching.set(true);
        try {
            const params = { page: 1, page_size: 25, q: query };
            const items: PickerOption[] =
                this.kind() === 'material'
                    ? ((await fetchMaterials(params)).data?.items ?? []).map((item) => ({ id: item.id, name: item.name, formula: item.formula }))
                    : ((await fetchSpecies(params)).data?.items ?? []).map((item) => ({ id: item.id, name: item.name, formula: item.formula }));
            this.results.set(items);
            this.lastQuery.set(query);
            this.searched.set(true);
            this.choose(null);
        } finally {
            this.searching.set(false);
        }
    }

    protected choose(id: number | null): void {
        this.selectedId.set(id);
        const item = this.results().find((option) => option.id === id);
        this.selected.emit(item ? { id: item.id, name: item.name } : null);
    }
}
