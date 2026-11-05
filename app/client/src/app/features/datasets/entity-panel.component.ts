// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, effect, input, output, signal, untracked } from '@angular/core';
import type { MaterialRecord, SpeciesRecord } from '../../models/materials.model';
import { fetchMaterials, fetchSpecies } from '../../services/materials.service';

export type EntityKind = 'material' | 'species';

export interface EntityRow {
    kind: EntityKind;
    id: number;
    name: string;
    formula: string | null;
    molarMass: number | null;
    experimentCount: number;
    note: string | null;
}

const PAGE_SIZE = 25;
const SEARCH_DELAY_MS = 250;

function materialRow(item: MaterialRecord): EntityRow {
    return {
        kind: 'material',
        id: item.id,
        name: item.name,
        formula: item.formula,
        molarMass: item.molar_mass_g_mol,
        experimentCount: item.isotherm_count,
        note: item.structure_count ? `${item.structure_count} structure${item.structure_count === 1 ? '' : 's'}` : null,
    };
}

function speciesRow(item: SpeciesRecord): EntityRow {
    return {
        kind: 'species',
        id: item.id,
        name: item.name,
        formula: item.formula,
        molarMass: item.molecular_weight,
        experimentCount: item.isotherm_count,
        note: item.pubchem_cid ? `PubChem ${item.pubchem_cid}` : null,
    };
}

@Component({
    selector: 'adsmod-entity-panel',
    standalone: true,
    template: `
        <section class="entity-panel" aria-labelledby="entity-panel-title">
            <h2 id="entity-panel-title" class="sr-only">Materials and species</h2>
            <div class="entity-tabs" role="tablist" aria-label="Registered entities">
                <button
                    id="entity-tab-material"
                    role="tab"
                    type="button"
                    [attr.aria-selected]="kind() === 'material'"
                    aria-controls="entity-tabpanel"
                    [class.active]="kind() === 'material'"
                    (click)="setKind('material')"
                >
                    Materials @if (totals().material !== null) {<span class="count">{{ totals().material }}</span>}
                </button>
                <button
                    id="entity-tab-species"
                    role="tab"
                    type="button"
                    [attr.aria-selected]="kind() === 'species'"
                    aria-controls="entity-tabpanel"
                    [class.active]="kind() === 'species'"
                    (click)="setKind('species')"
                >
                    Species @if (totals().species !== null) {<span class="count">{{ totals().species }}</span>}
                </button>
            </div>
            <div id="entity-tabpanel" role="tabpanel" [attr.aria-labelledby]="'entity-tab-' + kind()" class="entity-body">
                <input
                    type="search"
                    [attr.aria-label]="kind() === 'material' ? 'Search materials' : 'Search species'"
                    [placeholder]="kind() === 'material' ? 'Search materials' : 'Search species or synonyms'"
                    [value]="query()"
                    (input)="search($any($event.target).value)"
                />
                @if (error()) {
                    <p class="entity-message error" role="alert">{{ error() }}</p>
                } @else if (loading() && !rows().length) {
                    <p class="entity-message" role="status">Loading…</p>
                } @else if (!rows().length) {
                    <p class="entity-message">{{ query() ? 'No matches.' : emptyLabel() }}</p>
                } @else {
                    <ul class="entity-rows" [attr.aria-busy]="loading()">
                        @for (row of rows(); track row.id) {
                            <li>
                                <button
                                    type="button"
                                    class="entity-row"
                                    [class.selected]="row.id === selectedId()"
                                    [attr.aria-label]="'Show details for ' + row.name"
                                    (click)="selected.emit(row)"
                                >
                                    <span class="entity-name">{{ row.name }}</span>
                                    <span class="entity-uses" [class.unused]="!row.experimentCount">
                                        {{ row.experimentCount ? row.experimentCount + ' exp.' : 'Unused' }}
                                    </span>
                                    <span class="entity-meta">
                                        {{ row.formula || 'No formula' }}@if (row.molarMass) { · {{ row.molarMass.toFixed(2) }} g/mol}@if (row.note) { · {{ row.note }}}
                                    </span>
                                </button>
                            </li>
                        }
                    </ul>
                    @if (total() > pageSize) {
                        <div class="entity-pager">
                            <button class="button quiet" type="button" [disabled]="page() <= 1 || loading()" (click)="goTo(page() - 1)">Previous</button>
                            <span>{{ page() }} / {{ pageCount() }}</span>
                            <button class="button quiet" type="button" [disabled]="page() >= pageCount() || loading()" (click)="goTo(page() + 1)">Next</button>
                        </div>
                    }
                }
            </div>
        </section>
    `,
    styles: [
        `
            :host { display: block; min-width: 0; }
            .entity-panel { display: grid; gap: var(--spacing-xs); }
            .entity-tabs { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid var(--border-color); }
            .entity-tabs button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.4rem;
                min-height: 2.5rem;
                padding: 0 0.5rem;
                border: 0;
                border-bottom: 2px solid transparent;
                border-radius: 0;
                background: transparent;
                color: var(--text-secondary);
                font: inherit;
                font-size: 0.82rem;
                font-weight: 700;
                cursor: pointer;
            }
            .entity-tabs button.active { border-bottom-color: var(--primary-600); color: var(--text-primary); }
            .entity-tabs .count { padding: 0 6px; border-radius: 999px; background: var(--slate-100); font-size: 0.68rem; font-variant-numeric: tabular-nums; }
            .entity-body { display: grid; gap: var(--spacing-xs); padding-top: var(--spacing-xs); }
            .entity-body > input { width: 100%; }
            .entity-rows { margin: 0; padding: 0; list-style: none; }
            .entity-row {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 1px var(--spacing-xs);
                width: 100%;
                padding: 0.45rem 0.5rem;
                border: 0;
                border-bottom: 1px solid var(--border-color);
                border-radius: 0;
                background: transparent;
                color: inherit;
                font: inherit;
                text-align: left;
                cursor: pointer;
            }
            .entity-row:hover { background: var(--slate-50); }
            .entity-row.selected { background: var(--slate-50); box-shadow: inset 3px 0 0 var(--primary-600); }
            .entity-row:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }
            .entity-name { overflow: hidden; font-size: 0.82rem; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
            .entity-uses { color: var(--text-secondary); font-size: 0.72rem; font-variant-numeric: tabular-nums; }
            .entity-uses.unused { color: var(--text-tertiary); font-style: italic; }
            .entity-meta { grid-column: span 2; overflow: hidden; color: var(--text-tertiary); font-size: 0.72rem; text-overflow: ellipsis; white-space: nowrap; }
            .entity-pager { display: flex; align-items: center; justify-content: space-between; color: var(--text-secondary); font-size: 0.74rem; font-variant-numeric: tabular-nums; }
            .entity-message { margin: 0; padding: var(--spacing-sm) 0; color: var(--text-secondary); font-size: 0.82rem; }
            .entity-message.error { color: var(--danger-text); }
        `,
    ],
})
export class EntityPanelComponent {
    /** Changing this value reloads the current page, e.g. after a dataset import or deletion. */
    readonly refreshKey = input(0);
    readonly selectedId = input<number | null>(null);
    readonly selected = output<EntityRow>();
    readonly kindChanged = output<EntityKind>();

    protected readonly pageSize = PAGE_SIZE;
    protected readonly kind = signal<EntityKind>('material');
    protected readonly query = signal('');
    protected readonly page = signal(1);
    protected readonly rows = signal<EntityRow[]>([]);
    protected readonly total = signal(0);
    protected readonly loading = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly totals = signal<Record<EntityKind, number | null>>({ material: null, species: null });
    protected readonly emptyLabel = computed(() =>
        this.kind() === 'material'
            ? 'No materials yet. They are registered when datasets are imported.'
            : 'No species yet. They are registered when datasets are imported.',
    );

    private searchTimer: ReturnType<typeof setTimeout> | null = null;
    private requestId = 0;

    constructor() {
        effect(() => {
            this.refreshKey();
            untracked(() => void this.load());
        });
    }

    /** Reload after an entity was deleted from the drawer. */
    reload(): void {
        void this.load();
    }

    protected pageCount(): number {
        return Math.max(1, Math.ceil(this.total() / PAGE_SIZE));
    }

    protected setKind(kind: EntityKind): void {
        if (kind === this.kind()) return;
        this.kind.set(kind);
        this.query.set('');
        this.page.set(1);
        this.kindChanged.emit(kind);
        void this.load();
    }

    protected search(value: string): void {
        this.query.set(value);
        this.page.set(1);
        if (this.searchTimer) clearTimeout(this.searchTimer);
        this.searchTimer = setTimeout(() => void this.load(), SEARCH_DELAY_MS);
    }

    protected goTo(page: number): void {
        this.page.set(page);
        void this.load();
    }

    private async load(): Promise<void> {
        const requestId = ++this.requestId;
        const kind = this.kind();
        const params = { page: this.page(), page_size: PAGE_SIZE, q: this.query().trim() };
        this.loading.set(true);
        const result =
            kind === 'material'
                ? await fetchMaterials(params).then((response) => ({
                      error: response.error,
                      rows: response.data?.items.map(materialRow) ?? [],
                      total: response.data?.pagination.total ?? 0,
                  }))
                : await fetchSpecies(params).then((response) => ({
                      error: response.error,
                      rows: response.data?.items.map(speciesRow) ?? [],
                      total: response.data?.pagination.total ?? 0,
                  }));
        if (requestId !== this.requestId) return;
        this.loading.set(false);
        this.error.set(result.error);
        this.rows.set(result.rows);
        this.total.set(result.total);
        if (!params.q) {
            this.totals.update((totals) => ({ ...totals, [kind]: result.total }));
        }
    }
}
