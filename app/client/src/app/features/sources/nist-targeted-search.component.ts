// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import type { NISTSearchItem } from '../../models/public-data.model';
import { searchNistIsotherms } from '../../services/public-data.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';

type SearchField = 'adsorbate' | 'adsorbent' | 'experiment';

const FIELDS: { value: SearchField; label: string; placeholder: string }[] = [
    { value: 'adsorbate', label: 'Compound', placeholder: 'e.g. carbon dioxide, CO2, or an InChIKey' },
    { value: 'adsorbent', label: 'Adsorbent material', placeholder: 'e.g. ZIF-8 or a NIST material hashkey' },
    { value: 'experiment', label: 'Experiment', placeholder: 'Isotherm filename or DOI' },
];
const MAX_SELECTION = 100;

/** Targeted NIST acquisition: search the published isotherm index, then fetch only the chosen records. */
@Component({
    selector: 'adsmod-nist-targeted-search',
    standalone: true,
    imports: [FormsModule, DataPaginationComponent],
    template: `
        <div class="query-row">
            <label class="query-field query-select">
                <span>Search by</span>
                <select [(ngModel)]="field" aria-label="Search field">
                    @for (option of fields; track option.value) { <option [value]="option.value">{{ option.label }}</option> }
                </select>
            </label>
            <label class="query-field">
                <span>{{ currentField().label }}</span>
                <input [(ngModel)]="text" [placeholder]="currentField().placeholder" (keyup.enter)="search(1)" />
            </label>
            <button class="button primary" type="button" [disabled]="searching() || !text.trim()" (click)="search(1)">{{ searching() ? 'Searching…' : 'Search' }}</button>
        </div>
        <details class="advanced-options">
            <summary>Temperature range (K)</summary>
            <div class="advanced-fields">
                <label class="query-field"><span>Min</span><input type="number" min="0" [(ngModel)]="temperatureMin" aria-label="Minimum temperature in kelvin" /></label>
                <label class="query-field"><span>Max</span><input type="number" min="0" [(ngModel)]="temperatureMax" aria-label="Maximum temperature in kelvin" /></label>
            </div>
        </details>
        @if (message(); as text) { <p class="empty-inline" role="status">{{ text }}</p> }
        @if (items().length) {
            <div class="data-table-frame data-table-frame--compact targeted-frame">
                <table class="data-table targeted-table">
                    <caption class="sr-only">Matching NIST isotherms</caption>
                    <thead><tr>
                        <th class="select-column" scope="col"><input type="checkbox" [checked]="allSelected()" [disabled]="!selectable().length" aria-label="Select all fetchable results" (change)="toggleAll($event)" /></th>
                        <th scope="col">Isotherm</th><th scope="col">Compound</th><th scope="col">Adsorbent</th><th scope="col">T (K)</th><th scope="col">Status</th>
                    </tr></thead>
                    <tbody>
                        @for (item of items(); track item.filename) {
                            <tr>
                                <td class="select-column"><input type="checkbox" [checked]="selected().includes(item.filename)" [disabled]="item.local" [attr.aria-label]="'Select ' + item.filename" (change)="toggle(item.filename, $event)" /></td>
                                <td class="truncate-cell" [title]="item.filename + (item.doi ? ' · ' + item.doi : '')">{{ item.filename }}</td>
                                <td class="truncate-cell" [title]="item.adsorbate_names.join(', ')">{{ item.adsorbate_names.join(', ') || '—' }}</td>
                                <td class="truncate-cell" [title]="item.adsorbent_name || ''">{{ item.adsorbent_name || '—' }}</td>
                                <td>{{ item.temperature_k ?? '—' }}</td>
                                <td>{{ item.local ? 'Stored' : 'Available' }}</td>
                            </tr>
                        }
                    </tbody>
                </table>
            </div>
            <adsmod-data-pagination [page]="page()" [pageSize]="pageSize" [total]="total()" itemLabel="isotherm" [pageSizes]="[pageSize]" [disabled]="searching()" (pageChange)="search($event)" />
            <div class="targeted-actions">
                <span>{{ total().toLocaleString() }} matching isotherms · fetching also stores their compound and material records.</span>
                <button class="button primary" type="button" [disabled]="!selected().length || submitting()" (click)="fetchSelected.emit(selected())">{{ submitting() ? 'Starting…' : 'Fetch selected (' + selected().length + ')' }}</button>
            </div>
        }
    `,
    styles: `
        :host { display: grid; gap: 10px; min-width: 0; }
        .advanced-options summary { cursor: pointer; color: var(--text-secondary, #667085); font-size: 0.74rem; font-weight: 600; }
        .advanced-fields { display: flex; flex-wrap: wrap; gap: 8px 10px; margin-top: 8px; }
        .advanced-fields .query-field { flex: 0 1 8rem; }
        .targeted-frame { --data-table-min-width: 640px; }
        .targeted-table .select-column { width: 40px; }
        .targeted-table th:nth-child(5) { width: 64px; }
        .targeted-table th:nth-child(6) { width: 84px; }
        .targeted-actions { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; justify-content: space-between; color: var(--text-secondary, #667085); font-size: 0.74rem; }
        .empty-inline { margin: 0; color: var(--text-secondary, #667085); font-size: 0.8rem; }
    `,
})
export class NistTargetedSearchComponent {
    readonly submitting = input(false);
    readonly fetchSelected = output<string[]>();
    readonly failed = output<string>();

    protected readonly fields = FIELDS;
    protected readonly pageSize = 10;
    protected field: SearchField = 'adsorbate';
    protected text = '';
    protected temperatureMin: number | null = null;
    protected temperatureMax: number | null = null;
    protected readonly searching = signal(false);
    protected readonly items = signal<NISTSearchItem[]>([]);
    protected readonly total = signal(0);
    protected readonly page = signal(1);
    protected readonly message = signal<string | null>(null);
    protected readonly selected = signal<string[]>([]);
    protected readonly selectable = computed(() => this.items().filter((item) => !item.local));
    protected readonly allSelected = computed(
        () => this.selectable().length > 0 && this.selectable().every((item) => this.selected().includes(item.filename)),
    );

    protected currentField() {
        return FIELDS.find((item) => item.value === this.field) ?? FIELDS[0];
    }

    protected async search(page: number): Promise<void> {
        const query = this.text.trim();
        if (!query) return;
        this.searching.set(true);
        this.message.set(null);
        try {
            const result = await searchNistIsotherms({
                [this.field]: query,
                temperature_min: this.temperatureMin,
                temperature_max: this.temperatureMax,
                limit: this.pageSize,
                offset: (page - 1) * this.pageSize,
            });
            if (!result.data) {
                this.failed.emit(result.error ?? 'NIST search failed.');
                return;
            }
            if (page === 1) this.selected.set([]);
            this.page.set(page);
            this.items.set(result.data.items);
            this.total.set(result.data.total);
            if (!result.data.total) this.message.set('No matching NIST isotherms. Try a synonym or a shorter name.');
        } finally {
            this.searching.set(false);
        }
    }

    protected toggle(filename: string, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        const current = this.selected();
        if (!checked) {
            this.selected.set(current.filter((item) => item !== filename));
        } else if (current.length < MAX_SELECTION) {
            this.selected.set([...current, filename]);
        } else {
            (event.target as HTMLInputElement).checked = false;
            this.failed.emit(`Select at most ${MAX_SELECTION} isotherms per fetch.`);
        }
    }

    protected toggleAll(event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        const pageIds = this.selectable().map((item) => item.filename);
        const others = this.selected().filter((item) => !pageIds.includes(item));
        this.selected.set(checked ? [...others, ...pageIds].slice(0, MAX_SELECTION) : others);
    }

    /** Clears the selection after the parent has queued the fetch. */
    reset(): void {
        this.selected.set([]);
        void this.search(this.page());
    }
}
