// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import type { NISTSearchItem } from '../../models/public-data.model';
import { searchNistIsotherms } from '../../services/public-data.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';

const MAX_SELECTION = 100;
const PAGE_SIZE = 25;

/**
 * Targeted NIST acquisition: search the published isotherm index, then fetch only the chosen records.
 * NIST has no server-side search, so every filled criterion is matched locally against the index and
 * combined (AND). Compound, adsorbent and the isotherm/DOI text each count as an anchor criterion.
 */
@Component({
    selector: 'adsmod-nist-targeted-search',
    standalone: true,
    imports: [FormsModule, DataPaginationComponent],
    template: `
        <div class="query-row">
            <label class="query-field">
                <span>Compound</span>
                <input [(ngModel)]="adsorbate" placeholder="e.g. carbon dioxide, CO2, or an InChIKey" (keyup.enter)="search(1)" />
            </label>
            <label class="query-field">
                <span>Adsorbent material</span>
                <input [(ngModel)]="adsorbent" placeholder="e.g. ZIF-8 or a NIST hashkey" (keyup.enter)="search(1)" />
            </label>
            <button class="button primary" type="button" [disabled]="searching() || !canSearch()" (click)="search(1)">{{ searching() ? 'Searching…' : 'Search' }}</button>
        </div>
        <details class="advanced-options">
            <summary>More filters@if (advancedCount()) { <span class="filter-count">{{ advancedCount() }} active</span> }</summary>
            <div class="advanced-fields">
                <label class="query-field wide"><span>Isotherm or DOI</span><input [(ngModel)]="experiment" placeholder="Filename or DOI fragment" (keyup.enter)="search(1)" /></label>
                <label class="query-field"><span>Min T (K)</span><input type="number" min="0" [(ngModel)]="temperatureMin" (keyup.enter)="search(1)" aria-label="Minimum temperature in kelvin" /></label>
                <label class="query-field"><span>Max T (K)</span><input type="number" min="0" [(ngModel)]="temperatureMax" (keyup.enter)="search(1)" aria-label="Maximum temperature in kelvin" /></label>
                <button class="button quiet" type="button" (click)="clear()">Clear</button>
            </div>
        </details>
        @if (message(); as text) { <p class="empty-inline" role="status">{{ text }}</p> }
        @if (items().length) {
            <div class="data-table-frame targeted-frame">
                <table class="data-table targeted-table">
                    <caption class="sr-only">Matching NIST isotherms</caption>
                    <thead><tr>
                        <th class="select-column" scope="col"><input type="checkbox" [checked]="allSelected()" [disabled]="!selectable().length" title="Select all fetchable results on this page" aria-label="Select all fetchable results on this page" (change)="toggleAll($event)" /></th>
                        <th scope="col">Isotherm</th><th scope="col">Compound</th><th scope="col">Adsorbent</th><th class="num" scope="col">T (K)</th><th scope="col">Status</th>
                    </tr></thead>
                    <tbody>
                        @for (item of items(); track item.filename) {
                            <tr [class.selected]="selected().includes(item.filename)">
                                <td class="select-column"><input type="checkbox" [checked]="selected().includes(item.filename)" [disabled]="item.local" [attr.aria-label]="'Select ' + item.filename" (change)="toggle(item.filename, $event)" /></td>
                                <td class="truncate-cell" [title]="item.filename + (item.doi ? ' · ' + item.doi : '')">{{ item.filename }}</td>
                                <td class="truncate-cell" [title]="item.adsorbate_names.join(', ')">{{ item.adsorbate_names.join(', ') || '—' }}</td>
                                <td class="truncate-cell" [title]="item.adsorbent_name || ''">{{ item.adsorbent_name || '—' }}</td>
                                <td class="num">{{ item.temperature_k ?? '—' }}</td>
                                <td><span class="stored-state" [class.stored]="item.local">{{ item.local ? 'Stored' : 'Available' }}</span></td>
                            </tr>
                        }
                    </tbody>
                </table>
            </div>
            <adsmod-data-pagination [page]="page()" [pageSize]="pageSize" [total]="total()" itemLabel="isotherm" [pageSizes]="[pageSize]" [disabled]="searching()" (pageChange)="search($event)" />
            <div class="targeted-actions">
                <span>{{ total().toLocaleString() }} matching isotherms · fetching also stores their compound and material records.</span>
                <span class="targeted-submit">
                    @if (selected().length) {
                        <span>{{ selected().length }} selected across pages</span>
                        <button class="text-button" type="button" (click)="selected.set([])">Clear</button>
                    }
                    <button class="button primary" type="button" [disabled]="!selected().length || submitting()" (click)="fetchSelected.emit(selected())">{{ submitting() ? 'Starting…' : 'Fetch selected (' + selected().length + ')' }}</button>
                </span>
            </div>
        }
    `,
    styles: `
        :host { display: grid; gap: 10px; min-width: 0; }
        .advanced-options summary { display: inline-flex; gap: 8px; align-items: center; cursor: pointer; color: var(--text-secondary, #667085); font-size: 0.74rem; font-weight: 600; }
        .filter-count { padding: 1px 7px; border-radius: 999px; background: var(--primary-50); color: var(--link); font-size: 0.66rem; }
        .advanced-fields { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: flex-end; margin-top: 8px; }
        .advanced-fields .query-field { flex: 0 1 8rem; }
        .advanced-fields .query-field.wide { flex: 1 1 14rem; }
        .targeted-frame { --data-table-min-width: 680px; max-height: min(54vh, 28rem); }
        .targeted-table .select-column { width: 2.5rem; }
        .targeted-table th:nth-child(3) { width: 17%; }
        .targeted-table th:nth-child(4) { width: 22%; }
        .targeted-table th:nth-child(5) { width: 4.5rem; }
        .targeted-table th:nth-child(6) { width: 6rem; }
        .stored-state { color: var(--text-secondary); }
        .stored-state.stored { color: var(--success-text); font-weight: 600; }
        .targeted-actions { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; justify-content: space-between; color: var(--text-secondary, #667085); font-size: 0.74rem; }
        .targeted-submit { display: inline-flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
        .text-button { padding: 0; border: 0; background: none; color: var(--link); font: inherit; font-weight: 600; cursor: pointer; }
        .empty-inline { margin: 0; color: var(--text-secondary, #667085); font-size: 0.8rem; }
    `,
})
export class NistTargetedSearchComponent {
    readonly submitting = input(false);
    readonly fetchSelected = output<string[]>();
    readonly failed = output<string>();

    protected readonly pageSize = PAGE_SIZE;
    protected adsorbate = '';
    protected adsorbent = '';
    protected experiment = '';
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

    protected canSearch(): boolean {
        return Boolean(this.adsorbate.trim() || this.adsorbent.trim() || this.experiment.trim());
    }

    protected advancedCount(): number {
        return [this.experiment.trim(), this.temperatureMin, this.temperatureMax].filter((value) => value !== '' && value !== null).length;
    }

    protected clear(): void {
        this.experiment = '';
        this.temperatureMin = null;
        this.temperatureMax = null;
    }

    protected async search(page: number): Promise<void> {
        if (!this.canSearch() || this.searching()) return;
        const { temperatureMin: low, temperatureMax: high } = this;
        if (low !== null && high !== null && low > high) {
            this.message.set('The minimum temperature must not exceed the maximum.');
            return;
        }
        this.searching.set(true);
        this.message.set(null);
        try {
            const result = await searchNistIsotherms({
                adsorbate: this.adsorbate.trim(),
                adsorbent: this.adsorbent.trim(),
                experiment: this.experiment.trim(),
                temperature_min: low,
                temperature_max: high,
                limit: PAGE_SIZE,
                offset: (page - 1) * PAGE_SIZE,
            });
            if (!result.data) {
                this.failed.emit(result.error ?? 'NIST search failed.');
                return;
            }
            if (page === 1) this.selected.set([]);
            this.page.set(page);
            this.items.set(result.data.items);
            this.total.set(result.data.total);
            if (!result.data.total) this.message.set('No matching NIST isotherms. Try a synonym, a shorter name, or fewer filters.');
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
