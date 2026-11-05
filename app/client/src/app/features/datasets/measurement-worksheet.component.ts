// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, computed, effect, inject, input, signal, untracked } from '@angular/core';
import type { ExperimentSummary, ObservationRow } from '../../models/dataset.model';
import { fetchObservations } from '../../services/dataset.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import type { DatasetEditSession, ObservationField } from './dataset-edit-session';

export const WORKSHEET_PAGE_SIZE = 100;
const EDITABLE: readonly ObservationField[] = ['pressure_original', 'uptake_original', 'uptake_stddev_mol_kg'];

export function pressureUnitLabel(unit: string | null): string {
    return unit === '1' ? 'p/p₀' : (unit ?? '');
}

/** Full-precision text for a stored number; never rounds measured values. */
export function exact(value: number | null): string {
    return value === null || value === undefined ? '' : String(value);
}

@Component({
    selector: 'adsmod-measurement-worksheet',
    standalone: true,
    imports: [DataPaginationComponent],
    template: `
        <div class="worksheet-toolbar">
            <span class="worksheet-summary" aria-live="polite">{{ total() }} measurements</span>
            <label>Order
                <select class="select-input" [value]="order()" (change)="setOrder($any($event.target).value)">
                    <option value="sequence">As measured</option>
                    <option value="pressure">By pressure</option>
                </select>
            </label>
        </div>
        @if (error()) {
            <p class="dataset-status error" role="alert">{{ error() }}</p>
        }
        <div class="data-table-frame worksheet-frame" [class.editing]="editing()" [attr.aria-busy]="loading()">
            <table class="data-table worksheet" [style.--data-table-min-width]="mixture() ? '760px' : '640px'">
                <caption class="sr-only">Measurements of {{ experiment().name }}{{ editing() ? ' (editing)' : '' }}</caption>
                <colgroup>
                    <col class="col-index" />
                    <col class="col-row" />
                    @if (mixture()) { <col class="col-component" /> }
                    <col /><col /><col class="col-sigma" /><col class="col-derived" /><col class="col-derived" />
                </colgroup>
                <thead>
                    <tr>
                        <th scope="col" class="num">#</th>
                        <th scope="col" class="num" title="Row in the imported file">Src row</th>
                        @if (mixture()) { <th scope="col">Species</th> }
                        <th scope="col" class="num">Pressure <span class="unit">[{{ pressureUnit() }}]</span></th>
                        <th scope="col" class="num">Uptake <span class="unit">[{{ uptakeUnit() }}]</span></th>
                        <th scope="col" class="num">σ uptake <span class="unit">[mol/kg]</span></th>
                        <th scope="col" class="num derived">Pressure <span class="unit">[{{ canonicalPressureUnit() }}]</span></th>
                        <th scope="col" class="num derived">Uptake <span class="unit">[mol/kg]</span></th>
                    </tr>
                </thead>
                <tbody>
                    @for (row of rows(); track row.id; let rowIndex = $index) {
                        <tr>
                            <th scope="row" class="num index">{{ row.sequence_index + 1 }}</th>
                            <td class="num muted">{{ row.source_row ?? '—' }}</td>
                            @if (mixture()) { <td class="truncate-cell">{{ componentName(row) }}</td> }
                            @for (field of editable; track field; let column = $index) {
                                <td class="num" [class.changed]="session().isObservationChanged(row.id, field)" [class.invalid]="!!session().issueFor('observation', row.id, field)">
                                    @if (editing()) {
                                        <input
                                            class="cell-input"
                                            type="text"
                                            inputmode="decimal"
                                            autocomplete="off"
                                            [attr.data-row]="rowIndex"
                                            [attr.data-col]="column"
                                            [attr.aria-label]="columnLabel(field) + ' for measurement ' + (row.sequence_index + 1)"
                                            [attr.aria-invalid]="!!session().issueFor('observation', row.id, field)"
                                            [title]="session().issueFor('observation', row.id, field)?.message ?? ''"
                                            [value]="exact(session().observationValue(row, field))"
                                            (focus)="$any($event.target).select()"
                                            (change)="commit(row, field, $any($event.target).value)"
                                            (keydown)="navigate($event, row, field, rowIndex, column)"
                                        />
                                    } @else {
                                        {{ exact(row[field]) || '—' }}
                                    }
                                </td>
                            }
                            <td class="num derived" [class.stale]="editing() && rowChanged(row)">{{ exact(row.pressure_canonical) }}</td>
                            <td class="num derived" [class.stale]="editing() && rowChanged(row)">{{ exact(row.uptake_mol_kg) }}</td>
                        </tr>
                    } @empty {
                        @if (!loading()) {
                            <tr><td [attr.colspan]="mixture() ? 8 : 7" class="data-table-empty">No measurements in this experiment.</td></tr>
                        }
                    }
                </tbody>
            </table>
        </div>
        @if (editing()) {
            <p class="worksheet-hint">Enter or ↓ moves down, ↑ moves up, Tab moves across, Esc restores the stored value. Derived columns are recomputed when you save.</p>
        }
        @if (total() > pageSize) {
            <adsmod-data-pagination
                [page]="page()"
                [pageSize]="pageSize"
                [pageSizes]="[pageSize]"
                [total]="total()"
                itemLabel="measurement"
                [disabled]="loading()"
                (pageChange)="goTo($event)"
            />
        }
    `,
    styles: [
        `
            :host { display: grid; gap: 0.5rem; min-width: 0; }
            .worksheet-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
            .worksheet-toolbar label { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--text-secondary); font-size: 0.75rem; font-weight: 600; }
            .worksheet-toolbar .select-input { width: auto; }
            .worksheet-summary { color: var(--text-secondary); font-size: 0.78rem; font-variant-numeric: tabular-nums; }
            .worksheet-frame { max-height: min(62vh, 40rem); }
            .worksheet-frame.editing { outline: 2px solid color-mix(in srgb, var(--primary-600) 35%, transparent); outline-offset: -1px; }
            .worksheet .col-index { width: 3.5rem; }
            .worksheet .col-row { width: 4.5rem; }
            .worksheet .col-component { width: 8rem; }
            .worksheet .col-sigma { width: 7.5rem; }
            .worksheet .col-derived { width: 9rem; }
            .worksheet .num { text-align: right; font-variant-numeric: tabular-nums; }
            .worksheet th.index { position: sticky; left: 0; z-index: 1; background: var(--surface-subtle); color: var(--text-secondary); font-weight: 600; }
            .worksheet thead th:first-child { left: 0; z-index: 3; }
            .worksheet .unit { color: var(--text-tertiary); font-weight: 500; text-transform: none; }
            .worksheet .muted { color: var(--text-tertiary); }
            .worksheet .derived { color: var(--text-secondary); background: color-mix(in srgb, var(--slate-100) 45%, transparent); }
            .worksheet td.stale { color: var(--text-tertiary); font-style: italic; }
            .worksheet td.changed { background: color-mix(in srgb, var(--primary-600) 9%, transparent); }
            .worksheet td.invalid { box-shadow: inset 0 0 0 2px var(--danger); background: var(--danger-bg); }
            .worksheet td:has(.cell-input) { padding: 2px 4px; }
            .cell-input {
                width: 100%;
                min-height: 1.9rem !important;
                padding: 0 0.4rem;
                border: 1px solid transparent;
                border-radius: 4px;
                background: transparent;
                color: inherit;
                font: inherit;
                font-variant-numeric: tabular-nums;
                text-align: right;
            }
            .cell-input:hover { border-color: var(--border-color); }
            .cell-input:focus { border-color: var(--focus-ring); outline: none; background: var(--surface); box-shadow: 0 0 0 2px color-mix(in srgb, var(--focus-ring) 25%, transparent); }
            .worksheet-hint { margin: 0; color: var(--text-tertiary); font-size: 0.72rem; }
        `,
    ],
})
export class MeasurementWorksheetComponent {
    readonly datasetId = input.required<number>();
    readonly experiment = input.required<ExperimentSummary>();
    readonly session = input.required<DatasetEditSession>();
    readonly editing = input(false);
    /** Bump to reload the current page after a save. */
    readonly refreshKey = input(0);
    /** Sequence index to bring into view (for "go to" links on validation issues). */
    readonly focusSequence = input<number | null>(null);

    protected readonly editable = EDITABLE;
    protected readonly pageSize = WORKSHEET_PAGE_SIZE;
    protected readonly exact = exact;
    protected readonly rows = signal<ObservationRow[]>([]);
    protected readonly total = signal(0);
    protected readonly page = signal(1);
    protected readonly order = signal<'sequence' | 'pressure'>('sequence');
    protected readonly loading = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly mixture = computed(() => this.experiment().components.length > 1);
    protected readonly pressureUnit = computed(() => {
        const experiment = this.experiment();
        return pressureUnitLabel(this.session().experimentValue<string>(experiment, 'pressure_unit', experiment.pressure_original_unit));
    });
    protected readonly uptakeUnit = computed(() => {
        const experiment = this.experiment();
        return this.session().experimentValue<string>(experiment, 'uptake_unit', experiment.uptake_original_unit) ?? '';
    });
    protected readonly canonicalPressureUnit = computed(() => pressureUnitLabel(this.experiment().pressure_canonical_unit ?? 'Pa'));

    private readonly host: ElementRef<HTMLElement> = inject(ElementRef);
    private requestId = 0;
    private loadedExperiment: number | null = null;

    constructor() {
        effect(() => {
            const experimentId = this.experiment().id;
            this.datasetId();
            untracked(() => {
                if (this.loadedExperiment !== experimentId) {
                    this.loadedExperiment = experimentId;
                    this.page.set(1);
                }
            });
        });
        effect(() => {
            this.experiment();
            this.refreshKey();
            this.page();
            this.order();
            untracked(() => void this.load());
        });
        effect(() => {
            const sequence = this.focusSequence();
            if (sequence === null) return;
            untracked(() => {
                this.order.set('sequence');
                const components = Math.max(1, this.experiment().components.length);
                this.page.set(Math.floor((sequence * components) / this.pageSize) + 1);
            });
        });
    }

    protected componentName(row: ObservationRow): string {
        return this.experiment().components.find((component) => component.id === row.component_id)?.name ?? '';
    }

    protected columnLabel(field: ObservationField): string {
        return field === 'pressure_original' ? 'Pressure' : field === 'uptake_original' ? 'Uptake' : 'Uptake uncertainty';
    }

    protected rowChanged(row: ObservationRow): boolean {
        const session = this.session();
        return EDITABLE.some((field) => session.isObservationChanged(row.id, field)) || this.unitsChanged();
    }

    private unitsChanged(): boolean {
        const session = this.session();
        const id = this.experiment().id;
        return session.isExperimentChanged(id, 'pressure_unit') || session.isExperimentChanged(id, 'uptake_unit') || session.isExperimentChanged(id, 'pressure_basis');
    }

    protected setOrder(order: 'sequence' | 'pressure'): void {
        this.order.set(order);
        this.page.set(1);
    }

    protected goTo(page: number): void {
        this.page.set(page);
    }

    protected commit(row: ObservationRow, field: ObservationField, raw: string): void {
        const text = raw.trim().replace(',', '.');
        const value = text === '' ? null : Number(text);
        this.session().setObservationValue(this.experiment().id, row, field, value);
    }

    protected navigate(event: KeyboardEvent, row: ObservationRow, field: ObservationField, rowIndex: number, column: number): void {
        const target = event.target as HTMLInputElement;
        if (event.key === 'Escape') {
            event.preventDefault();
            target.value = exact(row[field]);
            this.session().setObservationValue(this.experiment().id, row, field, row[field]);
            return;
        }
        const step = event.key === 'Enter' || event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0;
        if (!step) return;
        event.preventDefault();
        this.commit(row, field, target.value);
        this.host.nativeElement
            .querySelector<HTMLInputElement>(`input[data-row="${rowIndex + step}"][data-col="${column}"]`)
            ?.focus();
    }

    private async load(): Promise<void> {
        const requestId = ++this.requestId;
        this.loading.set(true);
        const result = await fetchObservations(
            this.datasetId(),
            this.experiment().id,
            (this.page() - 1) * this.pageSize,
            this.pageSize,
            this.order(),
        );
        if (requestId !== this.requestId) return;
        this.loading.set(false);
        this.error.set(result.data ? null : result.error || 'Failed to load measurements.');
        if (result.data) {
            this.rows.set(result.data.rows);
            this.total.set(result.data.total);
        }
    }
}
