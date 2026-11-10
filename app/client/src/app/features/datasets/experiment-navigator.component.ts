// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, effect, inject, input, output, signal, untracked } from '@angular/core';
import type { ExperimentSummary } from '../../models/dataset.model';
import { fetchExperiments } from '../../services/dataset.service';

export const EXPERIMENT_PAGE_SIZE = 50;
const SEARCH_DELAY_MS = 250;

export function speciesLabel(experiment: ExperimentSummary): string {
    if (experiment.components.length <= 1) return experiment.adsorbates.join(', ') || 'Unknown species';
    return experiment.components
        .map((component) => (component.mole_fraction === null ? component.name : `${component.name} (${component.mole_fraction})`))
        .join(' + ');
}

/** Searchable, paginated list of a dataset's experiments. */
@Component({
    selector: 'adsmod-experiment-navigator',
    standalone: true,
    template: `
        <nav class="experiment-nav" aria-label="Experiments">
            <div class="experiment-nav-head">
                <h3>Experiments <span class="count">{{ total() }}</span></h3>
                <input type="search" aria-label="Search experiments" placeholder="Name, material, species" [value]="query()" (input)="search($any($event.target).value)" />
                <label class="edited-toggle"><input type="checkbox" [checked]="editedOnly()" (change)="toggleEdited($any($event.target).checked)" /> Locally edited only</label>
            </div>
            @if (error()) {
                <p class="nav-message error" role="alert">{{ error() }}</p>
            } @else if (loading() && !items().length) {
                <p class="nav-message" role="status">Loading experiments…</p>
            } @else if (!items().length) {
                <p class="nav-message">{{ query() || editedOnly() ? 'No experiments match.' : 'This dataset has no experiments.' }}</p>
            }
            <ul class="experiment-items" [attr.aria-busy]="loading()" (keydown)="navigate($event)">
                @for (experiment of items(); track experiment.id) {
                    <li>
                        <button
                            type="button"
                            class="experiment-item"
                            [class.selected]="experiment.id === selectedId()"
                            [attr.aria-current]="experiment.id === selectedId() ? 'true' : null"
                            [title]="experiment.name + ' — ' + experiment.adsorbent + ' · ' + speciesLabel(experiment)"
                            (click)="selected.emit(experiment)"
                        >
                            <span class="experiment-name">{{ experiment.name }}</span>
                            <span class="experiment-flags">
                                @if (experiment.components.length > 1) {
                                    <span class="flag mix" title="Multi-species experiment">mix</span>
                                }
                                @if (pendingIds().has(experiment.id)) {
                                    <span class="flag pending" title="Unsaved changes">●</span>
                                }
                                @if (experiment.edited_at) {
                                    <span class="flag edited" title="Edited locally">Edited</span>
                                }
                            </span>
                            <span class="experiment-meta">{{ experiment.adsorbent }} · {{ speciesLabel(experiment) }} · {{ experiment.temperature_k.toFixed(1) }} K · {{ experiment.observation_count }} pts</span>
                        </button>
                    </li>
                }
            </ul>
            @if (total() > pageSize) {
                <div class="experiment-pager">
                    <button class="button quiet" type="button" [disabled]="page() <= 1 || loading()" (click)="goTo(page() - 1)">Previous</button>
                    <span>{{ page() }} / {{ pageCount() }}</span>
                    <button class="button quiet" type="button" [disabled]="page() >= pageCount() || loading()" (click)="goTo(page() + 1)">Next</button>
                </div>
            }
        </nav>
    `,
    styles: [
        `
            :host { display: block; min-width: 0; }
            .experiment-nav { display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto; max-height: calc(100vh - 12rem); min-height: 0; }
            .experiment-nav-head { display: grid; gap: 0.4rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color); }
            .experiment-nav-head h3 { display: flex; align-items: center; gap: 0.4rem; margin: 0; font-size: 0.85rem; }
            .count { padding: 0 6px; border-radius: 999px; background: var(--slate-100); font-size: 0.68rem; font-variant-numeric: tabular-nums; }
            .edited-toggle { display: inline-flex; align-items: center; gap: 0.35rem; color: var(--text-secondary); font-size: 0.72rem; }
            .experiment-items { min-height: 0; margin: 0; padding: 0; overflow-y: auto; list-style: none; }
            .experiment-item {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 0 0.4rem;
                width: 100%;
                padding: 0.3rem 0.5rem;
                border: 0;
                border-bottom: 1px solid var(--border-color);
                border-radius: 0;
                background: transparent;
                color: inherit;
                font: inherit;
                text-align: left;
                cursor: pointer;
            }
            .experiment-item:hover { background: var(--slate-50); }
            .experiment-item.selected { background: var(--slate-50); box-shadow: inset 3px 0 0 var(--primary-600); }
            .experiment-item:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }
            .experiment-name { overflow: hidden; font-size: 0.8rem; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
            .experiment-flags { display: inline-flex; gap: 0.25rem; align-items: center; }
            .flag { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; }
            .flag.pending { color: var(--primary-600); font-size: 0.7rem; }
            .flag.mix { padding: 0 5px; border-radius: 999px; background: var(--slate-100); color: var(--text-secondary); }
            .flag.edited { padding: 0 5px; border-radius: 999px; background: color-mix(in srgb, #d97706 14%, transparent); color: #b45309; }
            .experiment-meta { grid-column: span 2; overflow: hidden; color: var(--text-tertiary); font-size: 0.7rem; text-overflow: ellipsis; white-space: nowrap; }
            .experiment-pager { display: flex; align-items: center; justify-content: space-between; gap: 0.25rem; padding-top: 0.4rem; color: var(--text-secondary); font-size: 0.72rem; font-variant-numeric: tabular-nums; }
            .nav-message { margin: 0; padding: 0.75rem 0; color: var(--text-secondary); font-size: 0.8rem; }
            .nav-message.error { color: var(--danger-text); }
            @media (max-width: 960px) {
                .experiment-nav { max-height: 22rem; }
            }
        `,
    ],
})
export class ExperimentNavigatorComponent {
    readonly datasetId = input.required<number>();
    readonly selectedId = input<number | null>(null);
    /** Bump to reload the current page, e.g. after saving edits. */
    readonly refreshKey = input(0);
    readonly pendingIds = input<ReadonlySet<number>>(new Set());
    readonly selected = output<ExperimentSummary>();
    /** Emitted with the first experiment of each loaded page. */
    readonly loaded = output<ExperimentSummary[]>();

    protected readonly pageSize = EXPERIMENT_PAGE_SIZE;
    protected readonly items = signal<ExperimentSummary[]>([]);
    protected readonly total = signal(0);
    protected readonly page = signal(1);
    protected readonly query = signal('');
    protected readonly editedOnly = signal(false);
    protected readonly loading = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly speciesLabel = speciesLabel;

    private readonly host: ElementRef<HTMLElement> = inject(ElementRef);
    private requestId = 0;
    private searchTimer: ReturnType<typeof setTimeout> | null = null;

    constructor() {
        effect(() => {
            this.datasetId();
            this.refreshKey();
            untracked(() => void this.load());
        });
        effect(() => {
            this.selectedId();
            this.items();
            queueMicrotask(() =>
                this.host.nativeElement.querySelector('.experiment-item.selected')?.scrollIntoView({ block: 'nearest' }),
            );
        });
    }

    protected pageCount(): number {
        return Math.max(1, Math.ceil(this.total() / this.pageSize));
    }

    /** Up/Down/Home/End move the selection through the loaded page. */
    protected navigate(event: KeyboardEvent): void {
        const items = this.items();
        if (!items.length) return;
        const current = items.findIndex((item) => item.id === this.selectedId());
        const target =
            event.key === 'ArrowDown' ? Math.min(items.length - 1, current + 1)
            : event.key === 'ArrowUp' ? Math.max(0, current - 1)
            : event.key === 'Home' ? 0
            : event.key === 'End' ? items.length - 1
            : -1;
        if (target < 0) return;
        event.preventDefault();
        this.selected.emit(items[target]);
        this.host.nativeElement.querySelectorAll<HTMLElement>('.experiment-item')[target]?.focus();
    }

    protected search(value: string): void {
        this.query.set(value);
        this.page.set(1);
        if (this.searchTimer) clearTimeout(this.searchTimer);
        this.searchTimer = setTimeout(() => void this.load(), SEARCH_DELAY_MS);
    }

    protected toggleEdited(value: boolean): void {
        this.editedOnly.set(value);
        this.page.set(1);
        void this.load();
    }

    protected goTo(page: number): void {
        this.page.set(page);
        void this.load();
    }

    private async load(): Promise<void> {
        const requestId = ++this.requestId;
        this.loading.set(true);
        const result = await fetchExperiments(this.datasetId(), (this.page() - 1) * this.pageSize, this.pageSize, {
            q: this.query().trim(),
            edited_only: this.editedOnly(),
        });
        if (requestId !== this.requestId) return;
        this.loading.set(false);
        this.error.set(result.error);
        this.items.set(result.data?.experiments ?? []);
        this.total.set(result.data?.total ?? 0);
        this.loaded.emit(this.items());
    }
}
