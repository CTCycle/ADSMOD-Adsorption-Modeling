// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import {
    Component,
    effect,
    input,
    output,
    signal,
    untracked,
} from '@angular/core';
import type { ImportIssue } from '../../../models/dataset.model';
import { DataPaginationComponent } from '../data-pagination/data-pagination.component';

export type ExplorerStatus = 'valid' | 'warning' | 'error';

export interface ExplorerRow {
    point: number;
    /** Set only for multi-species experiments. */
    species?: string;
    pressure: string;
    pressureSource: string | null;
    uptake: string;
    uptakeSource: string | null;
    status: ExplorerStatus;
    note: string | null;
}

export interface ExplorerExperiment {
    key: string;
    name: string;
    subtitle: string;
    temperature: string | null;
    pointCount: number;
    pressureRange: string | null;
    status: ExplorerStatus;
    errors: number;
    warnings: number;
    issues: ImportIssue[];
    rows: ExplorerRow[];
    /** Total rows available; greater than rows.length when the list is partial. */
    totalRows: number;
    paged: boolean;
    page: number;
    pageSize: number;
    loading: boolean;
    error: string | null;
}

export interface ExplorerPageChange {
    key: string;
    page: number;
    pageSize: number;
}

@Component({
    selector: 'adsmod-experiment-explorer',
    standalone: true,
    imports: [DataPaginationComponent],
    template: `
        @if (experiments().length > 1) {
            <div class="explorer-toolbar">
                <span>{{ experiments().length }} experiments</span>
                <div>
                    <button
                        type="button"
                        class="button quiet"
                        (click)="expandAll()"
                    >
                        Expand all
                    </button>
                    <button
                        type="button"
                        class="button quiet"
                        (click)="collapseAll()"
                    >
                        Collapse all
                    </button>
                </div>
            </div>
        }
        <div class="explorer-list">
            @for (experiment of experiments(); track experiment.key) {
                <section class="explorer-item" [class]="'explorer-item ' + experiment.status">
                    <h3 class="explorer-heading">
                        <button
                            type="button"
                            class="explorer-header"
                            [attr.aria-expanded]="isOpen(experiment.key)"
                            [attr.aria-controls]="panelId(experiment.key)"
                            (click)="toggle(experiment.key)"
                        >
                            <span class="chevron" aria-hidden="true">{{
                                isOpen(experiment.key) ? '▾' : '▸'
                            }}</span>
                            <span class="explorer-title">
                                <strong>{{ experiment.name }}</strong>
                                <small>{{ experiment.subtitle }}</small>
                            </span>
                            <span class="explorer-facts">
                                @if (experiment.temperature) {
                                    <span>{{ experiment.temperature }}</span>
                                }
                                <span
                                    >{{ experiment.pointCount }}
                                    {{
                                        experiment.pointCount === 1
                                            ? 'point'
                                            : 'points'
                                    }}</span
                                >
                                @if (experiment.pressureRange) {
                                    <span>{{ experiment.pressureRange }}</span>
                                }
                            </span>
                            <span class="explorer-badges">
                                @if (experiment.errors) {
                                    <span class="badge error"
                                        >{{ experiment.errors }}
                                        {{
                                            experiment.errors === 1
                                                ? 'error'
                                                : 'errors'
                                        }}</span
                                    >
                                }
                                @if (experiment.warnings) {
                                    <span class="badge warning"
                                        >{{ experiment.warnings }}
                                        {{
                                            experiment.warnings === 1
                                                ? 'warning'
                                                : 'warnings'
                                        }}</span
                                    >
                                }
                                @if (!experiment.errors && !experiment.warnings) {
                                    <span class="badge valid">Valid</span>
                                }
                            </span>
                        </button>
                    </h3>
                    @if (isOpen(experiment.key)) {
                        <div class="explorer-panel" [id]="panelId(experiment.key)">
                            @if (experiment.issues.length) {
                                <ul class="explorer-issues">
                                    @for (
                                        issue of experiment.issues;
                                        track $index
                                    ) {
                                        <li [class]="issue.severity">
                                            <strong>{{
                                                issue.severity === 'confirmation'
                                                    ? 'needs review'
                                                    : issue.severity
                                            }}</strong>
                                            @if (issue.source_row) {
                                                <span
                                                    >row {{ issue.source_row }}:</span
                                                >
                                            }
                                            {{ issue.message }}
                                            @if (issue.remediation) {
                                                <small>{{
                                                    issue.remediation
                                                }}</small>
                                            }
                                        </li>
                                    }
                                </ul>
                            }
                            @if (experiment.loading) {
                                <p class="explorer-note" role="status">
                                    Loading observations…
                                </p>
                            } @else if (experiment.error) {
                                <p class="explorer-note error" role="alert">
                                    {{ experiment.error }}
                                </p>
                            } @else if (experiment.rows.length) {
                                <div class="explorer-table-wrap">
                                    <table class="explorer-table">
                                        <caption class="sr-only">
                                            Observations of
                                            {{ experiment.name }}, ordered by
                                            pressure
                                        </caption>
                                        <thead>
                                            <tr>
                                                <th scope="col">Point</th>
                                                @if (experiment.rows[0].species) {
                                                    <th scope="col">Species</th>
                                                }
                                                <th scope="col">Pressure</th>
                                                <th scope="col">Uptake</th>
                                                <th scope="col">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            @for (
                                                row of experiment.rows;
                                                track $index
                                            ) {
                                                <tr [class]="row.status">
                                                    <td>{{ row.point }}</td>
                                                    @if (row.species) {
                                                        <td>{{ row.species }}</td>
                                                    }
                                                    <td>
                                                        {{ row.pressure }}
                                                        @if (row.pressureSource) {
                                                            <small
                                                                >source:
                                                                {{
                                                                    row.pressureSource
                                                                }}</small
                                                            >
                                                        }
                                                    </td>
                                                    <td>
                                                        {{ row.uptake }}
                                                        @if (row.uptakeSource) {
                                                            <small
                                                                >source:
                                                                {{
                                                                    row.uptakeSource
                                                                }}</small
                                                            >
                                                        }
                                                    </td>
                                                    <td>
                                                        <span
                                                            class="badge"
                                                            [class]="'badge ' + row.status"
                                                            [attr.title]="row.note"
                                                            >{{
                                                                row.status ===
                                                                'valid'
                                                                    ? 'Valid'
                                                                    : row.status ===
                                                                        'warning'
                                                                      ? 'Warning'
                                                                      : 'Error'
                                                            }}</span
                                                        >
                                                        @if (row.note) {
                                                            <small>{{
                                                                row.note
                                                            }}</small>
                                                        }
                                                    </td>
                                                </tr>
                                            }
                                        </tbody>
                                    </table>
                                </div>
                                @if (experiment.paged) {
                                    <adsmod-data-pagination
                                        itemLabel="observation"
                                        [page]="experiment.page"
                                        [pageSize]="experiment.pageSize"
                                        [total]="experiment.totalRows"
                                        (pageChange)="
                                            changePage(
                                                experiment,
                                                $event,
                                                experiment.pageSize
                                            )
                                        "
                                        (pageSizeChange)="
                                            changePage(experiment, 1, $event)
                                        "
                                    />
                                } @else if (
                                    experiment.totalRows > experiment.rows.length
                                ) {
                                    <p class="explorer-note">
                                        Showing the first
                                        {{ experiment.rows.length }} of
                                        {{ experiment.totalRows }} observations.
                                        The complete series is available on the
                                        dataset page after saving.
                                    </p>
                                }
                            } @else {
                                <p class="explorer-note">
                                    No valid observations could be read for
                                    this experiment.
                                </p>
                            }
                        </div>
                    }
                </section>
            } @empty {
                <p class="explorer-note">No experiments to show.</p>
            }
        </div>
    `,
    styles: [
        `
            :host {
                display: block;
            }
            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                overflow: hidden;
                clip-path: inset(50%);
            }
            .explorer-toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
                color: var(--slate-500);
                font-size: 0.8rem;
            }
            .explorer-list {
                display: grid;
                gap: 0.5rem;
            }
            .explorer-item {
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                background: var(--surface);
            }
            .explorer-item.error {
                border-color: #dc2626;
            }
            .explorer-item.warning {
                border-color: #ca8a04;
            }
            .explorer-heading {
                margin: 0;
                font-size: inherit;
            }
            .explorer-header {
                display: grid;
                grid-template-columns: auto minmax(0, 1.4fr) minmax(0, 1.2fr) auto;
                align-items: center;
                gap: 0.75rem;
                width: 100%;
                padding: 0.7rem 0.85rem;
                border: 0;
                border-radius: 10px;
                color: inherit;
                background: transparent;
                cursor: pointer;
                font: inherit;
                text-align: left;
            }
            .explorer-header:hover {
                background: var(--slate-50);
            }
            .explorer-header:focus-visible {
                outline: 2px solid var(--focus-ring);
                outline-offset: -2px;
            }
            .explorer-title {
                display: grid;
                min-width: 0;
            }
            .explorer-title strong,
            .explorer-title small {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .explorer-title small,
            .explorer-facts {
                color: var(--slate-500);
                font-size: 0.78rem;
            }
            .explorer-facts {
                display: flex;
                flex-wrap: wrap;
                gap: 0.2rem 0.75rem;
            }
            .explorer-badges {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.3rem;
            }
            .badge {
                display: inline-block;
                padding: 0.1rem 0.5rem;
                border-radius: 999px;
                font-size: 0.7rem;
                font-weight: 700;
            }
            .badge.valid {
                color: #0b6b3a;
                background: color-mix(in srgb, #16a34a 16%, transparent);
            }
            .badge.warning {
                color: #92400e;
                background: color-mix(in srgb, #f59e0b 22%, transparent);
            }
            .badge.error {
                color: #b91c1c;
                background: color-mix(in srgb, #dc2626 16%, transparent);
            }
            .explorer-panel {
                padding: 0 0.85rem 0.85rem;
            }
            .explorer-issues {
                display: grid;
                gap: 0.35rem;
                margin: 0 0 0.75rem;
                padding: 0;
                list-style: none;
                font-size: 0.8rem;
            }
            .explorer-issues li {
                padding: 0.5rem 0.65rem;
                border-left: 4px solid #ca8a04;
                background: var(--warning-bg);
            }
            .explorer-issues li.error {
                border-color: #dc2626;
                background: var(--danger-bg);
            }
            .explorer-issues strong {
                margin-right: 0.35rem;
                text-transform: uppercase;
                font-size: 0.7rem;
            }
            .explorer-issues small {
                display: block;
                color: var(--slate-500);
            }
            .explorer-table-wrap {
                border: 1px solid var(--slate-200);
                border-radius: 8px;
            }
            .explorer-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.82rem;
                font-variant-numeric: tabular-nums;
            }
            .explorer-table th,
            .explorer-table td {
                padding: 0.45rem 0.75rem;
                border-bottom: 1px solid var(--slate-200);
                text-align: left;
                vertical-align: top;
            }
            .explorer-table tbody tr:last-child td {
                border-bottom: 0;
            }
            .explorer-table thead th {
                background: var(--slate-50);
                color: var(--slate-600, var(--slate-500));
                font-size: 0.74rem;
            }
            .explorer-table td small {
                display: block;
                color: var(--slate-500);
                font-size: 0.72rem;
            }
            .explorer-table tr.warning td {
                background: var(--warning-bg);
            }
            .explorer-table tr.error td {
                background: var(--danger-bg);
            }
            .explorer-note {
                margin: 0.5rem 0 0;
                color: var(--slate-500);
                font-size: 0.8rem;
            }
            .explorer-note.error {
                color: #b91c1c;
            }
            @media (max-width: 760px) {
                .explorer-header {
                    grid-template-columns: auto minmax(0, 1fr);
                }
                .explorer-facts,
                .explorer-badges {
                    grid-column: 2;
                    justify-content: flex-start;
                }
            }
        `,
    ],
})
export class ExperimentExplorerComponent {
    readonly experiments = input.required<readonly ExplorerExperiment[]>();
    /** Keys expanded when the explorer first receives them. */
    readonly initiallyOpen = input<'none' | 'issues' | 'first'>('issues');
    readonly toggled = output<{ key: string; open: boolean }>();
    readonly pageChanged = output<ExplorerPageChange>();

    private readonly openKeys = signal<ReadonlySet<string>>(new Set());
    private readonly seen = new Set<string>();
    private readonly uid = Math.random().toString(36).slice(2, 8);

    constructor() {
        effect(() => {
            const items = this.experiments();
            untracked(() => this.seedOpenState(items));
        });
    }

    private seedOpenState(items: readonly ExplorerExperiment[]): void {
        const mode = this.initiallyOpen();
        const keys = new Set(this.openKeys());
        items.forEach((experiment, index) => {
            if (this.seen.has(experiment.key)) {
                return;
            }
            this.seen.add(experiment.key);
            const open =
                mode === 'issues'
                    ? experiment.status !== 'valid' || items.length === 1
                    : mode === 'first' && index === 0;
            if (open) {
                keys.add(experiment.key);
                this.toggled.emit({ key: experiment.key, open: true });
            }
        });
        this.openKeys.set(keys);
    }

    protected panelId(key: string): string {
        return `explorer-${this.uid}-${key.replace(/[^A-Za-z0-9_-]/g, '_')}`;
    }

    protected isOpen(key: string): boolean {
        return this.openKeys().has(key);
    }

    protected toggle(key: string): void {
        const keys = new Set(this.openKeys());
        const open = !keys.has(key);
        if (open) {
            keys.add(key);
        } else {
            keys.delete(key);
        }
        this.openKeys.set(keys);
        this.toggled.emit({ key, open });
    }

    protected expandAll(): void {
        const keys = new Set(this.experiments().map((item) => item.key));
        for (const key of keys) {
            if (!this.isOpen(key)) {
                this.toggled.emit({ key, open: true });
            }
        }
        this.openKeys.set(keys);
    }

    protected collapseAll(): void {
        this.openKeys.set(new Set());
    }

    protected changePage(
        experiment: ExplorerExperiment,
        page: number,
        pageSize: number,
    ): void {
        this.pageChanged.emit({ key: experiment.key, page, pageSize });
    }
}
