// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import {
    Component,
    computed,
    effect,
    inject,
    input,
    signal,
    untracked,
    viewChild,
} from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import type { EChartsCoreOption } from 'echarts/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type {
    ColumnRole,
    DatasetImportInspection,
    DatasetSummary,
    ExperimentSummary,
    ImportIssue,
    ImportMapping,
    ObservationPage,
} from '../../models/dataset.model';
import {
    fetchDatasetImport,
    fetchObservations,
} from '../../services/dataset.service';
import { ChartComponent } from '../dashboards/components/chart.component';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import {
    ExperimentExplorerComponent,
    type ExplorerExperiment,
    type ExplorerPageChange,
    type ExplorerRow,
} from '../../shared/components/experiment-explorer/experiment-explorer.component';
import { DatasetActionDialogsComponent } from '../source/dataset-action-dialogs.component';

type DetailsTab = 'overview' | 'experiments' | 'preview' | 'visualization';

interface ObservationState {
    page: number;
    pageSize: number;
    data: ObservationPage | null;
    loading: boolean;
    error: string | null;
}

interface ColumnMappingRow {
    name: string;
    role: ColumnRole;
    unit: string;
    grouped: boolean;
}

const TABS: readonly { id: DetailsTab; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'preview', label: 'Data preview' },
    { id: 'visualization', label: 'Visualization' },
];
const PLOT_POINT_LIMIT = 500;
const NUMBER_FORMAT = new Intl.NumberFormat('en-US', {
    maximumSignificantDigits: 6,
});

function formatNumber(value: number): string {
    return NUMBER_FORMAT.format(value);
}

function pressureUnitLabel(unit: string | null): string {
    return unit === '1' ? 'p/p₀' : (unit ?? '');
}

@Component({
    selector: 'adsmod-dataset-details-page',
    standalone: true,
    imports: [
        ChartComponent,
        DataPaginationComponent,
        DatasetActionDialogsComponent,
        DatePipe,
        ExperimentExplorerComponent,
        RouterLink,
    ],
    template: `
        <div class="data-page dataset-details-page">
            <a class="back-link" routerLink="/datasets">← Custom Datasets</a>

            @if (notFound()) {
                <section class="console-card details-empty" role="alert">
                    <h2>Dataset not found</h2>
                    <p>
                        This dataset does not exist or was deleted. Return to
                        the list to choose another one.
                    </p>
                    <a class="button primary" routerLink="/datasets"
                        >Back to Custom Datasets</a
                    >
                </section>
            } @else if (dataset(); as current) {
                <header class="details-header">
                    <div class="details-title">
                        <h2>{{ current.name }}</h2>
                        <p>{{ current.description || 'No description added yet.' }}</p>
                        <p class="details-tags">
                            {{ current.tags.join(', ') || 'No tags' }}
                        </p>
                    </div>
                    <div class="details-actions">
                        <button class="button secondary" type="button" (click)="dialogs().editMetadata(current)">
                            Edit
                        </button>
                        <button class="button secondary" type="button" (click)="dialogs().rename(current)">
                            Rename
                        </button>
                        <button class="button secondary danger" type="button" (click)="dialogs().remove(current)">
                            Delete
                        </button>
                    </div>
                </header>

                <nav class="details-tabs" aria-label="Dataset sections">
                    @for (item of tabs; track item.id) {
                        <a
                            class="details-tab"
                            [class.active]="activeTab() === item.id"
                            [routerLink]="[]"
                            [queryParams]="{ tab: item.id }"
                            [attr.aria-current]="activeTab() === item.id ? 'page' : null"
                            >{{ item.label }}</a
                        >
                    }
                </nav>

                @if (activeTab() === 'overview') {
                    <section class="console-card details-section" aria-label="Overview">
                        <div class="stat-grid">
                            <div>
                                <span>Origin</span>
                                <strong>{{ current.source === 'nist' ? 'NIST' : 'Workspace' }}</strong>
                            </div>
                            <div>
                                <span>Experiments</span>
                                <strong>{{ current.experiment_count }}</strong>
                            </div>
                            <div>
                                <span>Observations</span>
                                <strong>{{ current.observation_count }}</strong>
                            </div>
                            <div>
                                <span>Created</span>
                                <strong>{{ current.created_at | date: 'medium' }}</strong>
                            </div>
                            @if (inspection(); as imported) {
                                <div>
                                    <span>Source file</span>
                                    <strong class="wrap">{{ imported.original_filename }}</strong>
                                </div>
                                <div>
                                    <span>Structure</span>
                                    <strong>{{ imported.source_structure }}</strong>
                                </div>
                                <div>
                                    <span>Validation</span>
                                    <strong>{{ imported.validation.status }}</strong>
                                </div>
                                <div>
                                    <span>Parser</span>
                                    <strong>{{ imported.parser_version }}</strong>
                                </div>
                            }
                        </div>
                        @if (importLoading()) {
                            <p class="muted" role="status">Loading saved import details…</p>
                        }
                        @if (importError()) {
                            <p class="dataset-status error" role="alert">{{ importError() }}</p>
                        }
                        @if (inspection(); as imported) {
                            <h3>Saved column mapping</h3>
                            <div class="table-wrap">
                                <table class="details-table">
                                    <caption class="sr-only">Column mapping for {{ current.name }}</caption>
                                    <thead>
                                        <tr>
                                            <th scope="col">Source column</th>
                                            <th scope="col">Meaning</th>
                                            <th scope="col">Unit</th>
                                            <th scope="col">Grouping</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        @for (column of columnMappings(imported); track column.name) {
                                            <tr>
                                                <th scope="row">{{ column.name }}</th>
                                                <td>{{ roleLabel(column.role) }}</td>
                                                <td>{{ column.unit || '—' }}</td>
                                                <td>{{ column.grouped ? 'Yes' : 'No' }}</td>
                                            </tr>
                                        }
                                    </tbody>
                                </table>
                            </div>
                            @if (imported.warnings.length) {
                                <div class="warnings">
                                    <strong>Import warnings</strong>
                                    @for (warning of imported.warnings; track $index) {
                                        <p>{{ warning.message }}</p>
                                    }
                                </div>
                            }
                        } @else if (current.source === 'nist' && !importLoading()) {
                            <p class="muted">
                                Datasets acquired from NIST have no saved import mapping.
                            </p>
                        }
                    </section>
                }

                @if (activeTab() === 'experiments') {
                    <section class="console-card details-section" aria-label="Experiments">
                        @if (store.experimentsLoading()) {
                            <p class="muted" role="status">Loading experiments…</p>
                        } @else {
                            <adsmod-experiment-explorer
                                initiallyOpen="first"
                                [experiments]="explorerExperiments()"
                                (toggled)="onExperimentToggled($event)"
                                (pageChanged)="onExplorerPage($event)"
                            />
                        }
                    </section>
                }

                @if (activeTab() === 'preview') {
                    <section class="console-card details-section" aria-label="Data preview">
                        @if (store.experiments().length) {
                            <label class="experiment-picker">
                                <span>Experiment</span>
                                <select class="select-input" [value]="focusId()" (change)="focusExperiment($event)">
                                    @for (experiment of store.experiments(); track experiment.id) {
                                        <option [value]="experiment.id" [selected]="experiment.id === focusId()">
                                            {{ experiment.name }}
                                        </option>
                                    }
                                </select>
                            </label>
                            @if (previewState(); as state) {
                                @if (state.error) {
                                    <p class="dataset-status error" role="alert">{{ state.error }}</p>
                                } @else if (state.data) {
                                    <div class="table-wrap">
                                        <table class="details-table preview-table">
                                            <caption class="sr-only">Stored observations, ordered by pressure</caption>
                                            <thead>
                                                <tr>
                                                    <th scope="col">Point</th>
                                                    <th scope="col">Source row</th>
                                                    <th scope="col">Pressure (source)</th>
                                                    <th scope="col">Pressure (canonical)</th>
                                                    <th scope="col">Uptake (source)</th>
                                                    <th scope="col">Uptake (mol/kg)</th>
                                                    <th scope="col">Uncertainty (mol/kg)</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                @for (row of state.data.rows; track row.id) {
                                                    <tr>
                                                        <td data-label="Point">{{ row.sequence_index + 1 }}</td>
                                                        <td data-label="Source row">{{ row.source_row ?? '—' }}</td>
                                                        <td data-label="Pressure (source)">{{ format(row.pressure_original) }} {{ unitLabel(row.pressure_original_unit) }}</td>
                                                        <td data-label="Pressure (canonical)">{{ format(row.pressure_canonical) }} {{ unitLabel(row.pressure_canonical_unit) }}</td>
                                                        <td data-label="Uptake (source)">{{ format(row.uptake_original) }} {{ row.uptake_original_unit }}</td>
                                                        <td data-label="Uptake (mol/kg)">{{ format(row.uptake_mol_kg) }}</td>
                                                        <td data-label="Uncertainty (mol/kg)">{{ row.uptake_stddev_mol_kg === null ? '—' : format(row.uptake_stddev_mol_kg) }}</td>
                                                    </tr>
                                                }
                                            </tbody>
                                        </table>
                                    </div>
                                    <adsmod-data-pagination
                                        itemLabel="observation"
                                        [disabled]="state.loading"
                                        [page]="state.page"
                                        [pageSize]="state.pageSize"
                                        [total]="state.data.total"
                                        (pageChange)="loadObservationPage(focusId()!, $event, state.pageSize)"
                                        (pageSizeChange)="loadObservationPage(focusId()!, 1, $event)"
                                    />
                                } @else {
                                    <p class="muted" role="status">Loading observations…</p>
                                }
                            } @else {
                                <p class="muted" role="status">Loading observations…</p>
                            }
                        } @else if (store.experimentsLoading()) {
                            <p class="muted" role="status">Loading experiments…</p>
                        } @else {
                            <p class="muted">This dataset has no experiments.</p>
                        }
                    </section>
                }

                @if (activeTab() === 'visualization') {
                    <section class="console-card details-section" aria-label="Visualization">
                        @if (store.experiments().length) {
                            <label class="experiment-picker">
                                <span>Experiment</span>
                                <select class="select-input" [value]="focusId()" (change)="focusExperiment($event)">
                                    @for (experiment of store.experiments(); track experiment.id) {
                                        <option [value]="experiment.id" [selected]="experiment.id === focusId()">
                                            {{ experiment.name }}
                                        </option>
                                    }
                                </select>
                            </label>
                            @if (plotError()) {
                                <p class="dataset-status error" role="alert">{{ plotError() }}</p>
                            } @else if (plotOption(); as option) {
                                <div class="plot-frame">
                                    <adsmod-chart [option]="option" ariaLabel="Isotherm: uptake against pressure" />
                                </div>
                                @if (plotTruncated()) {
                                    <p class="muted">
                                        Showing the first {{ plotLimit }} observations of this experiment.
                                    </p>
                                }
                            } @else {
                                <p class="muted" role="status">Loading observations…</p>
                            }
                        } @else {
                            <p class="muted">This dataset has no experiments to plot.</p>
                        }
                    </section>
                }

                <adsmod-dataset-action-dialogs
                    (renamed)="store.renameDataset($event.id, $event.newName)"
                    (metadataSaved)="store.saveMetadata($event.id, $event.metadata)"
                    (deleted)="deleteAndReturn($event)"
                />
            } @else {
                <p class="muted" role="status">Loading dataset…</p>
            }
        </div>
    `,
    styles: [
        `
            :host {
                display: block;
            }
            .back-link {
                justify-self: start;
                font-size: 0.85rem;
                font-weight: 600;
            }
            .details-header {
                display: flex;
                flex-wrap: wrap;
                align-items: flex-start;
                justify-content: space-between;
                gap: 0.75rem 1.5rem;
            }
            .details-title {
                min-width: 0;
            }
            .details-title h2 {
                margin: 0;
                overflow-wrap: anywhere;
            }
            .details-title p {
                margin: 0.25rem 0 0;
                color: var(--slate-500);
                font-size: 0.85rem;
            }
            .details-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            .details-tabs {
                display: flex;
                flex-wrap: wrap;
                gap: 0.25rem;
                border-bottom: 1px solid var(--slate-200);
            }
            .details-tab {
                padding: 0.55rem 0.9rem;
                border-bottom: 2px solid transparent;
                color: var(--slate-500);
                font-size: 0.88rem;
                font-weight: 650;
                text-decoration: none;
            }
            .details-tab:hover {
                color: var(--link);
            }
            .details-tab.active {
                border-bottom-color: var(--link);
                color: var(--link);
            }
            .details-tab:focus-visible {
                outline: 2px solid var(--focus-ring);
                outline-offset: -2px;
            }
            .details-section {
                display: grid;
                gap: 0.9rem;
            }
            .details-section h3 {
                margin: 0.4rem 0 0;
                font-size: 0.95rem;
            }
            .stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
                gap: 0.6rem;
            }
            .stat-grid > div {
                display: grid;
                gap: 0.15rem;
                min-width: 0;
                padding: 0.7rem 0.8rem;
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                background: var(--slate-50);
            }
            .stat-grid span {
                color: var(--slate-500);
                font-size: 0.74rem;
            }
            .stat-grid .wrap {
                overflow-wrap: anywhere;
            }
            .muted {
                margin: 0;
                color: var(--slate-500);
                font-size: 0.85rem;
            }
            .table-wrap {
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                overflow-x: auto;
            }
            .details-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.82rem;
                font-variant-numeric: tabular-nums;
            }
            .details-table th,
            .details-table td {
                padding: 0.5rem 0.75rem;
                border-bottom: 1px solid var(--slate-200);
                text-align: left;
                white-space: nowrap;
            }
            .details-table thead th {
                background: var(--slate-50);
                font-size: 0.74rem;
            }
            .details-table tbody tr:last-child th,
            .details-table tbody tr:last-child td {
                border-bottom: 0;
            }
            .warnings {
                padding: 0.7rem 0.9rem;
                border-left: 4px solid #ca8a04;
                background: var(--warning-bg);
                font-size: 0.82rem;
            }
            .warnings p {
                margin: 0.25rem 0 0;
            }
            .experiment-picker {
                display: grid;
                gap: 0.3rem;
                max-width: 28rem;
                font-size: 0.8rem;
                font-weight: 650;
            }
            @media (max-width: 720px) {
                .preview-table thead {
                    position: absolute;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                    clip-path: inset(50%);
                }
                .preview-table,
                .preview-table tbody,
                .preview-table tr,
                .preview-table td {
                    display: block;
                }
                .preview-table tr {
                    padding: 0.4rem 0;
                    border-bottom: 1px solid var(--slate-200);
                }
                .preview-table td {
                    display: grid;
                    grid-template-columns: 9.5rem minmax(0, 1fr);
                    gap: 0.5rem;
                    border: 0;
                    padding: 0.2rem 0.75rem;
                    white-space: normal;
                }
                .preview-table td::before {
                    content: attr(data-label);
                    color: var(--slate-500);
                    font-size: 0.72rem;
                    font-weight: 700;
                }
            }
            .plot-frame {
                height: min(26rem, 60vh);
                min-height: 18rem;
            }
            .details-empty {
                display: grid;
                gap: 0.5rem;
                justify-items: start;
            }
            .details-empty h2,
            .details-empty p {
                margin: 0;
            }
        `,
    ],
})
export class DatasetDetailsPageComponent {
    /** Route parameter, bound by withComponentInputBinding. */
    readonly id = input.required<string>();
    /** Query parameter selecting the visible section. */
    readonly tab = input<string | undefined>(undefined);

    protected readonly store = inject(CoreWorkspaceStore);
    private readonly router = inject(Router);
    protected readonly dialogs = viewChild.required(DatasetActionDialogsComponent);
    protected readonly tabs = TABS;
    protected readonly plotLimit = PLOT_POINT_LIMIT;

    protected readonly notFound = signal(false);
    protected readonly inspection = signal<DatasetImportInspection | null>(null);
    protected readonly importLoading = signal(false);
    protected readonly importError = signal('');
    protected readonly focusId = signal<number | null>(null);
    protected readonly observationStates = signal<Record<number, ObservationState>>({});
    protected readonly plotRows = signal<ObservationPage | null>(null);
    protected readonly plotError = signal('');

    private loadRevision = 0;
    private plotRevision = 0;

    protected readonly datasetId = computed(() => {
        const parsed = Number(this.id());
        return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
    });
    protected readonly dataset = computed<DatasetSummary | undefined>(() => {
        const id = this.datasetId();
        return this.store.datasets().find((item) => item.id === id);
    });
    protected readonly activeTab = computed<DetailsTab>(() => {
        const requested = this.tab();
        return TABS.find((item) => item.id === requested)?.id ?? 'overview';
    });
    protected readonly previewState = computed(() => {
        const id = this.focusId();
        return id === null ? null : (this.observationStates()[id] ?? null);
    });
    protected readonly plotTruncated = computed(
        () => (this.plotRows()?.total ?? 0) > PLOT_POINT_LIMIT,
    );
    protected readonly plotOption = computed<EChartsCoreOption | null>(() => {
        const page = this.plotRows();
        if (!page) {
            return null;
        }
        const unit = pressureUnitLabel(page.rows[0]?.pressure_canonical_unit ?? null);
        return {
            animation: false,
            textStyle: { fontFamily: 'inherit' },
            grid: { left: 70, right: 24, top: 24, bottom: 52 },
            tooltip: {
                trigger: 'item',
                formatter: (params: { value: [number, number] }) =>
                    `p = ${formatNumber(params.value[0])} ${unit}<br>q = ${formatNumber(params.value[1])} mol/kg`,
            },
            xAxis: {
                type: 'value',
                name: `Pressure (${unit})`,
                nameLocation: 'middle',
                nameGap: 32,
                scale: true,
            },
            yAxis: {
                type: 'value',
                name: 'Uptake (mol/kg)',
                nameLocation: 'middle',
                nameGap: 50,
                scale: true,
            },
            series: [
                {
                    type: 'line',
                    symbol: 'circle',
                    symbolSize: 7,
                    lineStyle: { width: 1.5, opacity: 0.7 },
                    data: page.rows.map((row) => [
                        row.pressure_canonical,
                        row.uptake_mol_kg,
                    ]),
                },
            ],
        };
    });
    protected readonly explorerExperiments = computed<ExplorerExperiment[]>(() => {
        const warnings = this.inspection()?.warnings ?? [];
        return this.store.experiments().map((experiment) =>
            this.toExplorer(
                experiment,
                warnings.filter((issue) => issue.experiment === experiment.external_key),
                this.observationStates()[experiment.id],
            ),
        );
    });

    constructor() {
        effect(() => {
            const id = this.datasetId();
            untracked(() => void this.load(id));
        });
        effect(() => {
            const id = this.focusId();
            const tab = this.activeTab();
            if (id === null) {
                return;
            }
            untracked(() => {
                if (tab === 'preview' && !this.observationStates()[id]) {
                    void this.loadObservationPage(id, 1, 25);
                }
                if (tab === 'visualization') {
                    void this.loadPlot(id);
                }
            });
        });
    }

    private async load(id: number | null): Promise<void> {
        const revision = ++this.loadRevision;
        this.notFound.set(false);
        this.inspection.set(null);
        this.importError.set('');
        this.focusId.set(null);
        this.observationStates.set({});
        this.plotRows.set(null);
        if (id === null) {
            this.notFound.set(true);
            return;
        }
        if (!this.store.datasets().some((item) => item.id === id)) {
            await this.store.refreshDatasets();
        }
        if (revision !== this.loadRevision) {
            return;
        }
        if (!this.store.datasets().some((item) => item.id === id)) {
            this.notFound.set(true);
            return;
        }
        this.importLoading.set(true);
        const importRequest = fetchDatasetImport(id);
        await this.store.selectDataset(id);
        const result = await importRequest;
        if (revision !== this.loadRevision) {
            return;
        }
        this.importLoading.set(false);
        if (result.data) {
            this.inspection.set(result.data);
        } else if (this.dataset()?.source !== 'nist') {
            this.importError.set(
                result.error || 'Failed to load saved import details.',
            );
        }
        this.focusId.set(this.store.experiments()[0]?.id ?? null);
    }

    protected columnMappings(imported: DatasetImportInspection): ColumnMappingRow[] {
        const mapping: ImportMapping = imported.mapping;
        const first = imported.validation.experiments[0];
        const sourceUnits: Partial<Record<ColumnRole, string>> = {
            pressure: first?.observations[0]?.pressure_original_unit,
            uptake: first?.observations[0]?.uptake_original_unit,
            temperature: first?.temperature_original_unit,
        };
        return Object.entries(mapping.column_roles).map(([name, role]) => ({
            name,
            role,
            unit:
                mapping.unit_overrides[role] ||
                (sourceUnits[role] ? pressureUnitLabel(sourceUnits[role] ?? null) : ''),
            grouped: mapping.grouping_columns.includes(name),
        }));
    }

    protected roleLabel(role: ColumnRole): string {
        return role
            .split('_')
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(' ');
    }

    protected format(value: number): string {
        return formatNumber(value);
    }

    protected unitLabel(unit: string): string {
        return pressureUnitLabel(unit);
    }

    protected focusExperiment(event: Event): void {
        const id = Number((event.target as HTMLSelectElement).value);
        this.focusId.set(Number.isFinite(id) ? id : null);
    }

    protected onExperimentToggled(change: { key: string; open: boolean }): void {
        const id = Number(change.key);
        if (change.open && !this.observationStates()[id]) {
            void this.loadObservationPage(id, 1, 25);
        }
    }

    protected onExplorerPage(change: ExplorerPageChange): void {
        void this.loadObservationPage(Number(change.key), change.page, change.pageSize);
    }

    protected async loadObservationPage(
        experimentId: number,
        page: number,
        pageSize: number,
    ): Promise<void> {
        const datasetId = this.datasetId();
        if (datasetId === null) {
            return;
        }
        const previous = this.observationStates()[experimentId];
        this.setObservationState(experimentId, {
            page,
            pageSize,
            data: previous?.data ?? null,
            loading: true,
            error: null,
        });
        const result = await fetchObservations(
            datasetId,
            experimentId,
            (page - 1) * pageSize,
            pageSize,
        );
        if (datasetId !== this.datasetId()) {
            return;
        }
        this.setObservationState(experimentId, {
            page,
            pageSize,
            data: result.data ?? previous?.data ?? null,
            loading: false,
            error: result.data ? null : result.error || 'Failed to load observations.',
        });
    }

    private setObservationState(id: number, state: ObservationState): void {
        this.observationStates.update((current) => ({ ...current, [id]: state }));
    }

    private async loadPlot(experimentId: number): Promise<void> {
        const datasetId = this.datasetId();
        if (datasetId === null) {
            return;
        }
        const revision = ++this.plotRevision;
        this.plotRows.set(null);
        this.plotError.set('');
        const result = await fetchObservations(datasetId, experimentId, 0, PLOT_POINT_LIMIT);
        if (revision !== this.plotRevision) {
            return;
        }
        if (result.data) {
            this.plotRows.set(result.data);
        } else {
            this.plotError.set(result.error || 'Failed to load observations.');
        }
    }

    private toExplorer(
        experiment: ExperimentSummary,
        issues: ImportIssue[],
        state: ObservationState | undefined,
    ): ExplorerExperiment {
        const unit = pressureUnitLabel(experiment.pressure_canonical_unit);
        const rows: ExplorerRow[] = (state?.data?.rows ?? []).map((row) => {
            const pressure = `${formatNumber(row.pressure_canonical)} ${pressureUnitLabel(row.pressure_canonical_unit)}`;
            const pressureSource = `${formatNumber(row.pressure_original)} ${pressureUnitLabel(row.pressure_original_unit)}`;
            const uptake = `${formatNumber(row.uptake_mol_kg)} mol/kg`;
            const uptakeSource = `${formatNumber(row.uptake_original)} ${row.uptake_original_unit}`;
            return {
                point: row.sequence_index + 1,
                pressure,
                pressureSource: pressureSource === pressure ? null : pressureSource,
                uptake,
                uptakeSource: uptakeSource === uptake ? null : uptakeSource,
                status: 'valid',
                note: null,
            };
        });
        const temperature =
            experiment.temperature_original_unit &&
            typeof experiment.temperature_original === 'number' &&
            experiment.temperature_original_unit !== 'K'
                ? `${formatNumber(experiment.temperature_original)} ${experiment.temperature_original_unit} (${formatNumber(experiment.temperature_k)} K)`
                : `${formatNumber(experiment.temperature_k)} K`;
        const warnings = issues.length;
        return {
            key: String(experiment.id),
            name: experiment.name,
            subtitle: `${experiment.adsorbates.join(', ') || 'Unknown adsorbate'} on ${experiment.adsorbent}`,
            temperature,
            pointCount: experiment.observation_count,
            pressureRange:
                typeof experiment.pressure_min === 'number' &&
                typeof experiment.pressure_max === 'number'
                    ? `${formatNumber(experiment.pressure_min)}–${formatNumber(experiment.pressure_max)} ${unit}`
                    : null,
            status: warnings > 0 ? 'warning' : 'valid',
            errors: 0,
            warnings,
            issues,
            rows,
            totalRows: state?.data?.total ?? experiment.observation_count,
            paged: true,
            page: state?.page ?? 1,
            pageSize: state?.pageSize ?? 25,
            loading: !!state?.loading && !state.data,
            error: state?.error ?? null,
        };
    }

    protected async deleteAndReturn(datasetId: number): Promise<void> {
        await this.store.deleteDataset(datasetId);
        await this.router.navigate(['/datasets']);
    }
}
