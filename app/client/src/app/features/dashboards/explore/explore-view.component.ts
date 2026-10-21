// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import type { EChartsCoreOption } from 'echarts/core';
import type {
    AnalyticsFilter,
    DatasetSource,
    DistributionResponse,
    PressureBasis,
    SelectionResponse,
    SeriesResponse,
    UptakeAtPressureResponse,
} from '../../../models/analytics.model';
import {
    queryAnalyticsDistribution,
    queryAnalyticsSelection,
    queryAnalyticsSeries,
    queryUptakeAtPressure,
} from '../../../services/analytics.service';
import { DataPaginationComponent } from '../../../shared/components/data-pagination/data-pagination.component';
import type { ChartPointEvent } from '../components/chart.component';
import { ChartComponent } from '../components/chart.component';
import {
    binLabel,
    buildHistogramOption,
    buildIsothermOption,
    buildUptakeOption,
    formatValue,
    groupUptakeRows,
    pressureLabel,
    seriesLabel,
} from '../dashboard-charts';
import { ExploreInspectorComponent } from './explore-inspector.component';
import type { ExploreMode, ExploreState } from './explore-state';
import {
    chartFilter,
    DEFAULT_EXPLORE_STATE,
    PRESSURE_UNITS,
    stateFromQueryParams,
    stateToQueryParams,
    UPTAKE_UNITS,
} from './explore-state';

type IdFacet = 'adsorbent_ids' | 'adsorbate_ids' | 'dataset_ids';

const MAX_SELECTED = 200;
const SEARCH_DELAY_MS = 300;

@Component({
    selector: 'adsmod-dashboard-explore',
    standalone: true,
    imports: [FormsModule, ChartComponent, DataPaginationComponent, ExploreInspectorComponent],
    styleUrl: '../dashboards.css',
    template: `
        <div class="explore-layout" [class.with-inspector]="inspectedId() !== null">
            <aside class="console-card explore-sidebar" aria-label="Query builder">
                <div class="sidebar-head">
                    <h3>Query</h3>
                    <button type="button" class="ghost-button" (click)="resetFilters()">Reset</button>
                </div>

                <fieldset class="query-group">
                    <legend>Source</legend>
                    @for (source of sourceOptions; track source.key) {
                        <label class="check-row">
                            <input type="checkbox" [checked]="state().filter.dataset_sources.includes(source.key)" (change)="toggleSource(source.key)" />
                            <span class="facet-name">{{ source.label }}</span>
                            <span class="facet-count">{{ facets()?.sources?.[source.key] ?? 0 }}</span>
                        </label>
                    }
                </fieldset>

                <fieldset class="query-group">
                    <legend>Pressure basis</legend>
                    <select class="select-input" aria-label="Pressure basis" [ngModel]="state().filter.pressure_basis" (ngModelChange)="patchFilter({ pressure_basis: $event })">
                        @for (basis of basisOptions; track basis) {
                            <option [ngValue]="basis">{{ basis }} ({{ facets()?.pressure_bases?.[basis] ?? 0 }})</option>
                        }
                    </select>
                    <p class="query-hint">Series of different bases never share a pressure axis.</p>
                </fieldset>

                <fieldset class="query-group">
                    <legend>Temperature (K)</legend>
                    <div class="range-row">
                        <input type="number" min="0" step="any" aria-label="Minimum temperature in kelvin" placeholder="min" [value]="state().filter.temperature_min_k ?? ''" (change)="setTemperature('temperature_min_k', $event)" />
                        <span aria-hidden="true">–</span>
                        <input type="number" min="0" step="any" aria-label="Maximum temperature in kelvin" placeholder="max" [value]="state().filter.temperature_max_k ?? ''" (change)="setTemperature('temperature_max_k', $event)" />
                    </div>
                    @if (facets()?.temperature_min_k !== null && facets()?.temperature_min_k !== undefined) {
                        <p class="query-hint">Matching: {{ format(facets()!.temperature_min_k) }}–{{ format(facets()!.temperature_max_k) }} K</p>
                    }
                </fieldset>

                <fieldset class="query-group">
                    <legend>Materials</legend>
                    <input type="search" class="facet-search" placeholder="Search materials" aria-label="Search materials" [ngModel]="materialSearch()" (ngModelChange)="search('material', $event)" />
                    <ul class="facet-list">
                        @for (facet of facets()?.materials ?? []; track facet.id) {
                            <li>
                                <label class="check-row">
                                    <input type="checkbox" [checked]="isFacetSelected('adsorbent_ids', facet.id)" (change)="toggleFacet('adsorbent_ids', facet.id)" />
                                    <span class="facet-name" [title]="facet.name">{{ facet.name }}</span>
                                    <span class="facet-count">{{ facet.count }}</span>
                                </label>
                            </li>
                        } @empty {
                            <li class="facet-empty">No materials match.</li>
                        }
                    </ul>
                </fieldset>

                <fieldset class="query-group">
                    <legend>Adsorbates</legend>
                    <input type="search" class="facet-search" placeholder="Search adsorbates" aria-label="Search adsorbates" [ngModel]="adsorbateSearch()" (ngModelChange)="search('adsorbate', $event)" />
                    <ul class="facet-list">
                        @for (facet of facets()?.adsorbates ?? []; track facet.id) {
                            <li>
                                <label class="check-row">
                                    <input type="checkbox" [checked]="isFacetSelected('adsorbate_ids', facet.id)" (change)="toggleFacet('adsorbate_ids', facet.id)" />
                                    <span class="facet-name" [title]="facet.name">{{ facet.name }}</span>
                                    <span class="facet-count">{{ facet.count }}</span>
                                </label>
                            </li>
                        } @empty {
                            <li class="facet-empty">No adsorbates match.</li>
                        }
                    </ul>
                </fieldset>

                <fieldset class="query-group">
                    <legend>Datasets</legend>
                    <ul class="facet-list short">
                        @for (facet of facets()?.datasets ?? []; track facet.id) {
                            <li>
                                <label class="check-row">
                                    <input type="checkbox" [checked]="isFacetSelected('dataset_ids', facet.id)" (change)="toggleFacet('dataset_ids', facet.id)" />
                                    <span class="facet-name" [title]="facet.name">{{ facet.name }}</span>
                                    <span class="facet-count">{{ facet.count }}</span>
                                </label>
                            </li>
                        } @empty {
                            <li class="facet-empty">No datasets match.</li>
                        }
                    </ul>
                </fieldset>

                <label class="check-row">
                    <input type="checkbox" [checked]="state().filter.include_mixtures" (change)="patchFilter({ include_mixtures: !state().filter.include_mixtures })" />
                    <span class="facet-name">Include mixture isotherms</span>
                </label>
            </aside>

            <div class="explore-main">
                <section class="console-card explore-results" aria-labelledby="explore-results-title">
                    <div class="section-heading">
                        <div>
                            <h3 id="explore-results-title">Matching isotherms <span class="heading-count">{{ selection()?.pagination?.total ?? 0 }}</span></h3>
                            <p>
                                @if (state().selectedIds.length) {
                                    {{ state().selectedIds.length }} selected · charts use the selection.
                                } @else {
                                    No selection · charts use every matching isotherm (up to 200 series).
                                }
                            </p>
                        </div>
                        <div class="heading-actions">
                            <button type="button" class="button secondary" [disabled]="!selection()?.items?.length" (click)="selectPage()">Select page</button>
                            <button type="button" class="button secondary" [disabled]="!state().selectedIds.length" (click)="patch({ selectedIds: [] })">Clear selection</button>
                        </div>
                    </div>
                    @if (selectionError()) {
                        <div class="dashboard-alert" role="alert"><strong>Query failed.</strong> {{ selectionError() }} <button type="button" class="button secondary" (click)="loadSelection()">Retry</button></div>
                    }
                    <div class="data-table-frame explore-table-frame" [attr.aria-busy]="selectionLoading()">
                        <table class="data-table explore-table">
                            <caption class="sr-only">Isotherms matching the query</caption>
                            <thead>
                                <tr>
                                    <th scope="col" class="col-check"><span class="sr-only">Selected</span></th>
                                    <th scope="col">Material</th>
                                    <th scope="col">Adsorbate</th>
                                    <th scope="col" class="num">Temperature</th>
                                    <th scope="col">Basis</th>
                                    <th scope="col">Dataset</th>
                                    <th scope="col" class="num">Points</th>
                                    <th scope="col" class="col-actions"><span class="sr-only">Actions</span></th>
                                </tr>
                            </thead>
                            <tbody>
                                @for (item of selection()?.items ?? []; track item.id) {
                                    <tr [class.selected]="state().selectedIds.includes(item.id)">
                                        <td class="col-check"><input type="checkbox" [checked]="state().selectedIds.includes(item.id)" [attr.aria-label]="'Select ' + item.name" (change)="toggleSelected(item.id)" /></td>
                                        <td class="truncate-cell" [title]="item.material">{{ item.material }}</td>
                                        <td class="truncate-cell">{{ names(item.adsorbates) }}</td>
                                        <td class="num">{{ format(item.temperature_k) }} K</td>
                                        <td>{{ item.pressure_basis }}</td>
                                        <td class="truncate-cell" [title]="item.dataset"><span class="source-key">{{ item.dataset_source }}</span> {{ item.dataset }}</td>
                                        <td class="num">{{ item.point_count }}</td>
                                        <td class="col-actions"><div class="table-actions"><button class="table-action" type="button" (click)="inspect(item.id)">Inspect</button></div></td>
                                    </tr>
                                } @empty {
                                    <tr><td colspan="8" class="data-table-empty">{{ selectionLoading() ? 'Loading…' : 'No isotherms match the query. Widen the filters or import data.' }}</td></tr>
                                }
                            </tbody>
                        </table>
                    </div>
                    <adsmod-data-pagination
                        [page]="page()"
                        [pageSize]="pageSize()"
                        [total]="selection()?.pagination?.total ?? 0"
                        itemLabel="isotherm"
                        [disabled]="selectionLoading()"
                        (pageChange)="page.set($event); loadSelection()"
                        (pageSizeChange)="pageSize.set($event); page.set(1); loadSelection()"
                    />
                </section>

                <section class="console-card explore-chart" aria-labelledby="explore-chart-title">
                    <h3 id="explore-chart-title" class="sr-only">Analysis</h3>
                    <div class="mode-tabs" role="tablist" aria-label="Analysis type">
                        @for (mode of modes; track mode.key) {
                            <button type="button" role="tab" [attr.aria-selected]="state().mode === mode.key" [class.active]="state().mode === mode.key" (click)="patch({ mode: mode.key }, true)">{{ mode.label }}</button>
                        }
                    </div>

                    <div class="chart-controls">
                        @switch (state().mode) {
                            @case ('isotherms') {
                                <label>Colour by
                                    <select class="select-input" [ngModel]="state().colorBy" (ngModelChange)="patch({ colorBy: $event })">
                                        <option value="temperature">Temperature</option>
                                        <option value="material">Material</option>
                                        <option value="adsorbate">Adsorbate</option>
                                        <option value="dataset">Dataset</option>
                                    </select>
                                </label>
                                <label class="check-row inline"><input type="checkbox" [checked]="state().logPressure" (change)="patch({ logPressure: !state().logPressure })" /> Log pressure axis</label>
                            }
                            @case ('uptake') {
                                <label>Pressure
                                    <input type="number" min="0" step="any" [value]="state().targetPressure" (change)="setPositive('targetPressure', $event)" />
                                </label>
                                <label>Method
                                    <select class="select-input" [ngModel]="state().method" (ngModelChange)="patch({ method: $event }, true)">
                                        <option value="linear">Interpolate (log pressure)</option>
                                        <option value="nearest">Nearest measurement</option>
                                    </select>
                                </label>
                                @if (state().method === 'nearest') {
                                    <label>Tolerance (%)
                                        <input type="number" min="0.1" max="100" step="any" [value]="state().tolerance * 100" (change)="setTolerance($event)" />
                                    </label>
                                }
                                <label>Group by
                                    <select class="select-input" [ngModel]="state().groupBy" (ngModelChange)="patch({ groupBy: $event })">
                                        <option value="material">Material</option>
                                        <option value="adsorbate">Adsorbate</option>
                                    </select>
                                </label>
                            }
                            @case ('distribution') {
                                <label>Variable
                                    <select class="select-input" [ngModel]="state().variable" (ngModelChange)="patch({ variable: $event }, true)">
                                        <option value="temperature_k">Temperature (per isotherm)</option>
                                        <option value="pressure">Pressure (per measurement)</option>
                                        <option value="uptake_mol_kg">Uptake (per measurement)</option>
                                        <option value="points_per_isotherm">Points per isotherm</option>
                                    </select>
                                </label>
                                <label>Bins
                                    <input type="number" min="1" max="100" step="1" [value]="state().bins" (change)="setBins($event)" />
                                </label>
                                <label>Scale
                                    <select class="select-input" [ngModel]="state().scale" (ngModelChange)="patch({ scale: $event }, true)">
                                        <option value="linear">Linear</option>
                                        <option value="log">Logarithmic</option>
                                    </select>
                                </label>
                            }
                        }
                        @if (state().mode !== 'distribution' || state().variable === 'pressure') {
                            <label>Pressure unit
                                <select class="select-input" [disabled]="isRelative()" [ngModel]="isRelative() ? 'p/p₀' : state().pressureUnit" (ngModelChange)="patch({ pressureUnit: $event }, true)">
                                    @if (isRelative()) { <option value="p/p₀">p/p₀</option> }
                                    @for (unit of pressureUnits; track unit) { <option [value]="unit">{{ unit }}</option> }
                                </select>
                            </label>
                        }
                        @if (state().mode !== 'distribution') {
                            <label>Uptake unit
                                <select class="select-input" [ngModel]="state().uptakeUnit" (ngModelChange)="patch({ uptakeUnit: $event }, true)">
                                    @for (unit of uptakeUnits; track unit) { <option [value]="unit">{{ unit }}</option> }
                                </select>
                            </label>
                        }
                    </div>

                    <div class="chart-status" aria-live="polite">
                        <span>{{ chartSummary() }}</span>
                        @for (warning of chartWarnings(); track warning) {
                            <span class="chart-warning">{{ warning }}</span>
                        }
                    </div>

                    @if (chartError()) {
                        <div class="dashboard-alert" role="alert"><strong>Analysis failed.</strong> {{ chartError() }} <button type="button" class="button secondary" (click)="loadChart()">Retry</button></div>
                    }

                    <div class="chart-frame" [attr.aria-busy]="chartLoading()">
                        @if (chartOption(); as option) {
                            <adsmod-chart [option]="option" [ariaLabel]="chartLabel()" (pointClick)="chartClicked($event)" />
                        } @else if (!chartLoading() && !chartError()) {
                            <div class="chart-empty">
                                <strong>Nothing to plot yet</strong>
                                <span>{{ emptyMessage() }}</span>
                            </div>
                        }
                        @if (chartLoading()) {
                            <div class="chart-loading"><span class="dashboard-spinner" aria-hidden="true"></span> Querying…</div>
                        }
                    </div>

                    @if (state().mode === 'uptake' && uptake(); as result) {
                        <p class="chart-legend-note">Filled markers are measurements; hollow markers are values interpolated between the two listed observations. Nothing is extrapolated.</p>
                        <div class="data-table-frame">
                            <table class="data-table">
                                <caption class="sr-only">Uptake at {{ format(result.pressure) }} {{ unitName(result.pressure_unit) }}</caption>
                                <thead><tr><th scope="col">Series</th><th scope="col">Dataset</th><th scope="col" class="num">Uptake</th><th scope="col">Method</th><th scope="col">From observations</th></tr></thead>
                                <tbody>
                                    @for (row of result.rows; track row.component_id) {
                                        <tr>
                                            <td class="truncate-cell" [title]="label(row)"><button type="button" class="link-button" (click)="inspect(row.isotherm_id)">{{ label(row) }}</button></td>
                                            <td class="truncate-cell">{{ row.dataset }}</td>
                                            <td class="num">{{ row.value === null ? '—' : format(row.value) + ' ' + row.uptake_unit }}</td>
                                            <td>{{ row.method ?? statusLabel(row.status) }}</td>
                                            <td class="mono-cell">@for (point of row.sources; track point.observation_id) {({{ format(point.pressure) }}, {{ format(point.uptake) }}) }</td>
                                        </tr>
                                    }
                                </tbody>
                            </table>
                        </div>
                    } @else if (chartOption()) {
                        <button type="button" class="ghost-button" [attr.aria-expanded]="showTable()" aria-controls="explore-chart-table" (click)="showTable.set(!showTable())">{{ showTable() ? 'Hide' : 'Show' }} data table</button>
                        @if (showTable()) {
                            <div class="data-table-frame" id="explore-chart-table">
                                @if (state().mode === 'isotherms' && series(); as result) {
                                    <table class="data-table">
                                        <caption class="sr-only">Plotted isotherm series</caption>
                                        <thead><tr><th scope="col">Series</th><th scope="col">Dataset</th><th scope="col" class="num">Points</th><th scope="col" class="num">Pressure range ({{ unitName(result.pressure_unit) }})</th><th scope="col" class="num">Uptake range</th></tr></thead>
                                        <tbody>
                                            @for (item of result.series; track item.component_id) {
                                                <tr>
                                                    <td class="truncate-cell" [title]="label(item)"><button type="button" class="link-button" (click)="inspect(item.isotherm_id)">{{ label(item) }}</button></td>
                                                    <td class="truncate-cell">{{ item.dataset }}</td>
                                                    <td class="num">{{ item.points.length }}</td>
                                                    <td class="num">{{ range(item.points, 'pressure') }}</td>
                                                    <td class="num">{{ range(item.points, 'uptake') }} {{ item.uptake_unit }}</td>
                                                </tr>
                                            }
                                        </tbody>
                                    </table>
                                }
                                @if (state().mode === 'distribution' && distribution(); as result) {
                                    <table class="data-table">
                                        <caption class="sr-only">Histogram bins</caption>
                                        <thead><tr><th scope="col">Bin ({{ result.unit }})</th><th scope="col" class="num">Count</th></tr></thead>
                                        <tbody>
                                            @for (bin of result.bins; track bin.lower) { <tr><td>{{ binRange(bin) }}</td><td class="num">{{ bin.count }}</td></tr> }
                                        </tbody>
                                    </table>
                                }
                            </div>
                        }
                    }
                </section>
            </div>

            @if (inspectedId(); as isothermId) {
                <adsmod-explore-inspector [isothermId]="isothermId" (closed)="inspect(null)" />
            }
        </div>
    `,
})
export class ExploreViewComponent {
    private readonly router = inject(Router);
    private readonly route = inject(ActivatedRoute);

    protected readonly sourceOptions: { key: DatasetSource; label: string }[] = [
        { key: 'nist', label: 'NIST' },
        { key: 'uploaded', label: 'Uploaded' },
    ];
    protected readonly basisOptions: PressureBasis[] = ['absolute', 'partial', 'relative'];
    protected readonly modes: { key: ExploreMode; label: string }[] = [
        { key: 'isotherms', label: 'Isotherms' },
        { key: 'uptake', label: 'Uptake at pressure' },
        { key: 'distribution', label: 'Distribution' },
    ];
    protected readonly pressureUnits = PRESSURE_UNITS;
    protected readonly uptakeUnits = UPTAKE_UNITS;

    protected readonly state = signal<ExploreState>(stateFromQueryParams(this.route.snapshot.queryParams));
    protected readonly page = signal(1);
    protected readonly pageSize = signal(25);
    protected readonly materialSearch = signal('');
    protected readonly adsorbateSearch = signal('');
    protected readonly selection = signal<SelectionResponse | null>(null);
    protected readonly selectionLoading = signal(false);
    protected readonly selectionError = signal<string | null>(null);
    protected readonly series = signal<SeriesResponse | null>(null);
    protected readonly uptake = signal<UptakeAtPressureResponse | null>(null);
    protected readonly distribution = signal<DistributionResponse | null>(null);
    protected readonly chartLoading = signal(false);
    protected readonly chartError = signal<string | null>(null);
    protected readonly showTable = signal(false);
    protected readonly inspectedId = signal<number | null>(null);

    protected readonly facets = computed(() => this.selection()?.facets ?? null);
    protected readonly isRelative = computed(() => this.state().filter.pressure_basis === 'relative');

    protected readonly chartOption = computed<EChartsCoreOption | null>(() => {
        const state = this.state();
        if (state.mode === 'isotherms') {
            const result = this.series();
            return result?.series.length
                ? buildIsothermOption(result.series, { colorBy: state.colorBy, logPressure: state.logPressure, pressureUnit: result.pressure_unit })
                : null;
        }
        if (state.mode === 'uptake') {
            const result = this.uptake();
            return result?.rows.some((row) => row.status === 'ok') ? buildUptakeOption(result.rows, state.groupBy, state.uptakeUnit) : null;
        }
        const result = this.distribution();
        return result?.bins.length ? buildHistogramOption(result.bins, this.distributionAxis(result)) : null;
    });

    protected readonly chartSummary = computed(() => {
        const mode = this.state().mode;
        if (mode === 'isotherms' && this.series()) {
            const result = this.series()!;
            const points = result.series.reduce((sum, item) => sum + item.points.length, 0);
            return `${result.series.length} of ${result.total_series} series · ${points} measured points${result.truncated ? ' · truncated to the series and point limits; narrow the query or select isotherms' : ''}`;
        }
        if (mode === 'uptake' && this.uptake()) {
            const result = this.uptake()!;
            const ok = result.rows.filter((row) => row.status === 'ok').length;
            return `${ok} of ${result.rows.length} series have a value at ${formatValue(result.pressure)} ${pressureLabel(result.pressure_unit)}${result.truncated ? ' · truncated' : ''}`;
        }
        if (mode === 'distribution' && this.distribution()) {
            const { stats, excluded } = this.distribution()!;
            const parts = [`n = ${stats.count}`, `min ${formatValue(stats.min)}`, `median ${formatValue(stats.median)}`, `mean ${formatValue(stats.mean)}`, `max ${formatValue(stats.max)}`];
            if (excluded) parts.push(`${excluded} non-positive values excluded from the log scale`);
            return parts.join(' · ');
        }
        return '';
    });

    protected readonly chartWarnings = computed(() => {
        const mode = this.state().mode;
        if (mode === 'isotherms') return this.series()?.warnings ?? [];
        if (mode === 'uptake') return this.uptake()?.warnings ?? [];
        return [];
    });

    protected readonly chartLabel = computed(() => {
        const mode = this.state().mode;
        if (mode === 'isotherms') return `Adsorption isotherms, ${this.series()?.series.length ?? 0} series, uptake against pressure`;
        if (mode === 'uptake') return `Uptake at ${formatValue(this.state().targetPressure)} against temperature`;
        return `Histogram of ${this.state().variable}`;
    });

    private selectionToken = 0;
    private chartToken = 0;
    private searchTimer: ReturnType<typeof setTimeout> | null = null;

    constructor() {
        void this.loadSelection();
        void this.loadChart();
    }

    protected format(value: number | null | undefined): string {
        return formatValue(value);
    }

    protected unitName(unit: string): string {
        return pressureLabel(unit);
    }

    protected label(item: { material: string; adsorbate: string; temperature_k: number }): string {
        return seriesLabel(item);
    }

    protected binRange(bin: { lower: number; upper: number; count: number }): string {
        return binLabel(bin);
    }

    protected names(items: { name: string }[]): string {
        return items.map((item) => item.name).join(', ');
    }

    protected range(points: { pressure: number; uptake: number }[], key: 'pressure' | 'uptake'): string {
        if (!points.length) return '—';
        const values = points.map((point) => point[key]);
        return `${formatValue(Math.min(...values))}–${formatValue(Math.max(...values))}`;
    }

    protected statusLabel(status: string): string {
        return status === 'out_of_range' ? 'outside measured range' : 'no point within tolerance';
    }

    protected emptyMessage(): string {
        if (this.state().mode === 'uptake') return 'No matching isotherm has measurements bracketing the chosen pressure.';
        return 'No measurements match the query. Pick a material or adsorbate, or widen the temperature range.';
    }

    protected isFacetSelected(key: IdFacet, id: number): boolean {
        return this.state().filter[key].includes(id);
    }

    protected patch(changes: Partial<ExploreState>, reloadChart = false): void {
        this.state.update((state) => ({ ...state, ...changes }));
        this.syncUrl();
        if (reloadChart || 'selectedIds' in changes) void this.loadChart();
    }

    protected patchFilter(changes: Partial<AnalyticsFilter>): void {
        this.state.update((state) => ({ ...state, filter: { ...state.filter, ...changes } }));
        this.page.set(1);
        this.syncUrl();
        void this.loadSelection();
        void this.loadChart();
    }

    protected resetFilters(): void {
        this.materialSearch.set('');
        this.adsorbateSearch.set('');
        this.state.update((state) => ({ ...state, filter: DEFAULT_EXPLORE_STATE.filter, selectedIds: [] }));
        this.patchFilter({});
    }

    protected toggleSource(source: DatasetSource): void {
        const current = this.state().filter.dataset_sources;
        this.patchFilter({ dataset_sources: current.includes(source) ? current.filter((item) => item !== source) : [...current, source] });
    }

    protected toggleFacet(key: IdFacet, id: number): void {
        const current = this.state().filter[key];
        this.patchFilter({ [key]: current.includes(id) ? current.filter((item) => item !== id) : [...current, id] });
    }

    protected setTemperature(key: 'temperature_min_k' | 'temperature_max_k', event: Event): void {
        const raw = (event.target as HTMLInputElement).value;
        const value = raw === '' ? null : Number(raw);
        this.patchFilter({ [key]: value !== null && Number.isFinite(value) && value > 0 ? value : null });
    }

    protected setPositive(key: 'targetPressure', event: Event): void {
        const value = Number((event.target as HTMLInputElement).value);
        if (Number.isFinite(value) && value > 0) this.patch({ [key]: value }, true);
    }

    protected setTolerance(event: Event): void {
        const value = Number((event.target as HTMLInputElement).value) / 100;
        if (Number.isFinite(value) && value > 0 && value <= 1) this.patch({ tolerance: value }, true);
    }

    protected setBins(event: Event): void {
        const value = Math.round(Number((event.target as HTMLInputElement).value));
        if (value >= 1 && value <= 100) this.patch({ bins: value }, true);
    }

    protected search(kind: 'material' | 'adsorbate', value: string): void {
        (kind === 'material' ? this.materialSearch : this.adsorbateSearch).set(value);
        if (this.searchTimer) clearTimeout(this.searchTimer);
        this.searchTimer = setTimeout(() => void this.loadSelection(), SEARCH_DELAY_MS);
    }

    protected toggleSelected(id: number): void {
        const current = this.state().selectedIds;
        if (current.includes(id)) {
            this.patch({ selectedIds: current.filter((item) => item !== id) });
        } else if (current.length < MAX_SELECTED) {
            this.patch({ selectedIds: [...current, id] });
        }
    }

    protected selectPage(): void {
        const ids = (this.selection()?.items ?? []).map((item) => item.id);
        this.patch({ selectedIds: [...new Set([...this.state().selectedIds, ...ids])].slice(0, MAX_SELECTED) });
    }

    protected inspect(id: number | null): void {
        this.inspectedId.set(id);
    }

    protected chartClicked(event: ChartPointEvent): void {
        const state = this.state();
        if (state.mode === 'isotherms') {
            const item = this.series()?.series[event.seriesIndex];
            if (item) this.inspect(item.isotherm_id);
        } else if (state.mode === 'uptake' && this.uptake()) {
            const row = groupUptakeRows(this.uptake()!.rows, state.groupBy).grouped[event.seriesIndex]?.[event.dataIndex];
            if (row) this.inspect(row.isotherm_id);
        }
    }

    async loadSelection(): Promise<void> {
        const token = ++this.selectionToken;
        this.selectionLoading.set(true);
        const result = await queryAnalyticsSelection({
            filter: this.state().filter,
            page: this.page(),
            page_size: this.pageSize(),
            material_search: this.materialSearch() || null,
            adsorbate_search: this.adsorbateSearch() || null,
        });
        if (token !== this.selectionToken) return;
        this.selectionLoading.set(false);
        this.selectionError.set(result.error);
        if (result.data) this.selection.set(result.data);
    }

    async loadChart(): Promise<void> {
        const token = ++this.chartToken;
        const state = this.state();
        const filter = chartFilter(state);
        this.chartLoading.set(true);
        this.chartError.set(null);
        let error: string | null = null;
        if (state.mode === 'isotherms') {
            const result = await queryAnalyticsSeries({ filter, pressure_unit: state.pressureUnit, uptake_unit: state.uptakeUnit });
            if (token !== this.chartToken) return;
            this.series.set(result.data);
            error = result.error;
        } else if (state.mode === 'uptake') {
            const result = await queryUptakeAtPressure({
                filter,
                pressure_unit: state.pressureUnit,
                uptake_unit: state.uptakeUnit,
                pressure: state.targetPressure,
                method: state.method,
                tolerance: state.tolerance,
            });
            if (token !== this.chartToken) return;
            this.uptake.set(result.data);
            error = result.error;
        } else {
            const result = await queryAnalyticsDistribution({
                filter,
                variable: state.variable,
                bins: state.bins,
                scale: state.scale,
                pressure_unit: state.pressureUnit,
            });
            if (token !== this.chartToken) return;
            this.distribution.set(result.data);
            error = result.error;
        }
        this.chartLoading.set(false);
        this.chartError.set(error);
    }

    private distributionAxis(result: DistributionResponse): string {
        const names: Record<string, string> = {
            temperature_k: 'Temperature',
            pressure: 'Pressure',
            uptake_mol_kg: 'Uptake',
            points_per_isotherm: 'Points per isotherm',
        };
        const unit = result.variable === 'pressure' ? pressureLabel(result.unit) : result.unit;
        return `${names[result.variable]} (${unit})${result.scale === 'log' ? ', log bins' : ''}`;
    }

    private syncUrl(): void {
        void this.router.navigate([], { relativeTo: this.route, queryParams: stateToQueryParams(this.state()), replaceUrl: true });
    }
}

