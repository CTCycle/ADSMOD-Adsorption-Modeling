// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import type { AnalyticsOverviewResponse, FacetCount } from '../../../models/analytics.model';
import { fetchAnalyticsOverview } from '../../../services/analytics.service';
import { MetricCardComponent } from '../../../shared/components/metric-card/metric-card.component';
import type { ChartPointEvent } from '../components/chart.component';
import { ChartComponent } from '../components/chart.component';
import { buildHistogramOption, buildRankingOption, CATEGORICAL_PALETTE, formatValue, pressureLabel } from '../dashboard-charts';

@Component({
    selector: 'adsmod-dashboard-overview',
    standalone: true,
    imports: [ChartComponent, MetricCardComponent, RouterLink],
    styleUrl: '../dashboards.css',
    template: `
        @if (error()) {
            <div class="dashboard-alert" role="alert"><strong>Overview failed to load.</strong> {{ error() }} <button type="button" class="button secondary" (click)="load()">Retry</button></div>
        }
        @if (loading() && !overview()) {
            <div class="chart-loading static"><span class="dashboard-spinner" aria-hidden="true"></span> Summarising the local database…</div>
        }
        @if (overview(); as data) {
            @if (data.isotherms === 0) {
                <section class="console-card chart-empty standalone">
                    <strong>No isotherms stored yet</strong>
                    <span>Import a dataset or fetch NIST isotherms; the dashboard reads them directly, without copying data.</span>
                    <div class="heading-actions">
                        <a class="button secondary" routerLink="/datasets">Open Custom Datasets</a>
                        <a class="button secondary" routerLink="/public-data/acquire">Acquire public data</a>
                    </div>
                </section>
            } @else {
                <div class="overview-metrics" aria-label="Database composition">
                    <adsmod-metric-card label="Isotherms" [value]="count(data.isotherms)" />
                    <adsmod-metric-card label="Measurements" [value]="count(data.observations)" />
                    <adsmod-metric-card label="Materials" [value]="count(data.materials)" />
                    <adsmod-metric-card label="Adsorbates" [value]="count(data.adsorbates)" />
                    <adsmod-metric-card label="Datasets" [value]="count(data.datasets)" />
                    <adsmod-metric-card label="Mixture isotherms" [value]="count(data.mixture_isotherms)" />
                </div>

                <div class="overview-grid">
                    <section class="console-card overview-panel wide" aria-labelledby="overview-temperature">
                        <div class="panel-head"><h3 id="overview-temperature">Temperature coverage</h3><span>isotherms per temperature bin</span></div>
                        <div class="overview-chart"><adsmod-chart [option]="temperatureOption()" ariaLabel="Histogram of isotherm temperatures" /></div>
                    </section>
                    <section class="console-card overview-panel" aria-labelledby="overview-sources">
                        <div class="panel-head"><h3 id="overview-sources">Isotherms by source</h3></div>
                        <ul class="share-list">
                            @for (entry of sources(); track entry.key) {
                                <li>
                                    <div class="share-label"><span>{{ entry.label }}</span><strong>{{ count(entry.count) }}</strong></div>
                                    <div class="share-track" aria-hidden="true"><span [style.width.%]="entry.share" [style.background]="entry.color"></span></div>
                                </li>
                            }
                        </ul>
                    </section>
                    <section class="console-card overview-panel" aria-labelledby="overview-materials">
                        <div class="panel-head"><h3 id="overview-materials">Most measured materials</h3><span>select to explore</span></div>
                        <div class="overview-chart ranking"><adsmod-chart [option]="materialsOption()" ariaLabel="Materials ranked by isotherm count" (pointClick)="explore('materials', data.top_materials, $event)" /></div>
                    </section>
                    <section class="console-card overview-panel" aria-labelledby="overview-adsorbates">
                        <div class="panel-head"><h3 id="overview-adsorbates">Most measured adsorbates</h3><span>select to explore</span></div>
                        <div class="overview-chart ranking"><adsmod-chart [option]="adsorbatesOption()" ariaLabel="Adsorbates ranked by isotherm count" (pointClick)="explore('adsorbates', data.top_adsorbates, $event)" /></div>
                    </section>
                    <section class="console-card overview-panel wide" aria-labelledby="overview-spans">
                        <div class="panel-head"><h3 id="overview-spans">Measured spans by pressure basis</h3><span>canonical units; bases are never mixed</span></div>
                        <div class="data-table-frame">
                            <table class="data-table">
                                <caption class="sr-only">Pressure, uptake and temperature spans per pressure basis</caption>
                                <thead><tr><th scope="col">Basis</th><th scope="col" class="num">Isotherms</th><th scope="col" class="num">Measurements</th><th scope="col" class="num">Pressure</th><th scope="col" class="num">Uptake (mol/kg)</th><th scope="col" class="num">Temperature (K)</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead>
                                <tbody>
                                    @for (span of data.pressure_bases; track span.pressure_basis) {
                                        <tr>
                                            <td>{{ span.pressure_basis }}</td>
                                            <td class="num">{{ count(span.isotherms) }}</td>
                                            <td class="num">{{ count(span.observations) }}</td>
                                            <td class="num">{{ format(span.pressure_min) }}–{{ format(span.pressure_max) }} {{ unit(span.pressure_unit) }}</td>
                                            <td class="num">{{ format(span.uptake_min_mol_kg) }}–{{ format(span.uptake_max_mol_kg) }}</td>
                                            <td class="num">{{ format(span.temperature_min_k) }}–{{ format(span.temperature_max_k) }}</td>
                                            <td><a class="text-link" routerLink="/dashboards/explore" [queryParams]="{ basis: span.pressure_basis }">Explore</a></td>
                                        </tr>
                                    }
                                </tbody>
                            </table>
                        </div>
                    </section>
                </div>
            }
        }
    `,
})
export class OverviewViewComponent {
    private readonly router = inject(Router);

    protected readonly overview = signal<AnalyticsOverviewResponse | null>(null);
    protected readonly loading = signal(false);
    protected readonly error = signal<string | null>(null);

    protected readonly temperatureOption = computed(() => buildHistogramOption(this.overview()?.temperature_histogram ?? [], 'Temperature (K)'));
    protected readonly materialsOption = computed(() => buildRankingOption(this.overview()?.top_materials ?? [], 'Isotherms'));
    protected readonly adsorbatesOption = computed(() => buildRankingOption(this.overview()?.top_adsorbates ?? [], 'Isotherms', CATEGORICAL_PALETTE[1]));
    protected readonly sources = computed(() => {
        const bySource = this.overview()?.isotherms_by_source ?? {};
        const total = Object.values(bySource).reduce((sum, value) => sum + value, 0) || 1;
        return [
            { key: 'nist', label: 'NIST', color: CATEGORICAL_PALETTE[0] },
            { key: 'uploaded', label: 'Uploaded', color: CATEGORICAL_PALETTE[1] },
        ].map((entry) => ({ ...entry, count: bySource[entry.key] ?? 0, share: ((bySource[entry.key] ?? 0) / total) * 100 }));
    });

    constructor() {
        void this.load();
    }

    protected count(value: number): string {
        return value.toLocaleString();
    }

    protected format(value: number | null): string {
        return formatValue(value);
    }

    protected unit(value: string): string {
        return pressureLabel(value);
    }

    protected explore(key: 'materials' | 'adsorbates', items: FacetCount[], event: ChartPointEvent): void {
        // Ranking bars are drawn bottom-up, so the index is reversed.
        const item = items[items.length - 1 - event.dataIndex];
        if (item) void this.router.navigate(['/dashboards/explore'], { queryParams: { [key]: item.id } });
    }

    async load(): Promise<void> {
        this.loading.set(true);
        const result = await fetchAnalyticsOverview();
        this.loading.set(false);
        this.error.set(result.error);
        if (result.data) this.overview.set(result.data);
    }
}
