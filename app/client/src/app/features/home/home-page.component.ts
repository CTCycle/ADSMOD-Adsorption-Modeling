// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, OnInit, computed, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import type { HomeSummary } from '../../models/home.model';
import { fetchHomeSummary } from '../../services/home.service';
import { machineLearningAvailable } from '../../services/system.service';
import { fetchCheckpoints } from '../../services/training.service';

@Component({
    selector: 'adsmod-home-page',
    standalone: true,
    imports: [RouterLink, DecimalPipe, DatePipe],
    styles: [
        `
            .home-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; max-width: 1200px; }
            .home-card { background: var(--surface); border: 1px solid var(--slate-200); border-radius: 6px; padding: 1rem 1.1rem; display: flex; flex-direction: column; gap: 0.8rem; }
            .home-card h2 { margin: 0; font-size: 1rem; }
            .home-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; margin: 0; }
            .home-stat { display: block; padding: 0.55rem 0.7rem; border: 1px solid var(--slate-200); border-radius: 6px; color: inherit; text-decoration: none; }
            a.home-stat:hover, a.home-stat:focus-visible { border-color: var(--primary-600); background: var(--surface-subtle, var(--slate-50)); }
            .home-stat strong { display: block; font-size: 1.35rem; line-height: 1.2; }
            .home-stat span { font-size: 0.78rem; color: var(--text-secondary); }
            .home-note { margin: 0; font-size: 0.82rem; color: var(--text-secondary); }
            .home-link { margin-top: auto; font-size: 0.85rem; }
            .home-error { color: var(--danger-600, #b42318); }
        `,
    ],
    template: `
        @if (error(); as message) {
            <p class="home-error" role="alert">{{ message }}</p>
        }
        @if (summary(); as data) {
            <div class="home-grid">
                <section class="home-card" aria-labelledby="home-data">
                    <h2 id="home-data">Data</h2>
                    <div class="home-stats">
                        <a class="home-stat" routerLink="/datasets"><strong>{{ data.datasets | number }}</strong><span>datasets</span></a>
                        <a class="home-stat" routerLink="/dashboards"><strong>{{ data.experiments | number }}</strong><span>experiments</span></a>
                        <a class="home-stat" routerLink="/datasets" [queryParams]="{ kind: 'species' }"><strong>{{ data.species | number }}</strong><span>chemical species</span></a>
                        <a class="home-stat" routerLink="/datasets" [queryParams]="{ kind: 'material' }"><strong>{{ data.materials | number }}</strong><span>adsorbent materials</span></a>
                    </div>
                    <p class="home-note">Species and materials count every stored record, uploaded or acquired.</p>
                </section>

                <section class="home-card" aria-labelledby="home-sources">
                    <h2 id="home-sources">Sources</h2>
                    <div class="home-stats">
                        <div class="home-stat"><strong>{{ data.sources.length | number }}</strong><span>external sources in use</span></div>
                        <div class="home-stat"><strong>{{ totalRecords() | number }}</strong><span>source records</span></div>
                    </div>
                    <p class="home-note">
                        @if (data.sources.length) {
                            {{ sourceNames() }}
                        } @else {
                            No external provider data has been acquired yet.
                        }
                    </p>
                    <a class="home-link" routerLink="/sources/catalog">Open source catalog</a>
                </section>

                <section class="home-card" aria-labelledby="home-modeling">
                    <h2 id="home-modeling">Modeling</h2>
                    <div class="home-stats">
                        <a class="home-stat" routerLink="/modeling/fitting"><strong>{{ data.fitting_sessions | number }}</strong><span>fitting sessions</span></a>
                        @if (checkpointCount() !== null) {
                            <a class="home-stat" routerLink="/modeling/deep-learning"><strong>{{ checkpointCount() | number }}</strong><span>trained models</span></a>
                        } @else {
                            <a class="home-stat" routerLink="/modeling/models"><strong>{{ completedSessions() | number }}</strong><span>completed sessions</span></a>
                        }
                    </div>
                    <p class="home-note">
                        @if (data.latest_fitting_at; as latest) {
                            Latest session {{ latest | date: 'medium' }}.
                        } @else {
                            No fitting session has been run yet.
                        }
                    </p>
                    <a class="home-link" routerLink="/modeling/fitting">Open modeling</a>
                </section>
            </div>
        } @else if (!error()) {
            <p class="home-note" aria-live="polite">Loading overview…</p>
        }
    `,
})
export class HomePageComponent implements OnInit {
    protected readonly summary = signal<HomeSummary | null>(null);
    protected readonly error = signal<string | null>(null);
    protected readonly checkpointCount = signal<number | null>(null);
    protected readonly totalRecords = computed(() => (this.summary()?.sources ?? []).reduce((sum, source) => sum + source.record_count, 0));
    protected readonly sourceNames = computed(() => (this.summary()?.sources ?? []).map((source) => source.name).join(' · '));
    protected readonly completedSessions = computed(() => {
        const byStatus = this.summary()?.fitting_sessions_by_status ?? {};
        return (byStatus['completed'] ?? 0) + (byStatus['warning'] ?? 0);
    });

    async ngOnInit(): Promise<void> {
        const [result, mlAvailable] = await Promise.all([fetchHomeSummary(), machineLearningAvailable()]);
        this.summary.set(result.data);
        this.error.set(result.error);
        if (mlAvailable) {
            const checkpoints = await fetchCheckpoints();
            this.checkpointCount.set(checkpoints.error ? null : checkpoints.checkpoints.length);
        }
    }
}
