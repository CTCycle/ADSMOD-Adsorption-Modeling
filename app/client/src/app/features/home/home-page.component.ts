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
            .home-card { border-top: 3px solid var(--primary-600); }
            .home-card h2 { font-size: 0.95rem; letter-spacing: 0.02em; }
            .home-stat strong { font-variant-numeric: tabular-nums; }
            .home-flow { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 1.5rem; max-width: 1200px; margin-top: 1rem; padding: 1rem 1.4rem; background: linear-gradient(100deg, var(--surface) 55%, var(--primary-50)); border: 1px solid var(--slate-200); border-radius: 6px; }
            .home-flow h2 { margin: 0 0 0.15rem; font-size: 0.95rem; letter-spacing: 0.02em; }
            .home-flow ol { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.6rem; margin: 0.8rem 0 0; padding: 0; list-style: none; }
            .home-step { display: grid; grid-template-columns: auto 1fr; gap: 0 0.65rem; align-items: center; height: 100%; padding: 0.55rem 0.7rem; border: 1px solid var(--slate-200); border-radius: 6px; background: var(--surface); color: inherit; text-decoration: none; }
            .home-step:hover, .home-step:focus-visible { border-color: var(--primary-600); }
            .home-step-index { grid-row: span 2; display: grid; place-items: center; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--primary-50); color: var(--primary-600); font-size: 0.8rem; font-weight: 700; }
            .home-step strong { font-size: 0.88rem; }
            .home-step span:not(.home-step-index) { font-size: 0.76rem; color: var(--text-secondary); }
            .home-art { width: 21rem; height: auto; color: var(--primary-600); }
            .home-art .art-bond { stroke: currentColor; stroke-opacity: 0.35; stroke-width: 1.6; }
            .home-art .art-node { fill: currentColor; }
            .home-art .art-pore { fill: currentColor; fill-opacity: 0.06; }
            .home-art .art-molecules { color: var(--warning-text); }
            .home-art .art-gas { opacity: 0.55; }
            .home-art .art-arrow { stroke-dasharray: 3 3; stroke-linecap: round; stroke-linejoin: round; stroke-opacity: 0.7; }
            @media (max-width: 820px) { .home-flow { grid-template-columns: minmax(0, 1fr); } .home-art { display: none; } }
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

            <section class="home-flow" aria-labelledby="home-flow">
                <div>
                    <h2 id="home-flow">From source to model</h2>
                    <p class="home-note">Bring adsorption data in, tidy it, then fit isotherm models.</p>
                    <ol>
                        <li><a class="home-step" routerLink="/sources/acquire"><span class="home-step-index">1</span><strong>Acquire</strong><span>Fetch isotherms, species, structures</span></a></li>
                        <li><a class="home-step" routerLink="/datasets"><span class="home-step-index">2</span><strong>Curate</strong><span>Browse and edit stored datasets</span></a></li>
                        <li><a class="home-step" routerLink="/modeling/fitting"><span class="home-step-index">3</span><strong>Model</strong><span>Fit isotherms and train models</span></a></li>
                    </ol>
                </div>
                <!-- Schematic of gas molecules adsorbed in a porous framework (original artwork). -->
                <svg class="home-art" viewBox="0 0 330 130" role="img" aria-label="Gas molecules adsorbed in the pores of a porous framework">
                    <defs>
                        <g id="home-molecule" fill="currentColor" stroke="currentColor" stroke-width="1.6"><line x1="-4.5" y1="0" x2="4.5" y2="0" /><circle cx="-4.5" cy="0" r="3.4" stroke="none" /><circle cx="4.5" cy="0" r="3.4" stroke="none" /></g>
                    </defs>
                    <g class="art-molecules art-gas">
                        <use href="#home-molecule" x="18" y="30" transform="rotate(35 18 30)" />
                        <use href="#home-molecule" x="52" y="52" transform="rotate(-20 52 52)" />
                        <use href="#home-molecule" x="22" y="78" transform="rotate(80 22 78)" />
                        <use href="#home-molecule" x="60" y="104" transform="rotate(-50 60 104)" />
                    </g>
                    <path class="art-bond art-arrow" fill="none" d="M82 65H108M102 59l6 6-6 6" />
                    <g transform="translate(90 0)">
                        <rect class="art-pore" x="30" y="25" width="60" height="40" />
                        <rect class="art-pore" x="150" y="25" width="60" height="40" />
                        <rect class="art-pore" x="90" y="65" width="60" height="40" />
                        <path class="art-bond" fill="none" d="M30 25H210M30 65H210M30 105H210M30 25V105M90 25V105M150 25V105M210 25V105" />
                        <g class="art-node">
                            <circle cx="30" cy="25" r="4" /><circle cx="90" cy="25" r="4" /><circle cx="150" cy="25" r="4" /><circle cx="210" cy="25" r="4" />
                            <circle cx="30" cy="65" r="4" /><circle cx="90" cy="65" r="4" /><circle cx="150" cy="65" r="4" /><circle cx="210" cy="65" r="4" />
                            <circle cx="30" cy="105" r="4" /><circle cx="90" cy="105" r="4" /><circle cx="150" cy="105" r="4" /><circle cx="210" cy="105" r="4" />
                        </g>
                        <g class="art-molecules">
                            <use href="#home-molecule" x="60" y="45" transform="rotate(-25 60 45)" />
                            <use href="#home-molecule" x="180" y="45" transform="rotate(30 180 45)" />
                            <use href="#home-molecule" x="120" y="85" transform="rotate(-60 120 85)" />
                            <use href="#home-molecule" x="120" y="45" transform="rotate(70 120 45)" />
                            <use href="#home-molecule" x="60" y="85" transform="rotate(10 60 85)" />
                        </g>
                    </g>
                </svg>
            </section>
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
