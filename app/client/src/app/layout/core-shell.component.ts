// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, ElementRef, HostListener, ViewChild, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';
import { fetchApplicationCapabilities, fetchBackendReadiness } from '../services/system.service';

type HelpPage = 'datasets' | 'public-data' | 'dashboards' | 'fitting' | 'training' | 'settings';

interface HelpStep {
    title: string;
    description: string;
}

interface HelpContent {
    title: string;
    intro: string;
    steps: readonly HelpStep[];
    tips: readonly string[];
}

const HELP_CONTENT: Record<HelpPage, HelpContent> = {
    datasets: {
        title: 'Custom Datasets help',
        intro: 'Manage datasets uploaded directly into the ADSMOD workspace.',
        steps: [
            { title: 'Add a workspace dataset', description: 'Choose a CSV, TXT, XLSX, or JSON file to upload it into the workspace.' },
            { title: 'Review and maintain datasets', description: 'Open a spreadsheet, edit metadata, rename a dataset, or delete it from the workspace.' },
            { title: 'Continue to fitting', description: 'Select an uploaded dataset in Fitting after its import has completed.' },
        ],
        tips: ['Public-source records are discovered and managed in the Public Data workspace.', 'A workspace dataset must be available before it can be selected in Fitting or Training.'],
    },
    'public-data': {
        title: 'Public Data help',
        intro: 'Explore normalized adsorption, materials, chemical, and structural records with source provenance.',
        steps: [
            { title: 'Choose a data view', description: 'Use the workspace tabs to move between adsorption data, materials, chemicals, structures, and provider status.' },
            { title: 'Filter locally cached data', description: 'Use server-side filters and pagination to inspect normalized records without loading entire collections into the browser.' },
            { title: 'Retrieve source records', description: 'Use NIST acquisition, PubChem resolution, or COD structure search where the relevant source supports retrieval.' },
            { title: 'Inspect provenance', description: 'Open a record to review source identifiers, retrieval information, normalized values, and references.' },
        ],
        tips: ['NIST remains the adsorption acquisition source, while PubChem and COD add chemical and structural information.', 'External provider outages do not remove locally cached records.'],
    },
    dashboards: {
        title: 'Dashboards help',
        intro: 'Explore every stored isotherm across datasets. Charts are computed from the database on demand; no aggregated copy is created.',
        steps: [
            { title: 'Review the overview', description: 'See how measurements are distributed across temperatures, sources, materials, adsorbates, and pressure bases.' },
            { title: 'Build a query', description: 'In Explore & compare, filter by source, pressure basis, temperature, material, adsorbate, or dataset, and optionally select individual isotherms.' },
            { title: 'Choose an analysis', description: 'Plot isotherms coloured by temperature or category, compare uptake at one pressure, or inspect a distribution.' },
            { title: 'Inspect a series', description: 'Select a curve or row to see its provenance, material, and adsorbate details.' },
        ],
        tips: [
            'Uptake at a fixed pressure uses a measurement when one matches, otherwise interpolates in log pressure between bracketing points of the adsorption branch; it never extrapolates.',
            'The page address stores the query, so a view can be bookmarked and is always recomputed from current data.',
        ],
    },
    fitting: {
        title: 'Fitting help',
        intro: 'Configure an adsorption fit and compare enabled model equations against a selected dataset.',
        steps: [
            { title: 'Choose a dataset', description: 'Select a workspace or NIST dataset from the Dataset control.' },
            { title: 'Set the run options', description: 'Choose the maximum iterations and optimization method for the fit.' },
            { title: 'Choose models', description: 'Enable or disable the adsorption models you want to compare.' },
            { title: 'Start fitting', description: 'Use Start Fitting to run the workflow and follow its progress in the fitting log.' },
        ],
        tips: ['Reset Log clears the visible fitting messages.', 'At least one model should remain enabled for a meaningful comparison.'],
    },
    training: {
        title: 'Training help',
        intro: 'Prepare data, configure training, review checkpoints, and monitor training runs.',
        steps: [
            { title: 'Prepare data', description: 'Use Data Processing to select and prepare the datasets needed for training.' },
            { title: 'Configure a run', description: 'Use the training setup view to choose the dataset, model, and run parameters.' },
            { title: 'Monitor progress', description: 'Use the dashboard to follow active metrics and review completed runs.' },
            { title: 'Review checkpoints', description: 'Open Checkpoints to inspect saved training artifacts before reusing them.' },
        ],
        tips: ['Training requires the optional machine learning dependencies.', 'Keep the status and metric panels visible while a run is active.'],
    },
    settings: {
        title: 'Settings help',
        intro: 'Manage application defaults that affect future requests and jobs.',
        steps: [
            { title: 'Choose a category', description: 'Use the tabs to move between runtime, import, public-data, fitting, and training settings.' },
            { title: 'Save or discard edits', description: 'Save applies validated changes; Cancel restores the last saved values.' },
            { title: 'Restart when prompted', description: 'Host and port changes are stored immediately and take effect after ADSMOD restarts.' },
        ],
        tips: ['Database connection values and storage paths remain bootstrap settings in settings/.env.', 'A running job keeps the settings snapshot it started with.'],
    },
};

@Component({
    selector: 'adsmod-core-shell',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet],
    template: `
        <div class="console-shell">
            <aside #sidebar class="console-sidebar" aria-label="Primary navigation">
                <div class="console-brand">
                    <img class="console-brand-logo" src="/adsmod-logo-96.png" width="43" height="43" alt="" aria-hidden="true" />
                    <div>
                        <div class="console-brand-name">ADSMOD</div>
                        <div class="console-brand-subtitle">Adsorption Modeling<br />Unified Console</div>
                    </div>
                </div>

                <nav class="console-nav" aria-label="Primary">
                    <a class="console-nav-item" routerLink="/datasets" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">□</span>
                        <span>Custom Datasets</span>
                    </a>
                    <a class="console-nav-item" routerLink="/public-data" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">⇩</span>
                        <span>Public Data</span>
                    </a>
                    <a class="console-nav-item" routerLink="/dashboards" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">▦</span>
                        <span>Dashboards</span>
                    </a>
                    <a class="console-nav-item" routerLink="/fitting" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">⌁</span>
                        <span>Fitting</span>
                    </a>
                    @if (machineLearningAvailable()) {
                        <a class="console-nav-item" routerLink="/training" routerLinkActive="active" ariaCurrentWhenActive="page">
                            <span class="console-nav-icon" aria-hidden="true">✺</span>
                            <span>Training</span>
                        </a>
                    }
                </nav>

                <div class="console-sidebar-footer">
                    <button class="console-footer-link" type="button" aria-label="Docs, not available yet" title="Documentation is not available yet." disabled>
                        <span aria-hidden="true">?</span>
                        <span>Docs</span>
                    </button>
                    <a class="console-footer-link" routerLink="/settings" routerLinkActive="active" ariaCurrentWhenActive="page" aria-label="Settings" title="Application settings">
                        <span aria-hidden="true">⚙</span>
                        <span>Settings</span>
                    </a>
                </div>
            </aside>

            <section class="console-stage">
                <header class="console-header">
                    <div>
                        <h1>{{ pageTitle() }}</h1>
                        <p>{{ pageDescription() }}</p>
                    </div>
                    <div class="service-status-row">
                        <button #helpTrigger class="header-icon-button" type="button" aria-label="Help" (click)="openHelp()">?</button>
                    </div>
                </header>
                <main #mainContent class="app-main console-main">
                    <router-outlet />
                </main>
            </section>
        </div>
        <div class="console-status-bar" aria-label="Backend status" aria-live="polite">
            <div class="console-status-item"><span class="service-dot core" [class.offline]="backendStatus() === 'Offline'" aria-hidden="true"></span><strong>Backend</strong><em>{{ backendStatus() }}</em></div>
        </div>
        @if (helpOpen()) {
            <div class="help-modal-backdrop" (click)="closeHelp()">
                <section #helpDialog class="help-modal" role="dialog" aria-modal="true" aria-labelledby="help-modal-title" (click)="$event.stopPropagation()" (keydown)="handleHelpKeydown($event)">
                    <div class="help-modal-header">
                        <div>
                            <p class="eyebrow">Page guide</p>
                            <h2 id="help-modal-title">{{ helpContent().title }}</h2>
                            <p>{{ helpContent().intro }}</p>
                        </div>
                        <button #helpCloseButton class="button quiet help-modal-close" type="button" aria-label="Close help" title="Close help" autofocus (click)="closeHelp()"><span aria-hidden="true">×</span></button>
                    </div>
                    <div class="help-modal-body">
                        <h3>How to use this page</h3>
                        <ol class="help-step-list">
                            @for (step of helpContent().steps; track step.title) {
                                <li><strong>{{ step.title }}</strong><span>{{ step.description }}</span></li>
                            }
                        </ol>
                        <h3>Helpful tips</h3>
                        <ul class="help-tip-list">
                            @for (tip of helpContent().tips; track tip) {
                                <li>{{ tip }}</li>
                            }
                        </ul>
                    </div>
                    <div class="help-modal-footer">
                        <button class="button primary" type="button" (click)="closeHelp()">Done</button>
                    </div>
                </section>
            </div>
        }
    `,
})
export class CoreShellComponent {
    @ViewChild('helpTrigger') private helpTrigger?: ElementRef<HTMLButtonElement>;
    @ViewChild('helpCloseButton') private helpCloseButton?: ElementRef<HTMLButtonElement>;
    @ViewChild('helpDialog') private helpDialog?: ElementRef<HTMLElement>;
    @ViewChild('mainContent') private mainContent?: ElementRef<HTMLElement>;
    @ViewChild('sidebar') private sidebar?: ElementRef<HTMLElement>;
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);
    private readonly currentUrl = signal(this.router.url);
    protected readonly helpOpen = signal(false);
    protected readonly backendStatus = signal<'Checking' | 'Online' | 'Offline'>('Checking');
    protected readonly machineLearningAvailable = signal(false);
    protected readonly currentPage = computed<HelpPage>(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'dashboards';
        }
        if (url.startsWith('/public-data')) {
            return 'public-data';
        }
        if (url.startsWith('/fitting')) {
            return 'fitting';
        }
        if (url.startsWith('/training')) {
            return 'training';
        }
        if (url.startsWith('/settings')) {
            return 'settings';
        }
        return 'datasets';
    });
    protected readonly helpContent = computed(() => HELP_CONTENT[this.currentPage()]);
    protected readonly pageTitle = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'Dashboards';
        }
        if (url.startsWith('/public-data')) {
            return 'Public Data';
        }
        if (url.startsWith('/fitting')) {
            return 'Fitting';
        }
        if (url.startsWith('/training')) {
            return 'Training';
        }
        if (url.startsWith('/settings')) {
            return 'Settings';
        }
        return 'Custom Datasets';
    });
    protected readonly pageDescription = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'Query, compare, and visualise stored adsorption data without copying it.';
        }
        if (url.startsWith('/public-data')) {
            return 'Discover, normalize, inspect, and trace scientific records from integrated public sources.';
        }
        if (url.startsWith('/fitting')) {
            return 'Configure and run adsorption model fitting workflows.';
        }
        if (url.startsWith('/training')) {
            return 'Prepare datasets, checkpoints, and model training workflows.';
        }
        if (url.startsWith('/settings')) {
            return 'Configure application defaults and provider behavior.';
        }
        return 'Manage datasets uploaded directly into the workspace.';
    });

    protected openHelp(): void {
        this.helpOpen.set(true);
        setTimeout(() => this.helpCloseButton?.nativeElement.focus());
    }

    protected closeHelp(): void {
        this.helpOpen.set(false);
        queueMicrotask(() => this.helpTrigger?.nativeElement.focus());
    }

    protected handleHelpKeydown(event: KeyboardEvent): void {
        if (event.key !== 'Tab' || !this.helpOpen()) {
            return;
        }

        const dialog = this.helpDialog?.nativeElement;
        if (!dialog) {
            return;
        }

        const focusableElements = Array.from(
            dialog.querySelectorAll<HTMLElement>(
                'a[href], button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'
            )
        );
        if (focusableElements.length === 0) {
            event.preventDefault();
            return;
        }

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        const activeElement = document.activeElement;
        if (!dialog.contains(activeElement)) {
            event.preventDefault();
            firstElement.focus();
        } else if (event.shiftKey && activeElement === firstElement) {
            event.preventDefault();
            lastElement.focus();
        } else if (!event.shiftKey && activeElement === lastElement) {
            event.preventDefault();
            firstElement.focus();
        }
    }

    constructor() {
        void this.refreshBackendStatus();
        const serviceStatusTimer = window.setInterval(() => void this.refreshBackendStatus(), 10_000);
        this.destroyRef.onDestroy(() => window.clearInterval(serviceStatusTimer));
        this.router.events
            .pipe(
                filter((event): event is NavigationEnd => event instanceof NavigationEnd),
                takeUntilDestroyed(this.destroyRef)
            )
            .subscribe((event) => {
                this.currentUrl.set(event.urlAfterRedirects);
                void this.refreshBackendStatus();
                queueMicrotask(() => {
                    if (this.mainContent) {
                        this.mainContent.nativeElement.scrollTop = 0;
                        this.mainContent.nativeElement.scrollLeft = 0;
                    }
                    if (this.sidebar) {
                        this.sidebar.nativeElement.scrollTop = 0;
                        this.sidebar.nativeElement.scrollLeft = 0;
                    }
                });
                this.revealActiveNavItem();
            });
        this.revealActiveNavItem();
    }

    /** On narrow screens the nav is a scrollable row; keep the current page's item visible. */
    private revealActiveNavItem(): void {
        setTimeout(() => {
            const active = this.sidebar?.nativeElement.querySelector<HTMLElement>('.console-nav-item.active');
            active?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
        });
    }

    private async refreshBackendStatus(): Promise<void> {
        const [capabilities, readiness] = await Promise.all([fetchApplicationCapabilities(true), fetchBackendReadiness()]);
        const backendOnline = capabilities.data !== null && readiness.data?.state === 'ready';
        this.backendStatus.set(backendOnline ? 'Online' : 'Offline');
        this.machineLearningAvailable.set(capabilities.data?.features.machine_learning === true);
    }

    @HostListener('document:keydown.escape')
    protected handleEscape(): void {
        if (this.helpOpen()) {
            this.closeHelp();
        }
    }
}
