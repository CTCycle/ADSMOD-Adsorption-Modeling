import { Component, DestroyRef, ElementRef, HostListener, ViewChild, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';

type HelpPage = 'datasets' | 'dashboards' | 'fitting' | 'training';

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
        title: 'Datasets help',
        intro: 'Manage workspace datasets and fetch public adsorption source data in one place.',
        steps: [
            { title: 'Add a workspace dataset', description: 'Choose a CSV, TXT, XLSX, or JSON file to upload it into the workspace.' },
            { title: 'Review and maintain datasets', description: 'Open a spreadsheet, edit metadata, rename a dataset, or delete it from the workspace.' },
            { title: 'Use public data', description: 'Scroll to NIST-A Collection to check connectivity, update the index, fetch records, or enrich properties.' },
        ],
        tips: ['A workspace dataset must be available before it can be selected in Fitting or Training.', 'NIST progress and messages remain available while a category job is running.'],
    },
    dashboards: {
        title: 'Dashboards help',
        intro: 'Dashboards will provide a consolidated view of workspace activity and results.',
        steps: [{ title: 'Coming next', description: 'Dashboard views are not available yet.' }],
        tips: ['Use Datasets to manage workspace and public source data in the meantime.'],
    },
    fitting: {
        title: 'Fitting help',
        intro: 'Configure an adsorption fit and compare enabled model equations against a selected dataset.',
        steps: [
            { title: 'Choose a dataset', description: 'Select a workspace or NIST-A dataset from the Dataset control.' },
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
        tips: ['Training requires the optional ML service.', 'Keep the status and metric panels visible while a run is active.'],
    },
};

@Component({
    selector: 'adsmod-core-shell',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet],
    template: `
        <div class="console-shell">
            <aside class="console-sidebar" aria-label="Primary navigation">
                <div class="console-brand">
                    <img class="console-brand-logo" src="/adsmod-logo-96.png" width="43" height="43" alt="" aria-hidden="true" />
                    <div>
                        <div class="console-brand-name">ADSMOD</div>
                        <div class="console-brand-subtitle">Adsorption Modeling<br />Unified Console</div>
                    </div>
                </div>

                <nav class="console-nav" aria-label="Primary">
                    <a class="console-nav-item" routerLink="/datasets" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">□</span>
                        <span>Datasets</span>
                    </a>
                    <a class="console-nav-item" routerLink="/dashboards" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">▦</span>
                        <span>Dashboards</span>
                    </a>
                    <a class="console-nav-item" routerLink="/fitting" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">⌁</span>
                        <span>Fitting</span>
                    </a>
                    <a class="console-nav-item" routerLink="/training" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">✺</span>
                        <span>Training</span>
                    </a>
                </nav>

                <div class="console-sidebar-footer">
                    <button class="console-footer-link" type="button" aria-label="Docs (not available)">
                        <span aria-hidden="true">?</span>
                        <span>Docs</span>
                    </button>
                    <button class="console-footer-link" type="button" aria-label="Settings (not available)">
                        <span aria-hidden="true">⚙</span>
                        <span>Settings</span>
                    </button>
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
                <main class="app-main console-main">
                    <router-outlet />
                </main>
            </section>
        </div>
        <div class="console-status-bar" aria-label="Service status">
            <div class="console-status-item"><span class="service-dot core" aria-hidden="true"></span><strong>Core Service</strong><em>Online</em></div>
            <div class="console-status-item"><span class="service-dot ml" aria-hidden="true"></span><strong>ML Service</strong><em>Optional / Available</em></div>
        </div>
        @if (helpOpen()) {
            <div class="help-modal-backdrop" (click)="closeHelp()">
                <section class="help-modal" role="dialog" aria-modal="true" aria-labelledby="help-modal-title" (click)="$event.stopPropagation()">
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
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);
    private readonly currentUrl = signal(this.router.url);
    protected readonly helpOpen = signal(false);
    protected readonly currentPage = computed<HelpPage>(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'dashboards';
        }
        if (url.startsWith('/fitting')) {
            return 'fitting';
        }
        if (url.startsWith('/training')) {
            return 'training';
        }
        return 'datasets';
    });
    protected readonly helpContent = computed(() => HELP_CONTENT[this.currentPage()]);
    protected readonly pageTitle = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'Dashboards';
        }
        if (url.startsWith('/fitting')) {
            return 'Fitting';
        }
        if (url.startsWith('/training')) {
            return 'Training';
        }
        return 'Datasets';
    });
    protected readonly pageDescription = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'Monitor workspace activity and results.';
        }
        if (url.startsWith('/fitting')) {
            return 'Configure and run adsorption model fitting workflows.';
        }
        if (url.startsWith('/training')) {
            return 'Prepare datasets, checkpoints, and model training workflows.';
        }
        return 'Manage workspace datasets and public adsorption source data.';
    });

    protected openHelp(): void {
        this.helpOpen.set(true);
        setTimeout(() => this.helpCloseButton?.nativeElement.focus());
    }

    protected closeHelp(): void {
        this.helpOpen.set(false);
        queueMicrotask(() => this.helpTrigger?.nativeElement.focus());
    }

    constructor() {
        this.router.events
            .pipe(
                filter((event): event is NavigationEnd => event instanceof NavigationEnd),
                takeUntilDestroyed(this.destroyRef)
            )
                .subscribe((event) => this.currentUrl.set(event.urlAfterRedirects));
    }

    @HostListener('document:keydown.escape')
    protected handleEscape(): void {
        if (this.helpOpen()) {
            this.closeHelp();
        }
    }
}
