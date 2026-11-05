// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, ElementRef, ViewChild, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';
import { ThemeSwitchComponent } from '../shared/components/theme-switch/theme-switch.component';
import { fetchApplicationCapabilities, fetchBackendReadiness } from '../services/system.service';
import { DOCS_ANCHOR_BY_PAGE } from '../features/docs/docs-content';

type DocsPage = keyof typeof DOCS_ANCHOR_BY_PAGE;

@Component({
    selector: 'adsmod-core-shell',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet, ThemeSwitchComponent],
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
                        <span>Datasets</span>
                    </a>
                    <a class="console-nav-item" routerLink="/sources" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">⇩</span>
                        <span>Sources</span>
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
                    <a class="console-footer-link" routerLink="/help" [fragment]="docsAnchor()" routerLinkActive="active" ariaCurrentWhenActive="page" aria-label="Docs" title="Documentation, tutorials, and tips">
                        <span aria-hidden="true">?</span>
                        <span>Docs</span>
                    </a>
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
                        <adsmod-theme-switch />
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
    `,
})
export class CoreShellComponent {
    @ViewChild('mainContent') private mainContent?: ElementRef<HTMLElement>;
    @ViewChild('sidebar') private sidebar?: ElementRef<HTMLElement>;
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);
    private readonly currentUrl = signal(this.router.url);
    protected readonly backendStatus = signal<'Checking' | 'Online' | 'Offline'>('Checking');
    protected readonly machineLearningAvailable = signal(false);
    protected readonly docsAnchor = computed<string>(() => {
        const url = this.currentUrl();
        const page: DocsPage = url.startsWith('/dashboards')
            ? 'dashboards'
            : url.startsWith('/sources')
              ? 'sources'
              : url.startsWith('/fitting')
                ? 'fitting'
                : url.startsWith('/training')
                  ? 'training'
                  : url.startsWith('/settings')
                    ? 'settings'
                    : 'datasets';
        return DOCS_ANCHOR_BY_PAGE[page];
    });
    protected readonly pageTitle = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'Dashboards';
        }
        if (url.startsWith('/sources')) {
            return 'Sources';
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
        if (url.startsWith('/help')) {
            return 'Docs';
        }
        return url.startsWith('/datasets/') ? 'Dataset' : 'Datasets';
    });
    protected readonly pageDescription = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/dashboards')) {
            return 'Query, compare, and visualise stored adsorption data without copying it.';
        }
        if (url.startsWith('/sources')) {
            return 'Discover external data providers and acquire their records into the local database.';
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
        if (url.startsWith('/help')) {
            return 'Tutorials, guides, and tips for every ADSMOD workflow.';
        }
        return url.startsWith('/datasets/')
            ? 'Inspect and edit experiments and their measurements.'
            : 'Manage local adsorption datasets, adsorbent materials, and chemical species.';
    });

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
}
