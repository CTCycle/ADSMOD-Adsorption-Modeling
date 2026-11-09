// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, ElementRef, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { Meta, Title } from '@angular/platform-browser';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';
import { ThemeSwitchComponent } from '../shared/components/theme-switch/theme-switch.component';
import type { SystemStatus } from '../models/job.model';
import { fetchApplicationCapabilities, fetchBackendReadiness, fetchSystemStatus } from '../services/system.service';
import { formatBytes } from '../shared/utils/job-presentation';
import { JobsPopoverComponent } from './jobs-popover.component';
import { DOCS_ANCHOR_BY_PAGE } from '../features/docs/docs-content';

type DocsPage = keyof typeof DOCS_ANCHOR_BY_PAGE;

@Component({
    selector: 'adsmod-core-shell',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet, ThemeSwitchComponent, JobsPopoverComponent],
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
                    <a class="console-nav-item" routerLink="/home" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">⌂</span>
                        <span>Home</span>
                    </a>
                    <a class="console-nav-item" routerLink="/sources" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">⇩</span>
                        <span>Sources</span>
                    </a>
                    <a class="console-nav-item" routerLink="/datasets" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">□</span>
                        <span>Datasets</span>
                    </a>
                    <a class="console-nav-item" routerLink="/dashboards" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">▦</span>
                        <span>Dashboards</span>
                    </a>
                    <a class="console-nav-item" routerLink="/modeling" routerLinkActive="active" ariaCurrentWhenActive="page">
                        <span class="console-nav-icon" aria-hidden="true">⌁</span>
                        <span>Modeling</span>
                    </a>
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
        <div class="console-status-bar" aria-label="Application status">
            <div class="console-status-item" aria-live="polite"><span class="service-dot core" [class.offline]="backendStatus() === 'Offline'" aria-hidden="true"></span><strong>Backend</strong><em>{{ backendStatus() }}</em></div>
            @if (metrics(); as usage) {
                <div class="console-status-item metric" title="Backend process CPU usage"><span>CPU</span><em>{{ usage.cpu_percent.toFixed(0) }}%</em></div>
                <div class="console-status-item metric" title="Backend process memory"><span>RAM</span><em>{{ formatBytes(usage.memory_mb * 1048576) }}</em></div>
                @if (usage.database_size_bytes !== null) {
                    <div class="console-status-item metric" title="Database size on disk"><span>DB</span><em>{{ formatBytes(usage.database_size_bytes) }}</em></div>
                }
            }
            <adsmod-jobs-popover />
        </div>
    `,
})
export class CoreShellComponent {
    @ViewChild('mainContent') private mainContent?: ElementRef<HTMLElement>;
    @ViewChild('sidebar') private sidebar?: ElementRef<HTMLElement>;
    private readonly router = inject(Router);
    private readonly title = inject(Title);
    private readonly meta = inject(Meta);
    private readonly destroyRef = inject(DestroyRef);
    private statusRefresh: Promise<void> | null = null;
    private readonly currentUrl = signal(this.router.url);
    protected readonly backendStatus = signal<'Checking' | 'Online' | 'Offline'>('Checking');
    protected readonly metrics = signal<SystemStatus | null>(null);
    protected readonly formatBytes = formatBytes;
    protected readonly docsAnchor = computed<string>(() => {
        const url = this.currentUrl();
        const page: DocsPage = url.startsWith('/dashboards')
            ? 'dashboards'
            : url.startsWith('/sources')
              ? 'sources'
              : url.startsWith('/modeling')
                ? 'modeling'
                : url.startsWith('/settings')
                  ? 'settings'
                  : 'datasets';
        return DOCS_ANCHOR_BY_PAGE[page];
    });
    protected readonly pageTitle = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/home')) {
            return 'Home';
        }
        if (url.startsWith('/dashboards')) {
            return 'Dashboards';
        }
        if (url.startsWith('/sources')) {
            return 'Sources';
        }
        if (url.startsWith('/modeling')) {
            return 'Modeling';
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
        if (url.startsWith('/home')) {
            return 'The current state of your data, sources, and models.';
        }
        if (url.startsWith('/dashboards')) {
            return 'Compare and visualise stored adsorption data.';
        }
        if (url.startsWith('/sources')) {
            return 'Find and import data from external providers.';
        }
        if (url.startsWith('/modeling')) {
            return 'Fit adsorption models, manage the model catalog, and train deep learning models.';
        }
        if (url.startsWith('/settings')) {
            return 'Application defaults and provider options.';
        }
        if (url.startsWith('/help')) {
            return 'Guides and tips for every workflow.';
        }
        return url.startsWith('/datasets/')
            ? 'Inspect and edit experiments and measurements.'
            : 'Browse your datasets, materials, and species.';
    });

    constructor() {
        effect(() => {
            this.title.setTitle(`${this.pageTitle()} · ADSMOD`);
            this.meta.updateTag({ name: 'description', content: this.pageDescription() });
        });
        void this.refreshBackendStatus();
        const serviceStatusTimer = window.setInterval(() => {
            if (!document.hidden) {
                void this.refreshBackendStatus();
            }
        }, 10_000);
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

    /** Shares one in-flight request so a slow backend cannot accumulate overlapping polls. */
    private refreshBackendStatus(): Promise<void> {
        this.statusRefresh ??= this.fetchBackendStatus().finally(() => {
            this.statusRefresh = null;
        });
        return this.statusRefresh;
    }

    private async fetchBackendStatus(): Promise<void> {
        const [capabilities, readiness, usage] = await Promise.all([
            fetchApplicationCapabilities(true),
            fetchBackendReadiness(),
            fetchSystemStatus(),
        ]);
        const backendOnline = capabilities.data !== null && readiness.data?.state === 'ready';
        this.backendStatus.set(backendOnline ? 'Online' : 'Offline');
        this.metrics.set(backendOnline ? usage.data : null);
    }
}
