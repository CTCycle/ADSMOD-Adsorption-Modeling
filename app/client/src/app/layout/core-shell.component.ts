import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';

@Component({
    selector: 'adsmod-core-shell',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet],
    template: `
        <div class="console-shell">
            <aside class="console-sidebar" aria-label="Primary navigation">
                <div class="console-brand">
                    <div class="molecule-mark" aria-hidden="true">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <div>
                        <div class="console-brand-name">ADSMOD</div>
                        <div class="console-brand-subtitle">Adsorption Modeling<br />Unified Console</div>
                    </div>
                </div>

                <nav class="console-nav" aria-label="Primary">
                    <a class="console-nav-item" routerLink="/source" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">□</span>
                        <span>User Data</span>
                    </a>
                    <a class="console-nav-item" routerLink="/nist" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">⇣</span>
                        <span>NIST Data</span>
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
                    <div class="console-user">
                        <span class="console-avatar" aria-hidden="true">RS</span>
                        <span>rsmith</span>
                        <span aria-hidden="true">⌄</span>
                    </div>
                </div>
            </aside>

            <section class="console-stage">
                <header class="console-header">
                    <div>
                        <h1>{{ pageTitle() }}</h1>
                        <p>{{ pageDescription() }}</p>
                    </div>
                    <div class="service-status-row" aria-label="Service status">
                        <div class="service-chip">
                            <span class="service-dot core" aria-hidden="true"></span>
                            <span><strong>Core Service</strong><br /><em>Online</em></span>
                        </div>
                        <div class="service-chip">
                            <span class="service-dot ml" aria-hidden="true"></span>
                            <span><strong>ML Service</strong><br /><em>Optional / Available</em></span>
                        </div>
                        <button class="header-icon-button" type="button" aria-label="Help">?</button>
                        <button class="header-icon-button user" type="button" aria-label="User profile">○</button>
                        <button class="header-chevron" type="button" aria-label="Open user menu">⌄</button>
                    </div>
                </header>
                <main class="app-main console-main">
                    <router-outlet />
                </main>
            </section>
        </div>
    `,
})
export class CoreShellComponent {
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);
    private readonly currentUrl = signal(this.router.url);
    protected readonly pageTitle = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/nist')) {
            return 'NIST Data Fetch';
        }
        if (url.startsWith('/fitting')) {
            return 'Fitting';
        }
        if (url.startsWith('/training')) {
            return 'Training';
        }
        return 'User Data';
    });
    protected readonly pageDescription = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/nist')) {
            return 'Fetch and enrich NIST adsorption source data.';
        }
        if (url.startsWith('/fitting')) {
            return 'Configure and run adsorption model fitting workflows.';
        }
        if (url.startsWith('/training')) {
            return 'Prepare datasets, checkpoints, and model training workflows.';
        }
        return 'Upload, preview, edit, and manage user datasets.';
    });

    constructor() {
        this.router.events
            .pipe(
                filter((event): event is NavigationEnd => event instanceof NavigationEnd),
                takeUntilDestroyed(this.destroyRef)
            )
            .subscribe((event) => this.currentUrl.set(event.urlAfterRedirects));
    }
}
