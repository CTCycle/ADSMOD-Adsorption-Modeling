import { Component, computed, inject, signal } from '@angular/core';
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
                        <span>Source</span>
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
                    <a class="console-footer-link" href="#" aria-label="Docs">
                        <span aria-hidden="true">?</span>
                        <span>Docs</span>
                    </a>
                    <a class="console-footer-link" href="#" aria-label="Settings">
                        <span aria-hidden="true">⚙</span>
                        <span>Settings</span>
                    </a>
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
    private readonly currentUrl = signal(this.router.url);
    protected readonly pageTitle = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/fitting')) {
            return 'Fitting';
        }
        if (url.startsWith('/training')) {
            return 'Training';
        }
        return 'Source';
    });
    protected readonly pageDescription = computed(() => {
        const url = this.currentUrl();
        if (url.startsWith('/fitting')) {
            return 'Configure and run adsorption model fitting workflows.';
        }
        if (url.startsWith('/training')) {
            return 'Prepare datasets, checkpoints, and model training workflows.';
        }
        return 'Acquire and manage experimental data for adsorption modeling.';
    });

    constructor() {
        this.router.events
            .pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd))
            .subscribe((event) => this.currentUrl.set(event.urlAfterRedirects));
    }
}
