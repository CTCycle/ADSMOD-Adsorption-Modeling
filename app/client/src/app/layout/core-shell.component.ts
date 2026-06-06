import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HeaderTabsComponent } from './header-tabs.component';

@Component({
    selector: 'adsmod-core-shell',
    standalone: true,
    imports: [HeaderTabsComponent, RouterOutlet],
    template: `
        <div class="app-container">
            <header class="app-header">
                <div class="header-content">
                    <div class="header-brand">
                        <div class="brand-logo" aria-hidden="true">AD</div>
                        <div class="brand-wordmark">
                            <span>ADSMOD</span>
                        </div>
                    </div>
                    <adsmod-header-tabs />
                </div>
            </header>
            <main class="app-main">
                <router-outlet />
            </main>
        </div>
    `,
})
export class CoreShellComponent {}
