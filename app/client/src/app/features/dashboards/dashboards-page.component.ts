// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
    selector: 'adsmod-dashboards-page',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet],
    styleUrl: './dashboards.css',
    template: `
        <div class="dashboards-workspace">
            <nav class="dashboard-tabs" aria-label="Dashboard views">
                <a routerLink="overview" routerLinkActive="active" ariaCurrentWhenActive="page">Overview</a>
                <a routerLink="explore" routerLinkActive="active" ariaCurrentWhenActive="page">Explore &amp; compare</a>
            </nav>
            <router-outlet />
        </div>
    `,
})
export class DashboardsPageComponent {}
