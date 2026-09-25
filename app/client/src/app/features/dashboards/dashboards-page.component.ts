import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
    selector: 'adsmod-dashboards-page',
    standalone: true,
    imports: [RouterLink],
    template: `
        <section class="console-card dashboards-placeholder" aria-labelledby="dashboards-empty-title">
            <div class="dashboards-placeholder-content">
                <h2 id="dashboards-empty-title">No dashboards yet</h2>
                <p>Workspace dashboard views will appear here as activity and results become available.</p>
                <div class="dashboard-placeholder-actions">
                    <a class="button secondary" routerLink="/datasets">Open Custom Datasets</a>
                    <a class="button secondary" routerLink="/public-data/overview">Explore Public Data</a>
                </div>
            </div>
        </section>
    `,
})
export class DashboardsPageComponent {}
