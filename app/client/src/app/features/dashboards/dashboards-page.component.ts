import { Component } from '@angular/core';

@Component({
    selector: 'adsmod-dashboards-page',
    standalone: true,
    template: `
        <section class="console-card dashboards-placeholder" aria-labelledby="dashboards-title">
            <div class="dashboards-placeholder-content">
                <p class="eyebrow">Workspace overview</p>
                <h2 id="dashboards-title">Dashboards</h2>
                <p>Dashboard views will appear here as the workspace grows.</p>
            </div>
        </section>
    `,
})
export class DashboardsPageComponent {}
