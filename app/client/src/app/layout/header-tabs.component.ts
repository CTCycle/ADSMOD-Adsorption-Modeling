import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

@Component({
    selector: 'adsmod-header-tabs',
    standalone: true,
    imports: [RouterLink, RouterLinkActive],
    template: `
        <nav class="header-tabs header-tabs-embedded" aria-label="Primary">
            <a class="header-tab" routerLink="/datasets" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }">
                <span class="header-tab-label">Custom Datasets</span>
            </a>
            <a class="header-tab" routerLink="/public-data" routerLinkActive="active">
                <span class="header-tab-label">Public Data</span>
            </a>
            <a class="header-tab" routerLink="/public-materials" routerLinkActive="active">
                <span class="header-tab-label">Materials &amp; Adsorbates</span>
            </a>
            <a class="header-tab" routerLink="/fitting" routerLinkActive="active">
                <span class="header-tab-label">Fitting</span>
            </a>
            <a class="header-tab" routerLink="/training" routerLinkActive="active">
                <span class="header-tab-label">Training</span>
            </a>
        </nav>
    `,
})
export class HeaderTabsComponent {}
