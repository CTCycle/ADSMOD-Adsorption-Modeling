// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, OnInit, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { machineLearningAvailable } from '../../services/system.service';

/** Modeling workspace shell: curve fitting sessions, deep learning, and the adsorption model catalog. */
@Component({
    selector: 'adsmod-modeling-page',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, RouterOutlet],
    template: `
        <div class="modeling-workspace">
            <nav class="modeling-tabs" aria-label="Modeling sections">
                <a routerLink="/modeling/fitting" routerLinkActive="active" ariaCurrentWhenActive="page">Curve Fitting</a>
                @if (deepLearningAvailable()) {
                    <a routerLink="/modeling/deep-learning" routerLinkActive="active" ariaCurrentWhenActive="page">Deep Learning</a>
                }
                <a routerLink="/modeling/models" routerLinkActive="active" ariaCurrentWhenActive="page">Adsorption Models</a>
            </nav>
            <router-outlet />
        </div>
    `,
    styles: [`
        :host { display: block; min-width: 0; }
        .modeling-workspace { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--spacing-md); min-width: 0; padding-bottom: var(--spacing-lg); }
        .modeling-tabs { display: flex; flex-wrap: wrap; gap: 0 4px; border-bottom: 1px solid var(--border-color); scrollbar-width: thin; }
        .modeling-tabs a {
            flex: 0 0 auto;
            padding: 10px 14px;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 600;
            text-decoration: none;
            white-space: nowrap;
        }
        @media (max-width: 640px) { .modeling-tabs a { padding: 10px; font-size: 0.82rem; } }
        .modeling-tabs a:hover, .modeling-tabs a.active { color: var(--text-primary); border-bottom-color: var(--primary-600); }
        .modeling-tabs a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }
    `],
})
export class ModelingPageComponent implements OnInit {
    protected readonly deepLearningAvailable = signal(false);

    async ngOnInit(): Promise<void> {
        this.deepLearningAvailable.set(await machineLearningAvailable());
    }
}
