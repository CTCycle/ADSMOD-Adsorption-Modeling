// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

/** Shown for unknown addresses instead of silently redirecting to Home. */
@Component({
    selector: 'adsmod-not-found-page',
    standalone: true,
    imports: [RouterLink],
    template: `
        <section class="not-found" aria-labelledby="not-found-title">
            <h2 id="not-found-title">Page not found</h2>
            <p>This address does not match a page in ADSMOD. Check the link, or go back to <a routerLink="/home">Home</a>.</p>
        </section>
    `,
    styles: `
        .not-found { display: grid; gap: 0.5rem; max-width: 36rem; padding: var(--spacing-lg); }
        .not-found h2 { margin: 0; }
        .not-found p { margin: 0; color: var(--text-secondary); }
    `,
})
export class NotFoundPageComponent {}
