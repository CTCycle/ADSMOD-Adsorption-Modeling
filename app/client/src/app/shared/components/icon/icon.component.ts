// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, input } from '@angular/core';

export type IconName = 'upload' | 'eye' | 'trash' | 'refresh' | 'file' | 'database' | 'plus' | 'chevron' | 'filter';

/** Stroke paths on a 24px grid, shared so icon-only controls stay visually consistent. */
const PATHS: Record<IconName, string> = {
    upload: 'M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3',
    eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z',
    trash: 'M4 7h16M10 11v6M14 11v6M6 7l1 12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-12M9 7V4h6v3',
    refresh: 'M20 11a8 8 0 0 0-14.5-3.5L4 9M4 4v5h5M4 13a8 8 0 0 0 14.5 3.5L20 15M20 20v-5h-5',
    file: 'M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7l-4-4zM14 3v4h4M9 13h6M9 17h6',
    plus: 'M12 5v14M5 12h14',
    chevron: 'M6 9l6 6 6-6',
    filter: 'M4 5h16l-6 7.5V19l-4 1v-7.5L4 5z',
    database: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
};

@Component({
    selector: 'adsmod-icon',
    standalone: true,
    template: `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path [attr.d]="path()" />
        </svg>
    `,
    styles: [':host { display: inline-flex; flex: none; } svg { display: block; }'],
})
export class IconComponent {
    readonly name = input.required<IconName>();
    protected path(): string {
        return PATHS[this.name()];
    }
}
