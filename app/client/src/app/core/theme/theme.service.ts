// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DOCUMENT } from '@angular/common';
import { Injectable, effect, inject, signal } from '@angular/core';

export type ThemePreference = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'adsmod-theme';
const DARK_QUERY = '(prefers-color-scheme: dark)';

function readStoredPreference(): ThemePreference {
    try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        return stored === 'light' || stored === 'dark' ? stored : 'system';
    } catch {
        return 'system';
    }
}

@Injectable({ providedIn: 'root' })
export class ThemeService {
    private readonly document = inject(DOCUMENT);
    private readonly systemQuery = typeof matchMedia === 'function' ? matchMedia(DARK_QUERY) : null;
    private readonly systemDark = signal(this.systemQuery?.matches ?? false);

    readonly preference = signal<ThemePreference>(readStoredPreference());
    readonly resolved = signal<ResolvedTheme>('light');

    constructor() {
        this.systemQuery?.addEventListener('change', (event) => this.systemDark.set(event.matches));
        effect(() => {
            const preference = this.preference();
            const resolved: ResolvedTheme = preference === 'system' ? (this.systemDark() ? 'dark' : 'light') : preference;
            this.resolved.set(resolved);
            this.document.documentElement.dataset['theme'] = resolved;
        });
    }

    setPreference(preference: ThemePreference): void {
        this.preference.set(preference);
        try {
            if (preference === 'system') {
                localStorage.removeItem(THEME_STORAGE_KEY);
            } else {
                localStorage.setItem(THEME_STORAGE_KEY, preference);
            }
        } catch {
            // Storage can be unavailable (private mode); the choice then lasts for this session only.
        }
    }
}
