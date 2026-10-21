// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { THEME_STORAGE_KEY, ThemeService } from './theme.service';

function mockSystemTheme(dark: boolean): void {
    vi.stubGlobal('matchMedia', () => ({ matches: dark, addEventListener: () => undefined }));
}

describe('ThemeService', () => {
    beforeEach(() => {
        localStorage.clear();
        delete document.documentElement.dataset['theme'];
    });

    afterEach(() => vi.unstubAllGlobals());

    it('follows the system theme until a preference is chosen', () => {
        mockSystemTheme(true);
        const service = TestBed.inject(ThemeService);
        TestBed.tick();

        expect(service.preference()).toBe('system');
        expect(document.documentElement.dataset['theme']).toBe('dark');
    });

    it('persists an explicit choice and clears it when returning to system', () => {
        mockSystemTheme(true);
        const service = TestBed.inject(ThemeService);

        service.setPreference('light');
        TestBed.tick();
        expect(document.documentElement.dataset['theme']).toBe('light');
        expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');

        service.setPreference('system');
        TestBed.tick();
        expect(document.documentElement.dataset['theme']).toBe('dark');
        expect(localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
    });
});
