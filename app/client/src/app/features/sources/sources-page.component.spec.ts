// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { COLLAPSED_GROUPS_KEY, SOURCE_LOGOS, SourcesPageComponent } from './sources-page.component';

interface Internals {
    isCollapsed(id: string): boolean;
    toggleGroup(id: string): void;
    logoFor(key: string): { src: string; alt: string } | null;
    logoFailed(key: string): void;
}

const create = (): Internals => {
    TestBed.configureTestingModule({ providers: [provideRouter([])] });
    return TestBed.createComponent(SourcesPageComponent).componentInstance as unknown as Internals;
};

describe('SourcesPageComponent catalog', () => {
    beforeEach(() => {
        localStorage.clear();
        vi.stubGlobal('fetch', vi.fn(() => new Promise(() => undefined)));
    });
    afterEach(() => vi.unstubAllGlobals());

    it('starts expanded, persists collapsed groups, and restores them', () => {
        const page = create();
        expect(page.isCollapsed('adsorption')).toBe(false);
        page.toggleGroup('adsorption');
        expect(page.isCollapsed('adsorption')).toBe(true);
        expect(JSON.parse(localStorage.getItem(COLLAPSED_GROUPS_KEY) ?? '[]')).toEqual(['adsorption']);

        TestBed.resetTestingModule();
        expect(create().isCollapsed('adsorption')).toBe(true);
    });

    it('falls back to the monogram for unknown or failed logos', () => {
        const page = create();
        expect(page.logoFor('unknown')).toBeNull();
        expect(page.logoFor('nist')).toEqual(SOURCE_LOGOS['nist']);
        page.logoFailed('nist');
        expect(page.logoFor('nist')).toBeNull();
    });
});
