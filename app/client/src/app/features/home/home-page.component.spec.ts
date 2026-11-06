// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HomePageComponent } from './home-page.component';

const summary = {
    datasets: 2,
    experiments: 4,
    species: 23,
    materials: 94,
    sources: [{ key: 'nist', name: 'NIST', record_count: 117 }],
    fitting_sessions: 3,
    fitting_sessions_by_status: { completed: 2, failed: 1 },
    latest_fitting_at: null,
};

describe('HomePageComponent', () => {
    beforeEach(() => {
        vi.stubGlobal(
            'fetch',
            vi.fn(async (input: RequestInfo | URL) =>
                String(input).endsWith('/home/summary')
                    ? { ok: true, status: 200, json: async () => summary }
                    : { ok: false, status: 404, json: async () => ({}) },
            ),
        );
    });
    afterEach(() => vi.unstubAllGlobals());

    it('shows persisted totals with links into the relevant sections', async () => {
        TestBed.configureTestingModule({ providers: [provideRouter([])] });
        const fixture = TestBed.createComponent(HomePageComponent);
        fixture.detectChanges();
        await vi.waitFor(() => {
            fixture.detectChanges();
            expect((fixture.nativeElement as HTMLElement).querySelector('.home-grid')).not.toBeNull();
        });

        const element = fixture.nativeElement as HTMLElement;
        const stat = (label: string) => Array.from(element.querySelectorAll('.home-stat')).find((node) => node.textContent?.includes(label));
        expect(stat('chemical species')?.querySelector('strong')?.textContent).toBe('23');
        expect(stat('chemical species')?.getAttribute('href')).toBe('/datasets?kind=species');
        expect(stat('adsorbent materials')?.getAttribute('href')).toBe('/datasets?kind=material');
        expect(stat('external sources')?.querySelector('strong')?.textContent).toBe('1');
        expect(stat('fitting sessions')?.getAttribute('href')).toBe('/modeling/fitting');
        // Without the machine-learning profile the trained-model tile is replaced by completed sessions.
        expect(stat('completed sessions')?.querySelector('strong')?.textContent).toBe('2');
    });
});
