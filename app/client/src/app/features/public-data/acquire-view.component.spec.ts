// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AcquireViewComponent } from './acquire-view.component';

const run = (overrides: Record<string, unknown>): Record<string, unknown> => ({
    id: 'run-1',
    source_key: 'cod',
    operation: 'cod_import',
    parameters: { cod_id: '1000000' },
    status: 'completed',
    progress: 100,
    processed_count: 1,
    imported_count: 1,
    skipped_count: 0,
    failed_count: 0,
    checkpoint: { entity_id: 9, entity_ids: [9] },
    cancel_requested: false,
    error_detail: null,
    created_at: '2026-10-20T08:00:00Z',
    started_at: null,
    finished_at: null,
    ...overrides,
});

describe('AcquireViewComponent', () => {
    const fetchMock = vi.fn();
    let runs: Record<string, unknown>[] = [];
    let collectionRequests = 0;

    beforeEach(async () => {
        vi.useFakeTimers();
        collectionRequests = 0;
        fetchMock.mockReset();
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.includes('/sources/nist/collections')) {
                collectionRequests += 1;
                return {
                    ok: true,
                    json: async () => ({
                        collections: [{ category: 'guest', local_count: collectionRequests * 3, available_count: 12, indexed_at: null }],
                    }),
                };
            }
            if (url.includes('/sync-runs')) {
                return { ok: true, json: async () => ({ runs, total: runs.length }) };
            }
            return { ok: false, status: 404, json: async () => ({}) };
        });
        vi.stubGlobal('fetch', fetchMock);
        TestBed.resetTestingModule();
        await TestBed.configureTestingModule({
            imports: [AcquireViewComponent],
            providers: [provideRouter([])],
        }).compileComponents();
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
    });

    async function render(): Promise<HTMLElement> {
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.runOnlyPendingTimersAsync();
        await fixture.whenStable();
        fixture.detectChanges();
        return fixture.nativeElement as HTMLElement;
    }

    it('links a finished import straight to the stored record', async () => {
        runs = [run({})];

        const element = await render();

        const link = element.querySelector('#acquire-runs a.text-link') as HTMLAnchorElement;
        expect(link.textContent).toContain('Open structure');
        expect(link.getAttribute('href')).toBe('/public-data/structures?id=9');
    });

    it('refreshes NIST counts once a running collection fetch finishes', async () => {
        runs = [run({ source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, status: 'running', progress: 40 })];
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.runOnlyPendingTimersAsync();
        await fixture.whenStable();
        expect(collectionRequests).toBe(1);

        runs = [run({ source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, imported_count: 3 })];
        await vi.advanceTimersByTimeAsync(1600);
        await fixture.whenStable();
        fixture.detectChanges();

        expect(collectionRequests).toBe(2);
        const element = fixture.nativeElement as HTMLElement;
        expect(element.querySelector('.collection-count strong')?.textContent).toContain('6 / 12');
        expect(element.querySelector('#acquire-runs a.text-link')?.getAttribute('href')).toBe('/public-data/chemicals?source=nist');
    });
});
