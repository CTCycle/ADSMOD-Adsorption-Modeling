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
    let availableCount: number | null = 12;

    beforeEach(async () => {
        vi.useFakeTimers();
        collectionRequests = 0;
        availableCount = 12;
        fetchMock.mockReset();
        fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
            const url = String(input);
            if (url.includes('/sources/nist/collections')) {
                collectionRequests += 1;
                return {
                    ok: true,
                    json: async () => ({
                        collections: [{ category: 'guest', local_count: collectionRequests * 3, available_count: availableCount, indexed_at: availableCount === null ? null : '2026-10-20T08:00:00Z' }],
                    }),
                };
            }
            if (url.includes('/sync-runs/index-1')) {
                availableCount = 40;
                return { ok: true, json: async () => run({ id: 'index-1', source_key: 'nist', operation: 'nist_index', parameters: { category: 'guest' }, imported_count: 0 }) };
            }
            if (url.endsWith('/sync-runs') && init?.method === 'POST') {
                return { ok: true, json: async () => run({ id: 'index-1', source_key: 'nist', operation: 'nist_index', parameters: { category: 'guest' }, status: 'running', progress: 0, imported_count: 0 }) };
            }
            if (url.includes('/sync-runs')) {
                const scope = new URL(url, 'http://localhost').searchParams.get('scope');
                const scopedRuns = runs.filter((item) => {
                    const status = String(item['status']);
                    if (scope === 'active') return ['queued', 'running'].includes(status);
                    if (scope === 'history') return !['queued', 'running'].includes(status);
                    return true;
                });
                return { ok: true, json: async () => ({ runs: scopedRuns, total: scopedRuns.length }) };
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
        await vi.advanceTimersByTimeAsync(0);
        await vi.waitFor(() => {
            fixture.detectChanges();
            expect(collectionRequests).toBeGreaterThan(0);
        });
        fixture.detectChanges();
        return fixture.nativeElement as HTMLElement;
    }

    it('links a finished import straight to the stored record', async () => {
        runs = [run({})];

        const element = await render();

        const link = element.querySelector('#acquire-runs a.table-action') as HTMLAnchorElement;
        expect(link.textContent).toContain('Open structure');
        expect(link.getAttribute('href')).toBe('/sources/structures?id=9');
    });

    it('refreshes NIST counts once a running collection fetch finishes', async () => {
        runs = [run({ source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, status: 'running', progress: 40 })];
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.advanceTimersByTimeAsync(0);
        await vi.waitFor(() => expect(collectionRequests).toBe(1));
        expect(collectionRequests).toBe(1);

        runs = [run({ source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, imported_count: 3 })];
        await vi.advanceTimersByTimeAsync(1600);
        fixture.detectChanges();

        expect(collectionRequests).toBe(2);
        const element = fixture.nativeElement as HTMLElement;
        expect(element.querySelector('.collection-count strong')?.textContent).toContain('6 / 12');
        expect(element.querySelector('#acquire-runs a.table-action')?.getAttribute('href')).toBe('/datasets');
    });

    it('reports unchecked availability distinctly and shows stored versus available after a check', async () => {
        availableCount = null;
        runs = [];
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.advanceTimersByTimeAsync(0);
        await vi.waitFor(() => expect(collectionRequests).toBeGreaterThan(0));
        fixture.detectChanges();
        const element = fixture.nativeElement as HTMLElement;
        const cell = (): string => element.querySelector('.collection-count')?.textContent ?? '';
        expect(cell()).toContain('3 stored');
        expect(cell()).toContain('Availability not checked');
        expect(cell()).not.toContain('/');

        (element.querySelector('.collection-row .table-action') as HTMLButtonElement).click();
        await vi.advanceTimersByTimeAsync(1000);
        fixture.detectChanges();

        expect(cell()).toContain('/ 40 records');
        expect(element.querySelector('.acquisition-history-table .data-table-empty')).not.toBeNull();
    });
    it('keeps polling while closed and restores page scrolling on close and destruction', async () => {
        runs = [run({ status: 'running' })];
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.advanceTimersByTimeAsync(0);
        fixture.detectChanges();
        const element = fixture.nativeElement as HTMLElement;
        const dialog = element.querySelector('dialog')!;
        dialog.showModal = vi.fn(() => dialog.setAttribute('open', ''));
        dialog.close = vi.fn(() => dialog.removeAttribute('open'));
        const trigger = element.querySelector<HTMLButtonElement>('[data-acquisitions-trigger]')!;
        expect(dialog.open).toBe(false);
        expect(trigger.textContent).toContain('1 active');
        runs = [run({})];
        await vi.advanceTimersByTimeAsync(1600);
        fixture.detectChanges();
        expect(dialog.open).toBe(false);
        expect(trigger.textContent).toContain('0 active');
        trigger.click();
        fixture.detectChanges();
        expect(dialog.open).toBe(true);
        expect(document.body.style.overflow).toBe('hidden');
        element.querySelector<HTMLButtonElement>('[aria-label="Close acquisitions"]')!.click();
        fixture.detectChanges();
        expect(dialog.open).toBe(false);
        expect(document.body.style.overflow).not.toBe('hidden');
        trigger.click();
        fixture.detectChanges();
        expect(dialog.open).toBe(true);
        expect(element.querySelector('a.table-action')?.getAttribute('href')).toBe('/sources/structures?id=9');
        fixture.destroy();
        expect(document.body.style.overflow).not.toBe('hidden');
    });

});
