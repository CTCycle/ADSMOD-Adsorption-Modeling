// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { JobsMonitorService } from '../../services/jobs-monitor.service';
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

const job = (overrides: Record<string, unknown>): Record<string, unknown> => {
    const base = run(overrides);
    return {
        id: base['id'],
        kind: 'acquisition',
        name: base['operation'],
        source: base['source_key'],
        status: base['status'],
        progress: base['progress'],
        created_at: base['created_at'],
        finished_at: null,
        saved_count: base['imported_count'],
        skipped_count: 0,
        failed_count: 0,
        error: null,
        run: base,
    };
};

describe('AcquireViewComponent', () => {
    const fetchMock = vi.fn();
    let runs: Record<string, unknown>[] = [];
    let collectionRequests = 0;
    let availableCount: number | null = 12;
    let rejectSyncStart = false;

    const syncBodies = (): Record<string, unknown>[] =>
        fetchMock.mock.calls
            .filter(([url, init]) => String(url).endsWith('/sync-runs') && init?.method === 'POST')
            .map(([, init]) => JSON.parse(String(init.body)) as Record<string, unknown>);

    const render = async (): Promise<{ fixture: ReturnType<typeof TestBed.createComponent<AcquireViewComponent>>; element: HTMLElement }> => {
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.advanceTimersByTimeAsync(0);
        fixture.detectChanges();
        return { fixture, element: fixture.nativeElement as HTMLElement };
    };

    const deepLink = (params: Record<string, string>): void => {
        TestBed.overrideProvider(ActivatedRoute, { useValue: { snapshot: { queryParamMap: convertToParamMap(params) } } });
    };

    beforeEach(async () => {
        vi.useFakeTimers();
        collectionRequests = 0;
        availableCount = 12;
        rejectSyncStart = false;
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
            if (url.includes('/structures/search')) {
                return { ok: true, json: async () => ({ items: [{ cod_id: '1000000', name: 'ZIF-8', formula: 'C8H10N4Zn', space_group: 'I-43m', year: 2006, doi: null, source_url: 'https://example.test/1000000' }] }) };
            }
            if (url.endsWith('/sync-runs') && init?.method === 'POST' && rejectSyncStart) {
                return { ok: false, status: 503, json: async () => ({ detail: 'Provider unavailable' }) };
            }
            if (url.endsWith('/sync-runs') && init?.method === 'POST') {
                return { ok: true, json: async () => run({ id: 'index-1', source_key: 'nist', operation: 'nist_index', parameters: { category: 'guest' }, status: 'running', progress: 0, imported_count: 0 }) };
            }
            if (url.includes('/jobs')) {
                const active = runs.filter((item) => ['queued', 'running'].includes(String(item['status']))).length;
                return { ok: true, json: async () => ({ jobs: runs.map((item) => job(item)), active_count: active }) };
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

    it('uses the same Single / Bulk switch, in that order, on every acquisition card', async () => {
        runs = [];
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.advanceTimersByTimeAsync(0);
        fixture.detectChanges();

        const switches = Array.from((fixture.nativeElement as HTMLElement).querySelectorAll('adsmod-segmented-control'));
        expect(switches).toHaveLength(3);
        for (const control of switches) {
            expect(Array.from(control.querySelectorAll('button')).map((button) => button.textContent?.trim())).toEqual(['Single', 'Bulk']);
        }
    });

    it('refreshes NIST counts once a running collection fetch finishes', async () => {
        runs = [run({ source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, status: 'running', progress: 40 })];
        const jobs = TestBed.inject(JobsMonitorService);
        const fixture = TestBed.createComponent(AcquireViewComponent);
        fixture.detectChanges();
        await vi.advanceTimersByTimeAsync(0);
        await vi.waitFor(() => expect(jobs.activeCount()).toBe(1));
        expect(collectionRequests).toBe(1);

        runs = [run({ source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, imported_count: 3 })];
        await jobs.refresh();
        await vi.advanceTimersByTimeAsync(0);
        fixture.detectChanges();

        expect(collectionRequests).toBe(2);
        const element = fixture.nativeElement as HTMLElement;
        expect(element.querySelector('.collection-count strong')?.textContent).toContain('6 / 12');
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

        (element.querySelector('.availability-action') as HTMLButtonElement).click();
        await vi.advanceTimersByTimeAsync(3000);
        fixture.detectChanges();

        expect(cell()).toContain('/ 40 records');
        expect(syncBodies().map((body) => body['category'])).toEqual(['guest', 'host', 'experiments']);
    });

    it('runs the search a Datasets deep link prefilled and shows the link target first', async () => {
        runs = [];
        deepLink({ material_id: '8', material_name: 'ZIF-8' });
        const { element } = await render();

        expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/structures/search?q=ZIF-8'))).toBe(true);
        expect(element.querySelector('.target-chip')?.textContent).toContain('ZIF-8');
        expect(element.querySelector('.cod-results-table')).not.toBeNull();
    });

    it('keeps the COD selection when the import request fails', async () => {
        runs = [];
        rejectSyncStart = true;
        deepLink({ material_name: 'ZIF-8' });
        const { fixture, element } = await render();

        (element.querySelector('.cod-results-table tbody input') as HTMLInputElement).click();
        fixture.detectChanges();
        (element.querySelector('.cod-submit .button') as HTMLButtonElement).click();
        await vi.advanceTimersByTimeAsync(0);
        fixture.detectChanges();

        expect(element.querySelector('.public-alert')).not.toBeNull();
        expect(element.querySelector('.cod-submit')?.textContent).toContain('1 selected');
    });

    it('submits a percentage share as a fraction, previews it, and blocks invalid values', async () => {
        runs = [];
        const { fixture, element } = await render();
        const row = element.querySelector('.collection-row') as HTMLElement;
        const input = row.querySelector('input') as HTMLInputElement;
        const fetchButton = row.querySelector('.button.primary') as HTMLButtonElement;
        const setShare = (value: string): void => {
            input.value = value;
            input.dispatchEvent(new Event('input'));
            fixture.detectChanges();
        };

        setShare('150');
        expect(fetchButton.disabled).toBe(true);
        expect(row.querySelector('.share-preview')?.textContent).toContain('Enter 1 to 100');

        setShare('25');
        expect(fetchButton.disabled).toBe(false);
        expect(row.querySelector('.share-preview')?.textContent).toContain('First 3 of 12');

        fetchButton.click();
        await vi.advanceTimersByTimeAsync(0);
        expect(syncBodies()[0]).toMatchObject({ category: 'guest', fraction: 0.25 });
    });

    it('shows only runs from this visit on the card, with a link to the result', async () => {
        const finished = { id: 'run-2', source_key: 'nist', operation: 'nist_category', parameters: { category: 'guest', fraction: 0.1 }, imported_count: 3 };
        runs = [run({ ...finished, id: 'old-run' })];
        const jobs = TestBed.inject(JobsMonitorService);
        const { fixture, element } = await render();
        await vi.waitFor(() => expect(jobs.jobs()).toHaveLength(1));
        fixture.detectChanges();
        const activity = (): string => element.querySelector('adsmod-acquire-activity')?.textContent ?? '';
        expect(activity()).toBe('');

        runs = [run({ ...finished, status: 'running', progress: 40 })];
        await jobs.refresh();
        fixture.detectChanges();
        expect(activity()).toContain('40%');

        runs = [run(finished)];
        await jobs.refresh();
        await vi.advanceTimersByTimeAsync(0);
        fixture.detectChanges();
        expect(activity()).toContain('3 saved');
        expect(activity()).toContain('View species');
    });
});
