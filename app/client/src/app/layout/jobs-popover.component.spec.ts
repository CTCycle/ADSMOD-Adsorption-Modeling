// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { JobsPopoverComponent } from './jobs-popover.component';

const syncRun = {
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
    started_at: '2026-10-20T08:00:01Z',
    finished_at: '2026-10-20T08:00:05Z',
};

const jobs = [
    { id: 'fit-1', kind: 'fitting', name: 'fitting', source: null, status: 'running', progress: 35, created_at: '2026-10-20T09:00:00Z', finished_at: null, saved_count: null, skipped_count: null, failed_count: null, error: null, run: null },
    { id: 'run-1', kind: 'acquisition', name: 'cod_import', source: 'cod', status: 'completed', progress: 100, created_at: '2026-10-20T08:00:00Z', finished_at: '2026-10-20T08:00:05Z', saved_count: 1, skipped_count: 0, failed_count: 0, error: null, run: syncRun },
];

describe('JobsPopoverComponent', () => {
    beforeEach(async () => {
        vi.stubGlobal(
            'fetch',
            vi.fn(async (input: RequestInfo | URL) => {
                const url = String(input);
                if (url.includes('/jobs')) return { ok: true, json: async () => ({ jobs, active_count: 1 }) };
                if (url.endsWith('/failures?limit=100&offset=0')) return { ok: true, json: async () => ({ failures: [], total: 0 }) };
                if (url.includes('/sync-runs/run-1')) return { ok: true, json: async () => syncRun };
                return { ok: false, status: 404, json: async () => ({}) };
            }),
        );
        TestBed.resetTestingModule();
        await TestBed.configureTestingModule({ imports: [JobsPopoverComponent], providers: [provideRouter([])] }).compileComponents();
    });

    afterEach(() => vi.unstubAllGlobals());

    it('shows the active count, lists every job kind, and keeps acquisition details reachable', async () => {
        const fixture = TestBed.createComponent(JobsPopoverComponent);
        const element = fixture.nativeElement as HTMLElement;
        await vi.waitFor(() => {
            fixture.detectChanges();
            expect(element.querySelector('.jobs-badge')?.textContent?.trim()).toBe('1');
        });
        element.querySelector<HTMLButtonElement>('.jobs-trigger')!.click();
        fixture.detectChanges();

        const titles = Array.from(element.querySelectorAll('.job-title')).map((node) => node.textContent?.trim());
        expect(titles).toEqual(['Model fitting', 'COD import · 1000000']);

        element.querySelectorAll<HTMLButtonElement>('.job-summary')[1].click();
        await vi.waitFor(() => {
            fixture.detectChanges();
            expect(element.querySelector('.job-details a')?.getAttribute('href')).toBe('/sources/structures?id=9');
        });
    });
});
