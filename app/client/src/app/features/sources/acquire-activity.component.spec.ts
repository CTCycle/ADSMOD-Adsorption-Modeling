// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { UnifiedJob } from '../../models/job.model';
import { JobsMonitorService } from '../../services/jobs-monitor.service';
import { AcquireActivityComponent } from './acquire-activity.component';

const job = (status: string, checkpoint: Record<string, unknown> = {}): UnifiedJob => ({
    id: 'run-1',
    kind: 'acquisition',
    name: 'nist_category',
    source: 'nist',
    status,
    progress: 40,
    created_at: '2026-10-20T08:00:00Z',
    finished_at: null,
    saved_count: 3,
    skipped_count: 1,
    failed_count: 0,
    error: null,
    run: {
        id: 'run-1', source_key: 'nist', operation: 'nist_category', parameters: { category: 'experiments' }, status,
        progress: 40, processed_count: 4, imported_count: 3, skipped_count: 1, failed_count: 0, checkpoint,
        cancel_requested: false, error_detail: null, created_at: '2026-10-20T08:00:00Z', started_at: null, finished_at: null,
    },
} as UnifiedJob);

describe('AcquireActivityComponent', () => {
    const fetchMock = vi.fn();

    const render = (jobs: UnifiedJob[]) => {
        TestBed.inject(JobsMonitorService).jobs.set(jobs);
        const fixture = TestBed.createComponent(AcquireActivityComponent);
        fixture.componentRef.setInput('source', 'nist');
        fixture.componentRef.setInput('sessionRunIds', new Set(['run-1']));
        fixture.detectChanges();
        return { fixture, root: fixture.nativeElement as HTMLElement };
    };

    beforeEach(async () => {
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({ ok: true, json: async () => ({}) });
        vi.stubGlobal('fetch', fetchMock);
        TestBed.resetTestingModule();
        await TestBed.configureTestingModule({ imports: [AcquireActivityComponent], providers: [provideRouter([])] }).compileComponents();
    });

    afterEach(() => vi.unstubAllGlobals());

    it('tracks an active run with progress and a working Cancel', async () => {
        const { fixture, root } = render([job('running')]);
        expect(root.querySelector('adsmod-progress-bar')).toBeTruthy();
        const cancel = Array.from(root.querySelectorAll('button')).find((item) => item.textContent?.trim() === 'Cancel');
        expect(cancel).toBeTruthy();
        cancel!.click();
        await fixture.whenStable();
        expect(String(fetchMock.mock.calls[0][0])).toContain('/public-data/sync-runs/run-1/cancel');
    });

    it('reports already-stored records separately from skipped ones', () => {
        const { root } = render([job('completed', { already_stored_count: 7 })]);
        expect(root.textContent).toContain('3 saved · 1 skipped · 0 failed');
        expect(root.textContent).toContain('7 already stored');
    });
});
