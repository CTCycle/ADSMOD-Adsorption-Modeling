// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Injectable, computed, signal } from '@angular/core';
import { Subject } from 'rxjs';
import { API_BASE_URL } from '../core/config/api-base-url';
import type { UnifiedJob, UnifiedJobListResponse } from '../models/job.model';
import { isActiveStatus } from '../shared/utils/job-presentation';
import { extractErrorMessage, fetchWithTimeout } from './http-timeout.service';
import { toErrorMessage } from '../shared/utils/error-message';

const PAGE_SIZE = 20;
const MAX_JOBS = 100;
const ACTIVE_POLL_MS = 1500;
const IDLE_POLL_MS = 10_000;
const RETRY_MIN_MS = 3000;
const RETRY_MAX_MS = 15_000;
const REQUEST_TIMEOUT_MS = 15_000;

/** Application-wide view of background jobs; polls only while something watches it. */
@Injectable({ providedIn: 'root' })
export class JobsMonitorService {
    readonly jobs = signal<UnifiedJob[]>([]);
    readonly activeCount = signal(0);
    readonly error = signal<string | null>(null);
    readonly popoverOpen = signal(false);
    readonly canLoadMore = signal(false);
    /** Acquisition runs that are queued or running. */
    readonly activeRuns = computed(() =>
        this.jobs().flatMap((job) => (job.run !== null && isActiveStatus(job.status) ? [job.run] : [])),
    );
    /** Emits jobs that were active on the previous refresh and have now ended. */
    readonly finished = new Subject<UnifiedJob[]>();

    private limit = PAGE_SIZE;
    private watchers = 0;
    private timer: number | null = null;
    private retryDelay = 0;
    private inFlight: Promise<void> | null = null;

    /** Start periodic refreshes until the returned function is called. */
    watch(): () => void {
        this.watchers += 1;
        void this.refresh();
        let stopped = false;
        return () => {
            if (stopped) return;
            stopped = true;
            this.watchers -= 1;
            if (this.watchers === 0 && this.timer !== null) {
                window.clearTimeout(this.timer);
                this.timer = null;
            }
        };
    }

    refresh(): Promise<void> {
        this.inFlight ??= this.load().finally(() => {
            this.inFlight = null;
            this.schedule();
        });
        return this.inFlight;
    }

    loadMore(): Promise<void> {
        this.limit = Math.min(MAX_JOBS, this.limit + PAGE_SIZE);
        return this.refresh();
    }

    private async load(): Promise<void> {
        try {
            const response = await fetchWithTimeout(`${API_BASE_URL}/jobs?limit=${this.limit}`, { method: 'GET' }, REQUEST_TIMEOUT_MS);
            const body = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(extractErrorMessage(response, body));
            const payload = body as UnifiedJobListResponse;
            this.apply(payload);
            this.error.set(null);
            this.retryDelay = 0;
        } catch (error) {
            this.error.set(toErrorMessage(error, 'Jobs are temporarily unavailable.'));
            this.retryDelay = this.retryDelay ? Math.min(RETRY_MAX_MS, this.retryDelay * 2) : RETRY_MIN_MS;
        }
    }

    private apply(payload: UnifiedJobListResponse): void {
        const wasActive = new Set(this.jobs().filter((job) => isActiveStatus(job.status)).map((job) => job.id));
        const ended = payload.jobs.filter((job) => wasActive.has(job.id) && !isActiveStatus(job.status));
        this.jobs.set(payload.jobs);
        this.activeCount.set(payload.active_count);
        this.canLoadMore.set(this.limit < MAX_JOBS && payload.jobs.filter((job) => !isActiveStatus(job.status)).length >= this.limit);
        if (ended.length) this.finished.next(ended);
    }

    private schedule(): void {
        if (this.timer !== null) window.clearTimeout(this.timer);
        this.timer = null;
        if (this.watchers === 0) return;
        const delay = this.retryDelay || (this.activeCount() > 0 ? ACTIVE_POLL_MS : IDLE_POLL_MS);
        this.timer = window.setTimeout(() => {
            this.timer = null;
            void this.refresh();
        }, delay);
    }
}
