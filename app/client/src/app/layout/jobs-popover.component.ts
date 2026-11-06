// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, ElementRef, HostListener, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { UnifiedJob } from '../models/job.model';
import type { PublicDataSyncFailure, PublicDataSyncRun } from '../models/public-data.model';
import { JobsMonitorService } from '../services/jobs-monitor.service';
import {
    cancelPublicDataSyncRun,
    fetchPublicDataSyncFailures,
    fetchPublicDataSyncRun,
    resumePublicDataSyncRun,
} from '../services/public-data.service';
import {
    formatDate,
    formatPercent,
    formatShort,
    importedLink,
    isActiveStatus,
    jobTitle,
    pendingCount,
    statusLabel,
} from '../shared/utils/job-presentation';

@Component({
    selector: 'adsmod-jobs-popover',
    standalone: true,
    imports: [RouterLink],
    styleUrl: './jobs-popover.component.css',
    template: `
        <button class="jobs-trigger" type="button" aria-haspopup="dialog" aria-controls="jobs-panel"
            [attr.aria-expanded]="jobs.popoverOpen()" (click)="toggle()">
            Jobs
            @if (jobs.activeCount() > 0) {
                <span class="jobs-badge" [attr.aria-label]="jobs.activeCount() + ' active'">{{ jobs.activeCount() }}</span>
            }
        </button>
        @if (jobs.popoverOpen()) {
            <section id="jobs-panel" class="jobs-panel" role="dialog" aria-label="Background jobs">
                <header class="jobs-head">
                    <strong>Jobs</strong>
                    <span class="jobs-summary">{{ jobs.activeCount() }} active</span>
                    <button class="jobs-text-button" type="button" (click)="refresh()">Refresh</button>
                    <button class="jobs-close" type="button" aria-label="Close jobs" (click)="close()">×</button>
                </header>
                @if (jobs.error()) {
                    <p class="jobs-notice" role="status">Live updates paused; retrying automatically.</p>
                }
                @if (actionError()) {
                    <p class="jobs-notice error" role="alert">{{ actionError() }}</p>
                }
                <div class="jobs-list">
                    @for (job of jobs.jobs(); track job.id) {
                        <article class="job-row" [class.expanded]="expandedId() === job.id">
                            <button class="job-summary" type="button" [attr.aria-expanded]="expandedId() === job.id" (click)="toggleDetails(job)">
                                <span class="job-title" [title]="title(job)">{{ title(job) }}</span>
                                <span class="job-status" [attr.data-status]="job.status">{{ label(job) }}</span>
                            </button>
                            <div class="job-meta">
                                <progress max="100" [value]="job.progress" [attr.aria-label]="title(job) + ' progress'"></progress>
                                <span class="job-percent">{{ percent(job.progress) }}</span>
                                @if (job.saved_count !== null) {
                                    <span class="job-counts">{{ job.saved_count }} saved · {{ job.skipped_count }} skipped · {{ job.failed_count }} failed</span>
                                }
                                <span class="job-time">{{ short(job.finished_at ?? job.created_at) }}</span>
                            </div>
                            @if (expandedId() === job.id) {
                                <div class="job-details">
                                    <dl class="job-facts">
                                        <div><dt>Type</dt><dd>{{ job.kind }}{{ job.source ? ' · ' + job.source : '' }}</dd></div>
                                        <div><dt>Created</dt><dd>{{ date(job.created_at) }}</dd></div>
                                        <div><dt>Started</dt><dd>{{ date(detailRun()?.started_at ?? null) }}</dd></div>
                                        <div><dt>Finished</dt><dd>{{ date(job.finished_at) }}</dd></div>
                                        @if (detailRun(); as run) {
                                            <div><dt>Processed</dt><dd>{{ run.processed_count }}</dd></div>
                                        }
                                    </dl>
                                    @if (pending(); as remaining) {
                                        <p class="job-note">{{ remaining }} records remain; run the operation again to continue.</p>
                                    }
                                    @if (job.error) {
                                        <p class="job-error" role="alert">{{ job.error }}</p>
                                    }
                                    @if (failuresLoading()) {
                                        <p class="job-note" role="status">Loading failed records…</p>
                                    } @else if (failures().length) {
                                        <p class="job-note"><strong>Failed records</strong> · {{ failuresTotal() }} total</p>
                                        <ul class="job-failures">
                                            @for (failure of failures(); track failure.id) {
                                                <li><strong>{{ failure.external_id || 'Record' }}</strong><span>{{ failure.reason }}</span></li>
                                            }
                                        </ul>
                                    } @else if (job.failed_count) {
                                        <p class="job-note">No record-level failure details were returned for this run.</p>
                                    }
                                    <div class="job-actions">
                                        @if (detailRun(); as run) {
                                            @if (link(run); as target) {
                                                <a class="jobs-text-button" [routerLink]="target.path" [queryParams]="target.queryParams" (click)="close()">{{ target.label }}</a>
                                            }
                                            @if (isActive(job) && !run.cancel_requested) {
                                                <button class="jobs-text-button danger" type="button" (click)="cancel(run.id)">Cancel</button>
                                            }
                                            @if (run.status === 'interrupted') {
                                                <button class="jobs-text-button" type="button" (click)="resume(run.id)">Resume</button>
                                            }
                                        }
                                    </div>
                                </div>
                            }
                        </article>
                    } @empty {
                        <p class="jobs-empty">No jobs yet. Acquisitions, model fitting and training runs appear here.</p>
                    }
                </div>
                @if (jobs.canLoadMore()) {
                    <button class="jobs-more" type="button" (click)="loadMore()">Show older jobs</button>
                }
            </section>
        }
    `,
})
export class JobsPopoverComponent {
    protected readonly jobs = inject(JobsMonitorService);
    private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
    protected readonly expandedId = signal<string | null>(null);
    protected readonly detailRun = signal<PublicDataSyncRun | null>(null);
    protected readonly failures = signal<PublicDataSyncFailure[]>([]);
    protected readonly failuresTotal = signal(0);
    protected readonly failuresLoading = signal(false);
    protected readonly actionError = signal<string | null>(null);

    protected readonly title = jobTitle;
    protected readonly label = statusLabel;
    protected readonly percent = formatPercent;
    protected readonly short = formatShort;
    protected readonly date = formatDate;
    protected readonly link = importedLink;

    constructor() {
        const stop = this.jobs.watch();
        inject(DestroyRef).onDestroy(stop);
    }

    protected isActive(job: UnifiedJob): boolean {
        return isActiveStatus(job.status);
    }

    protected pending(): number {
        const run = this.detailRun();
        return run ? pendingCount(run) : 0;
    }

    protected toggle(): void {
        if (this.jobs.popoverOpen()) {
            this.close();
            return;
        }
        this.jobs.popoverOpen.set(true);
        void this.jobs.refresh();
    }

    protected close(): void {
        this.jobs.popoverOpen.set(false);
        this.collapse();
        this.actionError.set(null);
    }

    protected refresh(): void {
        void this.jobs.refresh();
    }

    protected loadMore(): void {
        void this.jobs.loadMore();
    }

    @HostListener('document:keydown.escape')
    protected onEscape(): void {
        if (this.jobs.popoverOpen()) this.close();
    }

    @HostListener('document:click', ['$event'])
    protected onDocumentClick(event: MouseEvent): void {
        if (this.jobs.popoverOpen() && !this.host.nativeElement.contains(event.target as Node)) this.close();
    }

    protected async toggleDetails(job: UnifiedJob): Promise<void> {
        if (this.expandedId() === job.id) {
            this.collapse();
            return;
        }
        this.collapse();
        this.expandedId.set(job.id);
        this.detailRun.set(job.run);
        if (!job.run) return;
        this.failuresLoading.set(true);
        const [runResult, failureResult] = await Promise.all([
            fetchPublicDataSyncRun(job.id),
            fetchPublicDataSyncFailures(job.id),
        ]);
        if (this.expandedId() !== job.id) return;
        if (runResult.data) this.detailRun.set(runResult.data);
        this.failuresLoading.set(false);
        if (failureResult.data) {
            this.failures.set(failureResult.data.failures);
            this.failuresTotal.set(failureResult.data.total);
        } else if (failureResult.error) {
            this.actionError.set(failureResult.error);
        }
    }

    protected async cancel(runId: string): Promise<void> {
        await this.act(cancelPublicDataSyncRun(runId));
    }

    protected async resume(runId: string): Promise<void> {
        await this.act(resumePublicDataSyncRun(runId));
    }

    private async act(request: Promise<{ data: PublicDataSyncRun | null; error: string | null }>): Promise<void> {
        this.actionError.set(null);
        const result = await request;
        if (result.error) this.actionError.set(result.error);
        if (result.data) this.detailRun.set(result.data);
        await this.jobs.refresh();
    }

    private collapse(): void {
        this.expandedId.set(null);
        this.detailRun.set(null);
        this.failures.set([]);
        this.failuresTotal.set(0);
        this.failuresLoading.set(false);
    }
}
