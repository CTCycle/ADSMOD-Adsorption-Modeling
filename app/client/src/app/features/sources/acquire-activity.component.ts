// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, inject, input, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { UnifiedJob } from '../../models/job.model';
import { ProgressBarComponent } from '../../shared/components/progress-bar/progress-bar.component';
import { JobsMonitorService } from '../../services/jobs-monitor.service';
import { cancelPublicDataSyncRun } from '../../services/public-data.service';
import { alreadyStoredCount, importedLink, isActiveStatus, jobTitle, pendingCount, statusLabel } from '../../shared/utils/job-presentation';

/** Summary of a card's latest finished run, so results are visible where they were started. Running jobs are tracked in Jobs. */
@Component({
    selector: 'adsmod-acquire-activity',
    standalone: true,
    imports: [RouterLink, ProgressBarComponent],
    template: `
        @if (active(); as running) {
            <div class="activity" role="status">
                <div class="activity-head">
                    <strong class="activity-title" [title]="title(running)">{{ title(running) }}</strong>
                    <span class="activity-status" [attr.data-status]="running.status">{{ label(running) }}</span>
                </div>
                <adsmod-progress-bar [value]="running.progress" label="Acquisition progress" />
                <div class="activity-foot">
                    @if (running.saved_count !== null) {
                        <span>{{ running.saved_count }} saved · {{ running.skipped_count }} skipped · {{ running.failed_count }} failed</span>
                    }
                    <span class="activity-links">
                        @if (running.run?.cancel_requested) {
                            <span>Cancelling after the current batch…</span>
                        } @else {
                            <button class="text-button" type="button" [disabled]="cancelling()" (click)="cancel(running)">Cancel</button>
                        }
                        <button class="text-button" type="button" (click)="openDetails($event)">Details</button>
                    </span>
                </div>
                @if (cancelError(); as message) { <p class="activity-error" role="alert">{{ message }}</p> }
            </div>
        } @else if (job(); as current) {
            <div class="activity" role="status">
                <div class="activity-head">
                    <strong class="activity-title" [title]="title(current)">{{ title(current) }}</strong>
                    <span class="activity-status" [attr.data-status]="current.status">{{ label(current) }}</span>
                </div>
                <div class="activity-foot">
                    @if (current.saved_count !== null) {
                        <span>{{ current.saved_count }} saved · {{ current.skipped_count }} skipped · {{ current.failed_count }} failed</span>
                    }
                    @if (alreadyStored(); as stored) { <span>{{ stored }} already stored</span> }
                    @if (remaining(); as count) { <span>{{ count }} remain; fetch again to continue</span> }
                    <span class="activity-links">
                        @if (link(); as target) { <a class="text-link" [routerLink]="target.path" [queryParams]="target.queryParams">{{ target.label }}</a> }
                        <button class="text-button" type="button" (click)="openDetails($event)">Details</button>
                    </span>
                </div>
                @if (current.error) { <p class="activity-error">{{ current.error }}</p> }
            </div>
        }
    `,
    styles: `
        :host { display: block; min-width: 0; }
        .activity { display: grid; gap: 5px; margin-top: 10px; padding: 8px 10px; border: 1px solid var(--border-color); background: var(--surface-subtle, var(--surface)); font-size: 0.74rem; }
        .activity-head, .activity-foot { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; }
        .activity-head { justify-content: space-between; }
        .activity-title { overflow: hidden; min-width: 0; flex: 1 1 12rem; font-size: 0.78rem; text-overflow: ellipsis; white-space: nowrap; }
        .activity-status { color: var(--text-secondary); font-weight: 600; text-transform: capitalize; font-variant-numeric: tabular-nums; }
        .activity-status[data-status='completed'] { color: var(--success-text); }
        .activity-status[data-status='failed'] { color: var(--danger-text); }
        .activity-foot { color: var(--text-secondary); }
        .activity-links { display: inline-flex; gap: 12px; margin-left: auto; align-items: baseline; }
        .text-link { color: var(--link); font-weight: 600; }
        .text-button { padding: 0; border: 0; background: none; color: var(--link); font: inherit; font-weight: 600; cursor: pointer; }
        .activity-error { margin: 0; color: var(--danger-text); }
    `,
})
export class AcquireActivityComponent {
    readonly source = input.required<'nist' | 'pubchem' | 'cod'>();
    /** Runs started or seen finishing while this page was open; older history stays in Jobs. */
    readonly sessionRunIds = input.required<ReadonlySet<string>>();

    private readonly jobs = inject(JobsMonitorService);
    private readonly sourceJobs = computed(() =>
        this.jobs.jobs().filter((job) => job.run?.source_key === this.source() && job.run.operation !== 'nist_index'),
    );

    protected readonly active = computed<UnifiedJob | undefined>(() => this.sourceJobs().find((job) => isActiveStatus(job.status)));
    protected readonly cancelling = signal(false);
    protected readonly cancelError = signal<string | null>(null);

    protected readonly job = computed<UnifiedJob | undefined>(() =>
        this.sourceJobs().find((job) => !isActiveStatus(job.status) && this.sessionRunIds().has(job.run?.id ?? '')),
    );
    protected readonly alreadyStored = computed(() => alreadyStoredCount(this.job()?.run));
    protected readonly remaining = computed(() => {
        const run = this.job()?.run;
        return run ? pendingCount(run) : 0;
    });
    protected readonly link = computed(() => {
        const run = this.job()?.run;
        return run ? importedLink(run) : null;
    });

    protected readonly title = jobTitle;
    protected readonly label = statusLabel;

    protected async cancel(job: UnifiedJob): Promise<void> {
        if (!job.run || this.cancelling()) return;
        this.cancelling.set(true);
        this.cancelError.set(null);
        const result = await cancelPublicDataSyncRun(job.run.id);
        this.cancelError.set(result.error);
        this.cancelling.set(false);
        await this.jobs.refresh();
    }

    protected openDetails(event: Event): void {
        // The popover closes on any outside document click, including this one.
        event.stopPropagation();
        this.jobs.popoverOpen.set(true);
        void this.jobs.refresh();
    }
}
