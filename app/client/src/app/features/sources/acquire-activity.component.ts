// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, inject, input } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { UnifiedJob } from '../../models/job.model';
import { JobsMonitorService } from '../../services/jobs-monitor.service';
import { importedLink, isActiveStatus, jobTitle, pendingCount, statusLabel } from '../../shared/utils/job-presentation';

/** Summary of a card's latest finished run, so results are visible where they were started. Running jobs are tracked in Jobs. */
@Component({
    selector: 'adsmod-acquire-activity',
    standalone: true,
    imports: [RouterLink],
    template: `
        @if (job(); as current) {
            <div class="activity" role="status">
                <div class="activity-head">
                    <strong class="activity-title" [title]="title(current)">{{ title(current) }}</strong>
                    <span class="activity-status" [attr.data-status]="current.status">{{ label(current) }}</span>
                </div>
                <div class="activity-foot">
                    @if (current.saved_count !== null) {
                        <span>{{ current.saved_count }} saved · {{ current.skipped_count }} skipped · {{ current.failed_count }} failed</span>
                    }
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

    protected readonly job = computed<UnifiedJob | undefined>(() =>
        this.sourceJobs().find((job) => !isActiveStatus(job.status) && this.sessionRunIds().has(job.run?.id ?? '')),
    );
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

    protected openDetails(event: Event): void {
        // The popover closes on any outside document click, including this one.
        event.stopPropagation();
        this.jobs.popoverOpen.set(true);
        void this.jobs.refresh();
    }
}
