// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ModelingStore } from '../../../core/state/modeling.store';
import type { ExperimentFitResult, FittingSessionDetail } from '../../../models/fitting.model';
import { pollJobStatus, resolvePollingIntervalMs } from '../../../services/job.service';
import { cancelFittingJob, fetchFittingSession } from '../../../services/fitting.service';
import { ProgressBarComponent } from '../../../shared/components/progress-bar/progress-bar.component';
import { OPTIMIZER_LABELS, SESSION_STATUS_LABELS, WEIGHTING_LABELS, formatDuration, formatNumber } from './session-format';

const FALLBACK_POLL_MS = 1500;

/** Execution log and persisted results for one fitting session. */
@Component({
    selector: 'adsmod-session-view',
    standalone: true,
    imports: [DatePipe, ProgressBarComponent, RouterLink],
    templateUrl: './session-view.component.html',
    styleUrl: './session-view.component.css',
})
export class SessionViewComponent implements OnInit {
    private readonly route = inject(ActivatedRoute);
    private readonly modeling = inject(ModelingStore);
    protected readonly statusLabels = SESSION_STATUS_LABELS;
    protected readonly optimizerLabels = OPTIMIZER_LABELS;
    protected readonly weightingLabels = WEIGHTING_LABELS;
    protected readonly formatNumber = formatNumber;
    protected readonly formatDuration = formatDuration;

    protected readonly session = signal<FittingSessionDetail | null>(null);
    protected readonly loadError = signal<string | null>(null);
    protected readonly liveLog = signal<string[] | null>(null);
    protected readonly progress = signal(0);
    protected readonly cancelling = signal(false);
    protected readonly actionError = signal<string | null>(null);

    protected readonly running = computed(() => this.session()?.status === 'running');
    protected readonly log = computed(() => (this.liveLog() ?? this.session()?.log ?? []).join('\n'));
    protected readonly fittedCount = computed(() =>
        (this.session()?.experiments ?? []).reduce((total, item) => total + item.results.filter((fit) => fit.status !== 'failed').length, 0),
    );
    protected readonly fitCount = computed(() => (this.session()?.experiments ?? []).reduce((total, item) => total + item.results.length, 0));

    private sessionId = 0;
    private timer: ReturnType<typeof setTimeout> | null = null;
    private destroyed = false;

    constructor() {
        inject(DestroyRef).onDestroy(() => {
            this.destroyed = true;
            if (this.timer) clearTimeout(this.timer);
        });
    }

    async ngOnInit(): Promise<void> {
        this.sessionId = Number(this.route.snapshot.paramMap.get('sessionId'));
        if (!Number.isInteger(this.sessionId) || this.sessionId < 1) {
            this.loadError.set('This fitting session does not exist.');
            return;
        }
        if (!this.modeling.catalogLoaded()) void this.modeling.loadCatalog();
        await this.load();
    }

    protected modelName(key: string | null): string {
        if (!key) return '—';
        return this.modeling.catalog().find((model) => model.key === key)?.name ?? key;
    }

    protected bestResultName(experiment: ExperimentFitResult): string {
        return experiment.results.find((fit) => fit.model === experiment.best_model)?.name ?? this.modelName(experiment.best_model);
    }

    protected boundsEntries(): { model: string; parameter: string; lower: number; upper: number; session: boolean }[] {
        const configuration = this.session()?.configuration;
        if (!configuration?.model_bounds) return [];
        return Object.entries(configuration.model_bounds).flatMap(([model, parameters]) =>
            Object.entries(parameters).map(([parameter, bounds]) => ({
                model,
                parameter,
                lower: bounds.lower,
                upper: bounds.upper,
                session: Boolean(configuration.session_overrides?.[model]?.[parameter]),
            })),
        );
    }

    protected async load(): Promise<void> {
        const result = await fetchFittingSession(this.sessionId);
        if (this.destroyed) return;
        this.loadError.set(result.error);
        if (!result.data) return;
        this.session.set(result.data);
        if (result.data.status === 'running') {
            this.schedule(() => this.track(result.data?.job_id ?? null), 0);
        } else {
            this.liveLog.set(null);
            this.progress.set(100);
            if (result.data.status !== this.modeling.sessions().find((item) => item.id === this.sessionId)?.status) {
                void this.modeling.loadSessions();
            }
        }
    }

    /** Follow the live job while it runs, then reload the persisted session. */
    private async track(jobId: string | null): Promise<void> {
        if (!jobId) {
            this.schedule(() => this.load(), FALLBACK_POLL_MS);
            return;
        }
        const status = await pollJobStatus('/fitting', jobId);
        if (this.destroyed) return;
        if (!status.data) {
            this.schedule(() => this.load(), FALLBACK_POLL_MS);
            return;
        }
        this.progress.set(status.data.progress);
        const log = status.data.result?.['log'];
        if (Array.isArray(log)) this.liveLog.set(log.map(String));
        if (['completed', 'failed', 'cancelled'].includes(status.data.status)) {
            this.cancelling.set(false);
            await this.load();
            return;
        }
        this.schedule(() => this.track(jobId), resolvePollingIntervalMs(status.data.poll_interval) ?? FALLBACK_POLL_MS);
    }

    private schedule(task: () => Promise<void>, delayMs: number): void {
        if (this.timer) clearTimeout(this.timer);
        if (!this.destroyed) this.timer = setTimeout(() => void task(), delayMs);
    }

    protected skippedList(session: FittingSessionDetail): string {
        return (session.configuration.skipped_experiments ?? []).map((item) => item.name).join(', ');
    }

    protected async cancel(): Promise<void> {
        const jobId = this.session()?.job_id;
        if (!jobId || this.cancelling()) return;
        this.cancelling.set(true);
        const result = await cancelFittingJob(jobId);
        if (result.error) {
            this.cancelling.set(false);
            this.actionError.set(`Cancellation could not be confirmed: ${result.error}`);
            return;
        }
        await this.load();
    }
}
