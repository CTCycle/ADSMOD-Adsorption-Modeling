import { Component, EventEmitter, OnInit, Output, signal } from '@angular/core';
import {
    fetchNistCategoryStatus,
    pingNistCategoryServer,
    pollNistJobUntilComplete,
    startNistCategoryEnrichJob,
    startNistCategoryFetchJob,
    startNistCategoryIndexJob,
} from '../../services/nist.service';
import type {
    NISTCategoryFetchRequest,
    NISTCategoryKey,
    NISTCategoryOperationResponse,
    NISTCategoryRecordStatus,
} from '../../models/nist.model';

type CategoryOperation = 'idle' | 'index' | 'fetch' | 'enrich';

interface CategoryOperationState {
    running: boolean;
    operation: CategoryOperation;
    progress: number;
}

const CATEGORY_ORDER: NISTCategoryKey[] = ['experiments', 'guest', 'host'];
const CATEGORY_LABELS: Record<NISTCategoryKey, string> = {
    experiments: 'Adsorption experiments',
    guest: 'Adsorbate species',
    host: 'Adsorbent materials',
};
const OPERATION_LABELS: Record<CategoryOperation, string> = {
    idle: 'Idle',
    index: 'Updating index',
    fetch: 'Fetching records',
    enrich: 'Enriching properties',
};

const FRACTION_MIN = 0.001;
const FRACTION_MAX = 1.0;
const FRACTION_STEP = 0.001;

const fractionStorageKey = (category: NISTCategoryKey): string => `adsmod.nist.fraction.${category}`;
const clampFraction = (value: number): number => Math.min(FRACTION_MAX, Math.max(FRACTION_MIN, value));
const normalizeFraction = (value: unknown, fallback = 1.0): number => {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return fallback;
    }
    return clampFraction(Number(value.toFixed(3)));
};
const readPersistedFraction = (category: NISTCategoryKey): number => {
    if (typeof window === 'undefined') {
        return 1.0;
    }
    const rawValue = window.localStorage.getItem(fractionStorageKey(category));
    if (!rawValue) {
        return 1.0;
    }
    return normalizeFraction(Number.parseFloat(rawValue), 1.0);
};
const persistFraction = (category: NISTCategoryKey, value: number): void => {
    if (typeof window !== 'undefined') {
        window.localStorage.setItem(fractionStorageKey(category), value.toFixed(3));
    }
};
const formatCompactDateTime = (value: string | null): string => {
    if (!value) {
        return 'N.A.';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return 'N.A.';
    }
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    const hh = String(date.getHours()).padStart(2, '0');
    const min = String(date.getMinutes()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
};
const emptyStatus = (category: NISTCategoryKey): NISTCategoryRecordStatus => ({
    category,
    local_count: 0,
    available_count: 0,
    last_update: null,
    server_ok: null,
    server_checked_at: null,
    supports_enrichment: category !== 'experiments',
});
const initialStatusMap = (): Record<NISTCategoryKey, NISTCategoryRecordStatus> => ({
    experiments: emptyStatus('experiments'),
    guest: emptyStatus('guest'),
    host: emptyStatus('host'),
});
const initialOperationMap = (): Record<NISTCategoryKey, CategoryOperationState> => ({
    experiments: { running: false, operation: 'idle', progress: 0 },
    guest: { running: false, operation: 'idle', progress: 0 },
    host: { running: false, operation: 'idle', progress: 0 },
});

@Component({
    selector: 'adsmod-nist-collection-rows',
    standalone: true,
    template: `
        <div class="nist-rows-wrapper">
            @for (category of categories; track category) {
                @let status = statuses()[category];
                @let operation = operations()[category];
                <div class="nist-category-row">
                    <div class="nist-category-row-main">
                        <span class="nist-row-led-dot" [class.available]="status.local_count >= 1" [class.empty]="status.local_count < 1" aria-hidden="true"></span>
                        <span class="nist-row-name">{{ labels[category] }}</span>
                        <span class="nist-row-count">{{ status.local_count }} / {{ status.available_count }}</span>
                        <span class="nist-row-updated">{{ formatDate(status.last_update) }}</span>
                    </div>

                    <div class="nist-category-row-controls">
                        <div class="nist-row-fraction-wrap">
                            <label [for]="'fraction-' + category">Fraction</label>
                            <input
                                [id]="'fraction-' + category"
                                type="number"
                                [min]="fractionMin"
                                [max]="fractionMax"
                                [step]="fractionStep"
                                [value]="fractionInputs()[category]"
                                [disabled]="operation.running"
                                (input)="handleFractionInput(category, $event)"
                                (blur)="commitFraction(category)"
                                (keydown.enter)="commitFraction(category)"
                            />
                        </div>

                        <div class="nist-category-row-actions">
                            <button class="nist-icon-button" [class]="serverStateClass(status)" title="Server Status" [attr.aria-label]="'Server status for ' + labels[category]" [disabled]="operation.running" (click)="handlePing(category)">
                                <span aria-hidden="true">Wi</span>
                            </button>
                            <button class="nist-icon-button" title="Update Index" [attr.aria-label]="'Update index for ' + labels[category]" [disabled]="operation.running" (click)="handleUpdateIndex(category)">
                                <span aria-hidden="true">Idx</span>
                            </button>
                            <button class="nist-icon-button" title="Get Records" [attr.aria-label]="'Get records for ' + labels[category]" [disabled]="operation.running" (click)="handleFetchRecords(category)">
                                <span aria-hidden="true">Get</span>
                            </button>
                            @if (status.supports_enrichment) {
                                <button class="nist-icon-button" title="Enrich Molecular Properties" [attr.aria-label]="'Enrich properties for ' + labels[category]" [disabled]="operation.running" (click)="handleEnrich(category)">
                                    <span aria-hidden="true">Enr</span>
                                </button>
                            } @else {
                                <span class="nist-icon-button-placeholder" aria-hidden="true"></span>
                            }
                        </div>
                    </div>

                    @if (operation.running) {
                        <div class="nist-row-progress">
                            <span class="nist-spinner" aria-hidden="true"></span>
                            <span>{{ operationLabels[operation.operation] }} {{ operation.progress.toFixed(0) }}%</span>
                            <div class="nist-row-progress-track" aria-hidden="true">
                                <div class="nist-row-progress-fill" [style.width.%]="operation.progress"></div>
                            </div>
                        </div>
                    }
                </div>
            }
        </div>
    `,
})
export class NistCollectionRowsComponent implements OnInit {
    @Output() readonly statusUpdate = new EventEmitter<string>();

    protected readonly categories = CATEGORY_ORDER;
    protected readonly labels = CATEGORY_LABELS;
    protected readonly operationLabels = OPERATION_LABELS;
    protected readonly fractionMin = FRACTION_MIN;
    protected readonly fractionMax = FRACTION_MAX;
    protected readonly fractionStep = FRACTION_STEP;
    protected readonly statuses = signal<Record<NISTCategoryKey, NISTCategoryRecordStatus>>(initialStatusMap());
    protected readonly operations = signal<Record<NISTCategoryKey, CategoryOperationState>>(initialOperationMap());
    protected readonly fractions = signal<Record<NISTCategoryKey, number>>({
        experiments: readPersistedFraction('experiments'),
        guest: readPersistedFraction('guest'),
        host: readPersistedFraction('host'),
    });
    protected readonly fractionInputs = signal<Record<NISTCategoryKey, string>>({
        experiments: readPersistedFraction('experiments').toFixed(3),
        guest: readPersistedFraction('guest').toFixed(3),
        host: readPersistedFraction('host').toFixed(3),
    });

    ngOnInit(): void {
        void this.loadCategoryStatus();
    }

    protected formatDate(value: string | null): string {
        return formatCompactDateTime(value);
    }

    protected serverStateClass(status: NISTCategoryRecordStatus): string {
        if (status.server_ok === true) {
            return 'is-online';
        }
        if (status.server_ok === false) {
            return 'is-offline';
        }
        return 'is-unknown';
    }

    protected handleFractionInput(category: NISTCategoryKey, event: Event): void {
        const input = event.target as HTMLInputElement;
        this.fractionInputs.update((previous) => ({ ...previous, [category]: input.value }));
        const parsed = Number.parseFloat(input.value);
        if (Number.isFinite(parsed)) {
            this.fractions.update((previous) => ({ ...previous, [category]: normalizeFraction(parsed, 1.0) }));
        }
    }

    protected commitFraction(category: NISTCategoryKey): void {
        const parsed = Number.parseFloat(this.fractionInputs()[category]);
        const normalized = Number.isFinite(parsed)
            ? normalizeFraction(parsed, this.fractions()[category])
            : normalizeFraction(this.fractions()[category], 1.0);
        persistFraction(category, normalized);
        this.fractions.update((previous) => ({ ...previous, [category]: normalized }));
        this.fractionInputs.update((previous) => ({ ...previous, [category]: normalized.toFixed(3) }));
    }

    protected async handlePing(category: NISTCategoryKey): Promise<void> {
        if (this.operations()[category].running) {
            return;
        }
        const result = await pingNistCategoryServer(category);
        if (result.error || !result.data) {
            this.statusUpdate.emit(`[ERROR] ${CATEGORY_LABELS[category]}: ${result.error || 'Ping failed.'}`);
            return;
        }
        const pingData = result.data;
        this.statuses.update((previous) => ({
            ...previous,
            [category]: {
                ...previous[category],
                server_ok: pingData.server_ok,
                server_checked_at: pingData.checked_at,
            },
        }));
        this.statusUpdate.emit(`[INFO] ${CATEGORY_LABELS[category]} server is ${pingData.server_ok ? 'reachable' : 'unreachable'}.`);
    }

    protected handleUpdateIndex(category: NISTCategoryKey): void {
        void this.runCategoryJob(category, 'index', () => startNistCategoryIndexJob(category), (result) => {
            this.statusUpdate.emit([
                `[INFO] ${CATEGORY_LABELS[category]} index updated.`,
                '',
                `- Available records: ${result.available_count ?? 0}`,
            ].join('\n'));
        });
    }

    protected handleFetchRecords(category: NISTCategoryKey): void {
        const fraction = normalizeFraction(this.fractions()[category], 1.0);
        persistFraction(category, fraction);
        this.fractions.update((previous) => ({ ...previous, [category]: fraction }));
        this.fractionInputs.update((previous) => ({ ...previous, [category]: fraction.toFixed(3) }));
        const payload: NISTCategoryFetchRequest = { fraction };
        void this.runCategoryJob(category, 'fetch', () => startNistCategoryFetchJob(category, payload), (result) => {
            this.statusUpdate.emit([
                `[INFO] ${CATEGORY_LABELS[category]} records updated.`,
                '',
                `- Requested records: ${result.requested_count ?? 0}`,
                `- New records fetched: ${result.fetched_count ?? 0}`,
                `- Local records: ${result.local_count ?? 0}`,
            ].join('\n'));
        });
    }

    protected handleEnrich(category: NISTCategoryKey): void {
        if (category === 'experiments') {
            return;
        }
        void this.runCategoryJob(category, 'enrich', () => startNistCategoryEnrichJob(category), (result) => {
            this.statusUpdate.emit([
                `[INFO] ${CATEGORY_LABELS[category]} enrichment completed.`,
                '',
                `- Names requested: ${result.names_requested ?? 0}`,
                `- Names matched: ${result.names_matched ?? 0}`,
                `- Rows updated: ${result.rows_updated ?? 0}`,
            ].join('\n'));
        });
    }

    private async loadCategoryStatus(): Promise<void> {
        const result = await fetchNistCategoryStatus();
        if (result.error || !result.data) {
            return;
        }
        const updatedStatus = initialStatusMap();
        result.data.categories.forEach((item) => {
            updatedStatus[item.category] = item;
        });
        this.statuses.set(updatedStatus);
    }

    private async runCategoryJob(
        category: NISTCategoryKey,
        operation: CategoryOperation,
        startJob: () => Promise<{ jobId: string | null; pollInterval?: number; error: string | null }>,
        onSuccess: (result: NISTCategoryOperationResponse) => void
    ): Promise<void> {
        if (this.operations()[category].running) {
            return;
        }
        this.setCategoryOperation(category, { running: true, operation, progress: 0 });
        try {
            const started = await startJob();
            if (started.error || !started.jobId) {
                this.statusUpdate.emit(`[ERROR] ${CATEGORY_LABELS[category]}: ${started.error || 'Unable to start job.'}`);
                return;
            }
            const finished = await pollNistJobUntilComplete(started.jobId, started.pollInterval, (status) => {
                const progressValue = typeof status.progress === 'number' ? Math.min(100, Math.max(0, status.progress)) : 0;
                this.setCategoryOperation(category, { progress: progressValue });
            });
            if (finished.error || !finished.result) {
                this.statusUpdate.emit(`[ERROR] ${CATEGORY_LABELS[category]}: ${finished.error || 'Job failed.'}`);
                return;
            }
            const rawResult = finished.result;
            onSuccess({
                ...rawResult,
                status: typeof rawResult['status'] === 'string' ? rawResult['status'] : 'success',
                category,
            } as NISTCategoryOperationResponse);
        } finally {
            this.setCategoryOperation(category, { running: false, operation: 'idle', progress: 0 });
            await this.loadCategoryStatus();
        }
    }

    private setCategoryOperation(category: NISTCategoryKey, update: Partial<CategoryOperationState>): void {
        this.operations.update((previous) => ({
            ...previous,
            [category]: { ...previous[category], ...update },
        }));
    }
}
