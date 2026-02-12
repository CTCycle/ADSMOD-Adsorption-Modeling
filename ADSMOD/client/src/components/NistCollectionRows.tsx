import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    fetchNistCategoryStatus,
    pingNistCategoryServer,
    pollNistJobUntilComplete,
    startNistCategoryEnrichJob,
    startNistCategoryFetchJob,
    startNistCategoryIndexJob,
} from '../services';
import type {
    NISTCategoryKey,
    NISTCategoryOperationResponse,
    NISTCategoryRecordStatus,
} from '../types';

interface NistCollectionRowsProps {
    onStatusUpdate: (message: string) => void;
}

interface CategoryOperationState {
    running: boolean;
    operation: 'idle' | 'index' | 'fetch' | 'enrich';
    progress: number;
}

const CATEGORY_ORDER: NISTCategoryKey[] = ['experiments', 'guest', 'host'];
const CATEGORY_LABELS: Record<NISTCategoryKey, string> = {
    experiments: 'Adsorption experiments',
    guest: 'Adsorbate species',
    host: 'Adsorbent materials',
};
const OPERATION_LABELS: Record<CategoryOperationState['operation'], string> = {
    idle: 'Idle',
    index: 'Updating index',
    fetch: 'Fetching records',
    enrich: 'Enriching properties',
};

const FRACTION_MIN = 0.001;
const FRACTION_MAX = 1.0;
const FRACTION_STEP = 0.001;

const fractionStorageKey = (category: NISTCategoryKey): string =>
    `adsmod.nist.fraction.${category}`;

const clampFraction = (value: number): number =>
    Math.min(FRACTION_MAX, Math.max(FRACTION_MIN, value));

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
    const parsedValue = Number.parseFloat(rawValue);
    return normalizeFraction(parsedValue, 1.0);
};

const persistFraction = (category: NISTCategoryKey, value: number): void => {
    if (typeof window === 'undefined') {
        return;
    }
    window.localStorage.setItem(fractionStorageKey(category), value.toFixed(3));
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

const PingIcon = () => (
    <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 18.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z" fill="currentColor" />
        <path d="M7.9 15.1a5.8 5.8 0 0 1 8.2 0l1.4-1.4a7.8 7.8 0 0 0-11 0l1.4 1.4Z" fill="currentColor" />
        <path d="M4.5 11.7a10.6 10.6 0 0 1 15 0l1.4-1.4a12.6 12.6 0 0 0-17.8 0l1.4 1.4Z" fill="currentColor" />
    </svg>
);

const IndexIcon = () => (
    <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 5h14v2H5V5Zm0 6h14v2H5v-2Zm0 6h14v2H5v-2Z" fill="currentColor" />
    </svg>
);

const DownloadIcon = () => (
    <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
            d="M11 3h2v9.2l2.9-2.9 1.4 1.4-5.3 5.3-5.3-5.3 1.4-1.4 2.9 2.9V3Zm-6 14h14v4H5v-4Z"
            fill="currentColor"
        />
    </svg>
);

const EnrichIcon = () => (
    <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
            d="M9 3h6v2h-1v4.2l4.7 7.8A2.3 2.3 0 0 1 16.7 21H7.3a2.3 2.3 0 0 1-2-3.5L10 9.2V5H9V3Zm2.2 8.1-4.2 7a.3.3 0 0 0 .3.4h9.4a.3.3 0 0 0 .3-.4l-4.2-7h-1.6Z"
            fill="currentColor"
        />
    </svg>
);

export const NistCollectionRows: React.FC<NistCollectionRowsProps> = ({
    onStatusUpdate,
}) => {
    const [statuses, setStatuses] = useState<Record<NISTCategoryKey, NISTCategoryRecordStatus>>(
        initialStatusMap
    );
    const [fractions, setFractions] = useState<Record<NISTCategoryKey, number>>(() => ({
        experiments: readPersistedFraction('experiments'),
        guest: readPersistedFraction('guest'),
        host: readPersistedFraction('host'),
    }));
    const [fractionInputs, setFractionInputs] = useState<Record<NISTCategoryKey, string>>(() => ({
        experiments: readPersistedFraction('experiments').toFixed(3),
        guest: readPersistedFraction('guest').toFixed(3),
        host: readPersistedFraction('host').toFixed(3),
    }));
    const [operations, setOperations] = useState<Record<NISTCategoryKey, CategoryOperationState>>(
        initialOperationMap
    );

    const loadCategoryStatus = useCallback(async () => {
        const result = await fetchNistCategoryStatus();
        if (result.error || !result.data) {
            return;
        }

        const updatedStatus = initialStatusMap();
        result.data.categories.forEach((item) => {
            updatedStatus[item.category] = item;
        });
        setStatuses(updatedStatus);
    }, []);

    useEffect(() => {
        void loadCategoryStatus();
    }, [loadCategoryStatus]);

    const commitFraction = useCallback(
        (category: NISTCategoryKey) => {
            const parsed = Number.parseFloat(fractionInputs[category]);
            const normalized = Number.isFinite(parsed)
                ? normalizeFraction(parsed, fractions[category])
                : normalizeFraction(fractions[category], 1.0);
            persistFraction(category, normalized);
            setFractions((previous) => ({ ...previous, [category]: normalized }));
            setFractionInputs((previous) => ({
                ...previous,
                [category]: normalized.toFixed(3),
            }));
        },
        [fractionInputs, fractions]
    );

    const handleFractionInput = useCallback((category: NISTCategoryKey, value: string) => {
        setFractionInputs((previous) => ({ ...previous, [category]: value }));
        const parsed = Number.parseFloat(value);
        if (!Number.isFinite(parsed)) {
            return;
        }
        const normalized = normalizeFraction(parsed, 1.0);
        setFractions((previous) => ({ ...previous, [category]: normalized }));
    }, []);

    const setCategoryOperation = useCallback(
        (
            category: NISTCategoryKey,
            update: Partial<CategoryOperationState>
        ) => {
            setOperations((previous) => ({
                ...previous,
                [category]: { ...previous[category], ...update },
            }));
        },
        []
    );

    const finalizeCategoryOperation = useCallback((category: NISTCategoryKey) => {
        setCategoryOperation(category, {
            running: false,
            operation: 'idle',
            progress: 0,
        });
    }, [setCategoryOperation]);

    const runCategoryJob = useCallback(async (
        category: NISTCategoryKey,
        operation: CategoryOperationState['operation'],
        startJob: () => Promise<{ jobId: string | null; pollInterval?: number; error: string | null }>,
        onSuccess: (result: NISTCategoryOperationResponse) => void
    ) => {
        if (operations[category].running) {
            return;
        }

        setCategoryOperation(category, { running: true, operation, progress: 0 });
        try {
            const started = await startJob();
            if (started.error || !started.jobId) {
                onStatusUpdate(`[ERROR] ${CATEGORY_LABELS[category]}: ${started.error || 'Unable to start job.'}`);
                return;
            }

            const finished = await pollNistJobUntilComplete(
                started.jobId,
                started.pollInterval,
                (status) => {
                    const progressValue = typeof status.progress === 'number'
                        ? Math.min(100, Math.max(0, status.progress))
                        : 0;
                    setCategoryOperation(category, { progress: progressValue });
                }
            );

            if (finished.error || !finished.result) {
                onStatusUpdate(`[ERROR] ${CATEGORY_LABELS[category]}: ${finished.error || 'Job failed.'}`);
                return;
            }

            const rawResult = finished.result as Record<string, unknown>;
            const operationResult: NISTCategoryOperationResponse = {
                ...rawResult,
                status: typeof rawResult.status === 'string' ? rawResult.status : 'success',
                category,
            } as NISTCategoryOperationResponse;
            onSuccess(operationResult);
        } finally {
            finalizeCategoryOperation(category);
            await loadCategoryStatus();
        }
    }, [finalizeCategoryOperation, loadCategoryStatus, onStatusUpdate, operations, setCategoryOperation]);

    const handlePing = useCallback(async (category: NISTCategoryKey) => {
        if (operations[category].running) {
            return;
        }
        const result = await pingNistCategoryServer(category);
        if (result.error || !result.data) {
            onStatusUpdate(`[ERROR] ${CATEGORY_LABELS[category]}: ${result.error || 'Ping failed.'}`);
            return;
        }
        const pingData = result.data;

        setStatuses((previous) => ({
            ...previous,
            [category]: {
                ...previous[category],
                server_ok: pingData.server_ok,
                server_checked_at: pingData.checked_at,
            },
        }));
        const statusText = pingData.server_ok ? 'reachable' : 'unreachable';
        onStatusUpdate(`[INFO] ${CATEGORY_LABELS[category]} server is ${statusText}.`);
    }, [onStatusUpdate, operations]);

    const handleUpdateIndex = useCallback((category: NISTCategoryKey) => {
        void runCategoryJob(
            category,
            'index',
            () => startNistCategoryIndexJob(category),
            (result) => {
                onStatusUpdate(
                    [
                        `[INFO] ${CATEGORY_LABELS[category]} index updated.`,
                        '',
                        `- Available records: ${result.available_count ?? 0}`,
                    ].join('\n')
                );
            }
        );
    }, [onStatusUpdate, runCategoryJob]);

    const handleFetchRecords = useCallback((category: NISTCategoryKey) => {
        const fraction = normalizeFraction(fractions[category], 1.0);
        persistFraction(category, fraction);
        setFractions((previous) => ({ ...previous, [category]: fraction }));
        setFractionInputs((previous) => ({ ...previous, [category]: fraction.toFixed(3) }));

        void runCategoryJob(
            category,
            'fetch',
            () => startNistCategoryFetchJob(category, { fraction }),
            (result) => {
                onStatusUpdate(
                    [
                        `[INFO] ${CATEGORY_LABELS[category]} records updated.`,
                        '',
                        `- Requested records: ${result.requested_count ?? 0}`,
                        `- New records fetched: ${result.fetched_count ?? 0}`,
                        `- Local records: ${result.local_count ?? 0}`,
                    ].join('\n')
                );
            }
        );
    }, [fractions, onStatusUpdate, runCategoryJob]);

    const handleEnrich = useCallback((category: NISTCategoryKey) => {
        if (category === 'experiments') {
            return;
        }
        void runCategoryJob(
            category,
            'enrich',
            () => startNistCategoryEnrichJob(category),
            (result) => {
                onStatusUpdate(
                    [
                        `[INFO] ${CATEGORY_LABELS[category]} enrichment completed.`,
                        '',
                        `- Names requested: ${result.names_requested ?? 0}`,
                        `- Names matched: ${result.names_matched ?? 0}`,
                        `- Rows updated: ${result.rows_updated ?? 0}`,
                    ].join('\n')
                );
            }
        );
    }, [onStatusUpdate, runCategoryJob]);

    const rows = useMemo(() => CATEGORY_ORDER.map((category) => {
        const status = statuses[category];
        const operation = operations[category];
        const hasLocalData = status.local_count >= 1;
        const serverStateClass = status.server_ok === true
            ? 'is-online'
            : status.server_ok === false
                ? 'is-offline'
                : 'is-unknown';
        return (
            <div key={category} className="nist-category-row">
                <div className="nist-category-row-main">
                    <span
                        className={`nist-row-led-dot ${hasLocalData ? 'available' : 'empty'}`}
                        aria-hidden="true"
                    />
                    <span className="nist-row-name">{CATEGORY_LABELS[category]}</span>
                    <span className="nist-row-count">
                        {status.local_count} / {status.available_count}
                    </span>
                    <span className="nist-row-updated">{formatCompactDateTime(status.last_update)}</span>
                    <div className="nist-row-fraction-wrap">
                        <label htmlFor={`fraction-${category}`}>Fraction</label>
                        <input
                            id={`fraction-${category}`}
                            type="number"
                            min={FRACTION_MIN}
                            max={FRACTION_MAX}
                            step={FRACTION_STEP}
                            value={fractionInputs[category]}
                            onChange={(event) => handleFractionInput(category, event.target.value)}
                            onBlur={() => commitFraction(category)}
                            onKeyDown={(event) => {
                                if (event.key === 'Enter') {
                                    commitFraction(category);
                                }
                            }}
                            disabled={operation.running}
                        />
                    </div>
                </div>

                <div className="nist-category-row-actions">
                    <button
                        className={`nist-icon-button ${serverStateClass}`}
                        title="Server Status"
                        aria-label={`Server status for ${CATEGORY_LABELS[category]}`}
                        onClick={() => void handlePing(category)}
                        disabled={operation.running}
                    >
                        <PingIcon />
                    </button>
                    <button
                        className="nist-icon-button"
                        title="Update Index"
                        aria-label={`Update index for ${CATEGORY_LABELS[category]}`}
                        onClick={() => handleUpdateIndex(category)}
                        disabled={operation.running}
                    >
                        <IndexIcon />
                    </button>
                    <button
                        className="nist-icon-button"
                        title="Get Records"
                        aria-label={`Get records for ${CATEGORY_LABELS[category]}`}
                        onClick={() => handleFetchRecords(category)}
                        disabled={operation.running}
                    >
                        <DownloadIcon />
                    </button>
                    {status.supports_enrichment && (
                        <button
                            className="nist-icon-button"
                            title="Enrich Molecular Properties"
                            aria-label={`Enrich properties for ${CATEGORY_LABELS[category]}`}
                            onClick={() => handleEnrich(category)}
                            disabled={operation.running}
                        >
                            <EnrichIcon />
                        </button>
                    )}
                </div>

                {operation.running && (
                    <div className="nist-row-progress">
                        <span className="nist-spinner" aria-hidden="true" />
                        <span>{OPERATION_LABELS[operation.operation]} {operation.progress.toFixed(0)}%</span>
                        <div className="nist-row-progress-track" aria-hidden="true">
                            <div
                                className="nist-row-progress-fill"
                                style={{ width: `${operation.progress}%` }}
                            />
                        </div>
                    </div>
                )}
            </div>
        );
    }), [
        commitFraction,
        fractionInputs,
        handleEnrich,
        handleFetchRecords,
        handleFractionInput,
        handlePing,
        handleUpdateIndex,
        operations,
        statuses,
    ]);

    return (
        <div className="nist-rows-wrapper">
            {rows}
        </div>
    );
};
