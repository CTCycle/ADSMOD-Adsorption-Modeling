// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { UnifiedJob } from '../../models/job.model';
import type { NISTCategory, PublicDataSyncRun } from '../../models/public-data.model';

export interface ImportedLink {
    label: string;
    path: string[];
    queryParams: Record<string, string | number>;
}

export const NIST_COLLECTIONS: { category: NISTCategory; label: string }[] = [
    { category: 'guest', label: 'Adsorbate species' },
    { category: 'host', label: 'Adsorbent materials' },
    { category: 'experiments', label: 'Adsorption isotherms' },
];

const ACTIVE_STATUSES = new Set(['queued', 'running']);

export const isActiveStatus = (status: string): boolean => ACTIVE_STATUSES.has(status);

export function statusLabel(job: Pick<UnifiedJob, 'status' | 'run'>): string {
    if (job.run?.cancel_requested && job.status === 'running') return 'cancelling';
    return job.status.replaceAll('_', ' ');
}

export function operationLabel(run: PublicDataSyncRun): string {
    const category = NIST_COLLECTIONS.find((item) => item.category === run.parameters['category'])?.label ?? 'collection';
    switch (run.operation) {
        case 'nist_category': {
            const picked = run.parameters['identifiers'] as string[] | undefined;
            return picked ? `NIST targeted fetch · ${picked.length} isotherms` : `NIST fetch · ${category}`;
        }
        case 'pubchem_resolve': return `PubChem · ${String(run.parameters['query'] ?? 'compound')}`;
        case 'pubchem_enrich': return run.parameters['conformers'] ? 'PubChem 3D structures for stored chemicals' : 'PubChem enrichment of stored chemicals';
        case 'cod_import': {
            if ('fraction' in run.parameters) return 'COD structures for stored materials';
            const ids = (run.parameters['cod_ids'] as string[] | undefined) ?? [String(run.parameters['cod_id'] ?? '')];
            return ids.length === 1 ? `COD import · ${ids[0]}` : `COD import · ${ids.length} structures`;
        }
        case 'nist_index': return `NIST availability check · ${category}`;
        default: return 'Provider health check';
    }
}

export function jobTitle(job: UnifiedJob): string {
    if (job.run) return operationLabel(job.run);
    if (job.kind === 'fitting') return 'Model fitting';
    return job.name === 'training_dataset' ? 'Training dataset build' : 'Deep learning training';
}

export function importedLink(run: PublicDataSyncRun): ImportedLink | null {
    if (!['completed', 'partially_completed'].includes(run.status)) return null;
    const entityId = Number(run.checkpoint['entity_id']);
    const entityIds = run.checkpoint['entity_ids'];
    const single = Number.isInteger(entityId) ? entityId : Array.isArray(entityIds) && entityIds.length === 1 ? Number(entityIds[0]) : null;
    const category = run.parameters['category'];
    switch (run.operation) {
        case 'nist_category':
            if (!run.imported_count && !run.skipped_count) return null;
            if (category === 'experiments') return { label: 'View datasets', path: ['/datasets'], queryParams: {} };
            return category === 'guest'
                ? { label: 'View species', path: ['/datasets'], queryParams: { kind: 'species' } }
                : { label: 'View materials', path: ['/datasets'], queryParams: { kind: 'material' } };
        case 'pubchem_resolve':
        case 'pubchem_enrich':
            return run.imported_count || run.skipped_count ? { label: 'View species', path: ['/datasets'], queryParams: { kind: 'species' } } : null;
        case 'cod_import':
            if (single !== null) return { label: 'Open structure', path: ['/sources', 'structures'], queryParams: { id: single } };
            return run.imported_count || run.skipped_count ? { label: 'View structures', path: ['/sources', 'structures'], queryParams: { source: 'cod' } } : null;
        default:
            return null;
    }
}

export function pendingCount(run: PublicDataSyncRun): number {
    if (isActiveStatus(run.status) || (run.operation === 'cod_import' && !('fraction' in run.parameters))) return 0;
    return Number(run.checkpoint['pending_count'] ?? 0);
}

/** Records scanned past because they were stored already; reported apart from skipped (unsupported) ones. */
export function alreadyStoredCount(run: PublicDataSyncRun | null | undefined): number {
    return Number(run?.checkpoint?.['already_stored_count'] ?? 0);
}

export function formatPercent(value: number): string {
    return `${Math.round(value)}%`;
}

export function formatDate(value: string | null): string {
    if (!value) return '—';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

/** Time of day for today's events, short date and time otherwise. */
export function formatShort(value: string | null): string {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    const sameDay = date.toDateString() === new Date().toDateString();
    return new Intl.DateTimeFormat(undefined, sameDay ? { timeStyle: 'short' } : { dateStyle: 'short', timeStyle: 'short' }).format(date);
}

export function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}
