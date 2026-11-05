// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { FittingSessionStatus, FittingWeighting, OptimizationMethod } from '../../../models/fitting.model';

export const SESSION_STATUS_LABELS: Record<FittingSessionStatus, string> = {
    running: 'Running',
    completed: 'Completed',
    warning: 'Completed with warnings',
    failed: 'Failed',
    cancelled: 'Cancelled',
};

export const OPTIMIZER_LABELS: Record<OptimizationMethod, string> = {
    trf: 'Trust Region Reflective (TRF)',
    dogbox: 'Dogbox',
};

export const WEIGHTING_LABELS: Record<FittingWeighting, string> = {
    unweighted: 'Unweighted',
    inverse_sigma: 'Inverse sigma',
};

export function formatNumber(value: number | null | undefined): string {
    return value === null || value === undefined || !Number.isFinite(value)
        ? '—'
        : new Intl.NumberFormat(undefined, {
              maximumSignificantDigits: 6,
              notation: value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e6) ? 'scientific' : 'standard',
          }).format(value);
}

export function formatDuration(seconds: number | null): string {
    if (seconds === null || !Number.isFinite(seconds)) return '—';
    if (seconds < 1) return '< 1 s';
    if (seconds < 60) return `${Math.round(seconds)} s`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes} min ${Math.round(seconds % 60)} s`;
}
