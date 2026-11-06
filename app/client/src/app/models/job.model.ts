// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { PublicDataSyncRun } from './public-data.model';

export type UnifiedJobStatus = 'queued' | 'running' | 'completed' | 'partially_completed' | 'failed' | 'cancelled' | 'interrupted';

/** One background job of any kind, as reported by `GET /jobs`. */
export interface UnifiedJob {
    id: string;
    kind: 'acquisition' | 'fitting' | 'training';
    name: string;
    source: string | null;
    status: UnifiedJobStatus;
    progress: number;
    created_at: string;
    finished_at: string | null;
    saved_count: number | null;
    skipped_count: number | null;
    failed_count: number | null;
    error: string | null;
    run: PublicDataSyncRun | null;
}

export interface UnifiedJobListResponse {
    jobs: UnifiedJob[];
    active_count: number;
}

export interface SystemStatus {
    cpu_percent: number;
    memory_mb: number;
    database_size_bytes: number | null;
}

export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: string;
    message: string;
    poll_interval?: number;
}

export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    result?: Record<string, unknown>;
    error?: string;
    poll_interval?: number;
}
