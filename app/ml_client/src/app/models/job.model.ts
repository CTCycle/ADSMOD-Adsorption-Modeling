export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    result?: Record<string, unknown>;
    error?: string;
    poll_interval?: number;
}

export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: string;
    message: string;
    poll_interval?: number;
}
