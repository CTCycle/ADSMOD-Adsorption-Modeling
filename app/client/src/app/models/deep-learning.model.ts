// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export type DeepLearningStatusName = 'not_installed' | 'installing' | 'verifying' | 'installed' | 'failed' | 'incompatible';
export type DeepLearningAction = 'install' | 'uninstall' | 'repair';

export interface DeepLearningError { code: string; summary: string; hint: string; }

export interface DeepLearningOperation {
    id: string;
    kind: DeepLearningAction;
    status: 'running' | 'succeeded' | 'failed';
    phase: string | null;
    started_at: string;
    finished_at: string | null;
    error: DeepLearningError | null;
    warnings: string[];
    restart_healthy: boolean | null;
}

export interface DeepLearningEnvironment {
    python_version: string;
    platform: string;
    torch_version: string | null;
    keras_version: string | null;
    scikit_learn_version: string | null;
    cuda_runtime: string | null;
    cuda_available: boolean | null;
    devices: string[];
    gpu_names: string[];
    driver_version: string | null;
    driver_cuda_level: string | null;
}

export interface DeepLearningStatus {
    status: DeepLearningStatusName;
    available: boolean;
    reason: string | null;
    warnings: string[];
    environment: DeepLearningEnvironment;
    operation: DeepLearningOperation | null;
    last_validated_at: string | null;
    log_path: string;
}

export interface DeepLearningCheck { name: string; status: 'ok' | 'warning' | 'blocked'; detail: string; }

export interface DeepLearningPlan {
    action: DeepLearningAction;
    allowed: boolean;
    checks: DeepLearningCheck[];
    packages: string[];
    download_bytes: number | null;
    download_estimated: boolean;
    requires_restart: boolean;
}
