// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    CustomUnitDefinition,
    DatasetDeleteImpact,
    DatasetDetail,
    DatasetEditIssue,
    DatasetImportResponse,
    DatasetImportInspection,
    DatasetSummary,
    DatasetUpdateRequest,
    ExperimentDetail,
    ExperimentFilters,
    ExperimentPage,
    ImportMapping,
    ImportPreview,
    ImportValidation,
    ObservationBranch,
    ObservationPage,
    UnitCatalog,
    UnitOption,
} from '../models/dataset.model';
import {
    extractErrorMessage,
    fetchWithTimeout,
    HTTP_TIMEOUT,
} from './http-timeout.service';

type ApiResult<T> = { data: T | null; error: string | null };

async function request<T>(
    path: string,
    init: RequestInit,
): Promise<ApiResult<T>> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/datasets${path}`,
            init,
            HTTP_TIMEOUT,
        );
        const data: unknown =
            response.status === 204
                ? null
                : await response.json().catch(() => ({}));
        return response.ok
            ? { data: data as T, error: null }
            : {
                  data: null,
                  error: extractErrorMessage(response, data),
              };
    } catch (error) {
        return {
            data: null,
            error:
                error instanceof Error
                    ? error.message
                    : 'Unable to reach the backend.',
        };
    }
}

function importForm(file: File, mapping?: ImportMapping): FormData {
    const form = new FormData();
    form.append('file', file);
    if (mapping) {
        form.append('mapping', JSON.stringify(mapping));
    }
    return form;
}

export function previewDataset(file: File): Promise<ApiResult<ImportPreview>> {
    return request('/import/preview', {
        method: 'POST',
        body: importForm(file),
    });
}

export function validateDataset(
    file: File,
    mapping: ImportMapping,
): Promise<ApiResult<ImportValidation>> {
    return request('/import/validate', {
        method: 'POST',
        body: importForm(file, mapping),
    });
}

export function commitDataset(
    file: File,
    mapping: ImportMapping,
): Promise<ApiResult<DatasetImportResponse>> {
    return request('/import/commit', {
        method: 'POST',
        body: importForm(file, mapping),
    });
}

export function fetchDatasets(): Promise<
    ApiResult<{ datasets: DatasetSummary[] }>
> {
    return request('', { method: 'GET' });
}

export function fetchDatasetConfiguration(): Promise<ApiResult<UnitCatalog>> {
    return request('/supported-units', { method: 'GET' });
}

export function saveCustomUnit(
    unit: CustomUnitDefinition,
): Promise<ApiResult<{ unit: UnitOption }>> {
    return request('/units', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(unit),
    });
}

export function deleteCustomUnit(unitId: number): Promise<ApiResult<null>> {
    return request(`/units/${unitId}`, { method: 'DELETE' });
}

export function fetchDatasetImport(
    datasetId: number,
): Promise<ApiResult<DatasetImportInspection>> {
    return request(`/${datasetId}/import`, { method: 'GET' });
}

export function fetchDataset(datasetId: number): Promise<ApiResult<DatasetDetail>> {
    return request(`/${datasetId}`, { method: 'GET' });
}

export function fetchDeleteImpact(datasetId: number): Promise<ApiResult<DatasetDeleteImpact>> {
    return request(`/${datasetId}/delete-impact`, { method: 'GET' });
}

export function fetchExperiments(
    datasetId: number,
    offset = 0,
    limit = 50,
    filters: ExperimentFilters = {},
): Promise<ApiResult<ExperimentPage>> {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    for (const [key, value] of Object.entries(filters)) {
        if (value !== undefined && value !== null && value !== '' && value !== false) {
            query.set(key, String(value));
        }
    }
    return request(`/${datasetId}/experiments?${query.toString()}`, { method: 'GET' });
}

export function fetchExperimentDetail(
    datasetId: number,
    experimentId: number,
): Promise<ApiResult<ExperimentDetail>> {
    return request(`/${datasetId}/experiments/${experimentId}`, { method: 'GET' });
}

export function fetchObservations(
    datasetId: number,
    isothermId: number,
    offset = 0,
    limit = 100,
    order: 'pressure' | 'source' = 'pressure',
    adsorbateId: number | null = null,
    branch: ObservationBranch | null = null,
): Promise<ApiResult<ObservationPage>> {
    const species = adsorbateId === null ? '' : `&adsorbate_id=${adsorbateId}`;
    const branchFilter = branch === null ? '' : `&branch=${branch}`;
    return request(
        `/${datasetId}/experiments/${isothermId}/observations?offset=${offset}&limit=${limit}&order=${order}${species}${branchFilter}`,
        { method: 'GET' },
    );
}

export function deleteDataset(datasetId: number): Promise<ApiResult<null>> {
    return request(`/${datasetId}`, { method: 'DELETE' });
}

export interface DatasetUpdateResult extends ApiResult<{ dataset: DatasetSummary }> {
    issues: DatasetEditIssue[];
}

/** Save an edit change set; validation failures return field-specific issues. */
export async function updateDataset(
    datasetId: number,
    changes: DatasetUpdateRequest,
): Promise<DatasetUpdateResult> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/datasets/${datasetId}`,
            {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(changes),
            },
            HTTP_TIMEOUT,
        );
        const payload: unknown = await response.json().catch(() => ({}));
        if (response.ok) {
            return { data: payload as { dataset: DatasetSummary }, error: null, issues: [] };
        }
        const detail = (payload as { detail?: { issues?: DatasetEditIssue[] } }).detail;
        return {
            data: null,
            error: extractErrorMessage(response, payload),
            issues: Array.isArray(detail?.issues) ? detail.issues : [],
        };
    } catch (error) {
        return {
            data: null,
            error: error instanceof Error ? error.message : 'Unable to reach the backend.',
            issues: [],
        };
    }
}
