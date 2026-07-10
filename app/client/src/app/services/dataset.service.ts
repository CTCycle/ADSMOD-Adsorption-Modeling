import { API_BASE_URL } from '../core/config/api-base-url';
import type { DatasetMetadata, DatasetRowsPage, DatasetRowMutation, DatasetSummary, DatasetUploadResponse } from '../models/dataset.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

async function request<T>(path: string, init: RequestInit): Promise<{ data: T | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/datasets${path}`, init, HTTP_TIMEOUT);
        const data = await response.json().catch(() => ({}));
        return response.ok ? { data: data as T, error: null } : { data: null, error: extractErrorMessage(response, data) };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'Unable to reach the backend.' };
    }
}

export async function uploadDataset(file: File) {
    const form = new FormData(); form.append('file', file);
    return request<DatasetUploadResponse>('', { method: 'POST', body: form });
}
export async function fetchDatasets() { return request<{ datasets: DatasetSummary[] }>('', { method: 'GET' }); }
export async function deleteDataset(name: string) { return request<unknown>(`/by-name/${encodeURIComponent(name)}`, { method: 'DELETE' }); }
export async function renameDataset(name: string, newName: string) { return request<{ dataset: DatasetSummary }>(`/by-name/${encodeURIComponent(name)}/rename`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ new_name: newName }) }); }
export async function getMetadata(name: string) { return request<DatasetMetadata>(`/by-name/${encodeURIComponent(name)}/metadata`, { method: 'GET' }); }
export async function updateMetadata(name: string, metadata: DatasetMetadata) { return request<{ dataset: DatasetSummary }>(`/by-name/${encodeURIComponent(name)}/metadata`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(metadata) }); }
export async function fetchRows(name: string, offset: number, limit: number) { return request<DatasetRowsPage>(`/by-name/${encodeURIComponent(name)}/rows?offset=${offset}&limit=${limit}`, { method: 'GET' }); }
export async function mutateRows(name: string, operations: DatasetRowMutation[]) { return request<{ dataset: DatasetSummary }>(`/by-name/${encodeURIComponent(name)}/rows`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ operations }) }); }