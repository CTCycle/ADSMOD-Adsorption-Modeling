import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from './core-workspace.store';

describe('CoreWorkspaceStore', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        TestBed.resetTestingModule();
        fetchMock.mockReset();
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.includes('/datasets')) {
                return {
                    ok: true,
                    json: async () => ({
                        datasets: [
                            { id: 1, name: 'uploaded', source: 'uploaded', created_at: '2026-01-01T00:00:00Z', experiment_count: 1, observation_count: 2, tags: [], description: '' },
                            { id: 2, name: 'NIST ISODB', source: 'nist', created_at: '2026-01-02T00:00:00Z', experiment_count: 3, observation_count: 4, tags: [], description: '' },
                        ],
                    }),
                };
            }
            if (url.includes('/fitting/models')) {
                return { ok: true, json: async () => ({ status: 'success', pressure_unit: 'bar', uptake_unit: 'mmol/g', models: [] }) };
            }
            throw new Error(`Unhandled URL ${url}`);
        });
        vi.stubGlobal('fetch', fetchMock);
    });

    it('keeps all datasets for fitting while exposing uploaded-only custom datasets', async () => {
        const store = TestBed.inject(CoreWorkspaceStore);
        await store.refreshDatasets();

        expect(store.datasets().map((dataset) => dataset.source)).toEqual(['uploaded', 'nist']);
        expect(store.customDatasets().map((dataset) => dataset.name)).toEqual(['uploaded']);
    });
});
