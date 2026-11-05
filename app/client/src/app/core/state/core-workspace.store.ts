// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Injectable, signal } from '@angular/core';
import type { DatasetSummary } from '../../models/dataset.model';
import { deleteDataset, fetchDatasets } from '../../services/dataset.service';

export type DatasetSort = 'last_update' | 'created_at' | 'name' | 'measurements';

export interface DatasetListState {
    page: number;
    pageSize: number;
    query: string;
    source: 'all' | DatasetSummary['source'];
    sort: DatasetSort;
}

@Injectable({ providedIn: 'root' })
export class CoreWorkspaceStore {
    readonly datasets = signal<DatasetSummary[]>([]);
    /** Dataset list state, kept so the list survives a visit to a dataset page. */
    readonly datasetListState = signal<DatasetListState>({
        page: 1,
        pageSize: 12,
        query: '',
        source: 'all',
        sort: 'last_update',
    });
    readonly managementStatus = signal('');

    constructor() {
        void this.refreshDatasets();
    }

    async refreshDatasets(): Promise<void> {
        const result = await fetchDatasets();
        if (result.error || !result.data) {
            this.managementStatus.set(result.error || 'Failed to load datasets.');
            return;
        }
        this.managementStatus.set('');
        this.datasets.set(result.data.datasets);
    }

    async deleteDataset(datasetId: number): Promise<boolean> {
        const result = await deleteDataset(datasetId);
        if (result.error) {
            this.managementStatus.set(result.error);
            return false;
        }
        await this.refreshDatasets();
        return true;
    }
}
