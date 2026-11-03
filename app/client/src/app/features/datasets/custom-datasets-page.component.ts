// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, inject, signal } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { DatasetManagementComponent } from '../source/dataset-management.component';
import { DatasetImportWizardComponent } from '../source/dataset-import-wizard.component';
import { fetchDatasetConfiguration } from '../../services/dataset.service';

const DEFAULT_FILE_ACCEPT = '.csv,.xls,.xlsx';

@Component({
    selector: 'adsmod-custom-datasets-page',
    standalone: true,
    imports: [DatasetManagementComponent, DatasetImportWizardComponent],
    template: `
        <div class="data-page custom-datasets-page">
            <input #sourceFileInput class="source-file-input" type="file" aria-label="Choose a dataset file" [accept]="acceptedFileTypes()" (change)="fileChanged($event)" />
            @if (store.managementStatus()) {
                <div class="dataset-status error" role="alert">
                    <span>{{ store.managementStatus() }}</span>
                    <button class="button secondary" type="button" (click)="store.refreshDatasets()">Retry</button>
                </div>
            }
            <adsmod-dataset-management
                #datasetManagement
                [datasets]="store.customDatasets()"
                [initialState]="store.datasetListState()"
                (stateChanged)="store.datasetListState.set($event)"
                (addRequested)="sourceFileInput.click()"
                (deleted)="store.deleteDataset($event)"
                (renamed)="store.renameDataset($event.id, $event.newName)"
                (metadataSaved)="store.saveMetadata($event.id, $event.metadata)"
            />
            @if (pendingFile(); as file) {
                <adsmod-dataset-import-wizard [file]="file" (closed)="wizardClosed(datasetManagement)" (saved)="wizardSaved()" />
            }
        </div>
    `,
})
export class CustomDatasetsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);
    protected readonly pendingFile = signal<File | null>(null);
    protected readonly acceptedFileTypes = signal(DEFAULT_FILE_ACCEPT);

    constructor() {
        void this.loadDatasetConfiguration();
    }

    private async loadDatasetConfiguration(): Promise<void> {
        const result = await fetchDatasetConfiguration();
        if (result.data?.allowed_extensions?.length) {
            this.acceptedFileTypes.set(result.data.allowed_extensions.join(','));
        }
    }

    protected fileChanged(event: Event): void {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        if (file) {
            this.pendingFile.set(file);
        }
        // Allow choosing the same file again after the wizard closes.
        input.value = '';
    }

    protected wizardSaved(): void {
        this.pendingFile.set(null);
        void this.store.refreshDatasets();
    }

    protected wizardClosed(management: DatasetManagementComponent): void {
        this.pendingFile.set(null);
        queueMicrotask(() => management.focusAddButton());
    }
}
