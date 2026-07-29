import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { NistDataPageComponent } from '../nist/nist-data-page.component';
import { DatasetManagementComponent } from '../source/dataset-management.component';

@Component({
    selector: 'adsmod-datasets-page',
    standalone: true,
    imports: [CommonModule, DatasetManagementComponent, NistDataPageComponent],
    template: `
        <div class="datasets-page">
            <input
                #sourceFileInput
                id="source-file-input"
                class="source-file-input"
                type="file"
                accept=".csv,.txt,.xlsx,.json"
                [disabled]="store.isDatasetUploading()"
                (change)="fileChanged($event)"
            />
            @if (store.managementStatus()) {
                <p class="dataset-status error" role="alert">{{ store.managementStatus() }}</p>
            }
            <adsmod-dataset-management
                [datasets]="store.userDatasets()"
                [selected]="store.selectedDataset()"
                (addRequested)="sourceFileInput.click()"
                (opened)="store.openDataset($event)"
                (metadataRequested)="store.setSelectedDataset($event)"
                (deleted)="store.deleteDataset($event)"
                (renamed)="store.renameDataset($event.name, $event.newName)"
                (metadataSaved)="store.saveMetadata($event)"
            />
            <adsmod-nist-data-page />
        </div>
    `,
})
export class DatasetsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);

    protected fileChanged(event: Event): void {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file) {
            this.store.setPendingFile(file);
            void this.store.uploadPendingDataset();
        }
    }
}
