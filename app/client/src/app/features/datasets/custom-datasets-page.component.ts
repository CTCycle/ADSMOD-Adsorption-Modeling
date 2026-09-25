import { CommonModule } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type {
    DatasetImportInspection,
    ObservationPage,
} from '../../models/dataset.model';
import { DatasetManagementComponent } from '../source/dataset-management.component';
import { DatasetImportWizardComponent } from '../source/dataset-import-wizard.component';
import { DatasetInspectorComponent } from '../source/dataset-inspector.component';
import {
    fetchDatasetConfiguration,
    fetchDatasetImport,
    fetchObservations,
} from '../../services/dataset.service';

const DEFAULT_FILE_ACCEPT = '.csv,.xls,.xlsx';

@Component({
    selector: 'adsmod-custom-datasets-page',
    standalone: true,
    imports: [
        CommonModule,
        DatasetManagementComponent,
        DatasetImportWizardComponent,
        DatasetInspectorComponent,
    ],
    template: `
        <div class="data-page custom-datasets-page">
            <p class="page-context">Public source collections are available under Public Data.</p>
            <input #sourceFileInput class="source-file-input" type="file" [accept]="acceptedFileTypes()" (change)="fileChanged($event)" />
            @if (store.managementStatus()) {
                <div class="dataset-status error" role="alert">
                    <span>{{ store.managementStatus() }}</span>
                    <button class="button secondary" type="button" (click)="store.refreshDatasets()">Retry</button>
                </div>
            }
            <adsmod-dataset-management
                [datasets]="store.customDatasets()"
                [selected]="store.selectedDatasetId()"
                (addRequested)="sourceFileInput.click()"
                (opened)="inspectDataset($event)"
                (deleted)="store.deleteDataset($event)"
                (renamed)="store.renameDataset($event.id, $event.newName)"
                (metadataSaved)="store.saveMetadata($event.id, $event.metadata)"
            />
            @if (store.selectedDataset(); as selectedDataset) {
                <adsmod-dataset-inspector
                    [dataset]="selectedDataset"
                    [inspection]="importInspection()"
                    [importLoading]="importLoading()"
                    [importError]="importError()"
                    [experiments]="store.experiments()"
                    [experimentsLoading]="store.experimentsLoading()"
                    [selectedExperimentId]="store.selectedExperimentId()"
                    [observations]="observations()"
                    [observationsLoading]="observationsLoading()"
                    [observationsError]="observationsError()"
                    (experimentSelected)="inspectExperiment($event)"
                />
            }
            @if (pendingFile(); as file) {
                <adsmod-dataset-import-wizard [file]="file" (closed)="pendingFile.set(null)" (saved)="wizardSaved()" />
            }
        </div>
    `,
})
export class CustomDatasetsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);
    protected readonly pendingFile = signal<File | null>(null);
    protected readonly acceptedFileTypes = signal(DEFAULT_FILE_ACCEPT);
    protected readonly importInspection = signal<DatasetImportInspection | null>(null);
    protected readonly importLoading = signal(false);
    protected readonly importError = signal('');
    protected readonly observations = signal<ObservationPage | null>(null);
    protected readonly observationsLoading = signal(false);
    protected readonly observationsError = signal('');
    private inspectionRevision = 0;

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
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file) {
            this.pendingFile.set(file);
        }
    }

    protected async inspectDataset(datasetId: number): Promise<void> {
        const revision = ++this.inspectionRevision;
        this.importInspection.set(null);
        this.importLoading.set(true);
        this.importError.set('');
        this.observations.set(null);
        this.observationsLoading.set(false);
        this.observationsError.set('');

        const importRequest = fetchDatasetImport(datasetId);
        await this.store.selectDataset(datasetId);
        const result = await importRequest;
        if (
            revision !== this.inspectionRevision ||
            this.store.selectedDatasetId() !== datasetId
        ) {
            return;
        }
        this.importLoading.set(false);
        if (result.error || !result.data) {
            this.importError.set(
                result.error || 'Failed to load saved import details.',
            );
        } else {
            this.importInspection.set(result.data);
        }

        const experimentId = this.store.selectedExperimentId();
        if (experimentId !== null) {
            await this.loadObservations(datasetId, experimentId, revision);
        }
    }

    protected async inspectExperiment(experimentId: number): Promise<void> {
        const datasetId = this.store.selectedDatasetId();
        if (datasetId === null) {
            return;
        }
        this.store.setSelectedExperiment(experimentId);
        await this.loadObservations(datasetId, experimentId, this.inspectionRevision);
    }

    private async loadObservations(
        datasetId: number,
        experimentId: number,
        revision: number,
    ): Promise<void> {
        this.observations.set(null);
        this.observationsLoading.set(true);
        this.observationsError.set('');
        const result = await fetchObservations(datasetId, experimentId);
        if (
            revision !== this.inspectionRevision ||
            this.store.selectedDatasetId() !== datasetId ||
            this.store.selectedExperimentId() !== experimentId
        ) {
            return;
        }
        this.observationsLoading.set(false);
        if (result.error || !result.data) {
            this.observationsError.set(
                result.error || 'Failed to load observations.',
            );
            return;
        }
        this.observations.set(result.data);
    }

    protected wizardSaved(): void {
        this.pendingFile.set(null);
        void this.store.refreshDatasets();
    }
}
