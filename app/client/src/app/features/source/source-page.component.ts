import { CommonModule } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { DatasetManagementComponent } from './dataset-management.component';
import { DatasetImportWizardComponent } from './dataset-import-wizard.component';

@Component({ selector: 'adsmod-source-page', standalone: true, imports: [CommonModule, DatasetManagementComponent, DatasetImportWizardComponent], template: `
<div class="user-data-page">
    <input #sourceFileInput class="source-file-input" type="file" accept=".csv,.txt,.xlsx,.xls,.json" (change)="fileChanged($event)" />
    @if (store.managementStatus()) { <p class="dataset-status error" role="alert">{{ store.managementStatus() }}</p> }
    <adsmod-dataset-management [datasets]="store.userDatasets()" [selected]="store.selectedDatasetId()" (addRequested)="sourceFileInput.click()" (opened)="store.selectDataset($event)" (deleted)="store.deleteDataset($event)" (renamed)="store.renameDataset($event.id,$event.newName)" (metadataSaved)="store.saveMetadata($event.id,$event.metadata)" />
    @if (pendingFile(); as file) { <adsmod-dataset-import-wizard [file]="file" (closed)="pendingFile.set(null)" (saved)="wizardSaved($event)" /> }
</div>` })
export class SourcePageComponent { protected readonly store = inject(CoreWorkspaceStore); protected readonly pendingFile = signal<File | null>(null); protected fileChanged(event: Event): void { const file = (event.target as HTMLInputElement).files?.[0]; if (file) this.pendingFile.set(file); } protected wizardSaved(_id: number): void { this.pendingFile.set(null); void this.store.refreshDatasets(); } }
