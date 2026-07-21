import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { DatasetManagementComponent } from './dataset-management.component';
@Component({ selector:'adsmod-source-page', standalone:true, imports:[CommonModule, DatasetManagementComponent], template:`
<div class="user-data-page">
<input #sourceFileInput id="source-file-input" class="source-file-input" type="file" accept=".csv,.txt,.xlsx,.json" [disabled]="store.isDatasetUploading()" (change)="fileChanged($event)"/>
@if (store.managementStatus()) { <p class="dataset-status error" role="alert">{{ store.managementStatus() }}</p> }
<adsmod-dataset-management [datasets]="store.userDatasets()" [selected]="store.selectedDataset()" (addRequested)="sourceFileInput.click()" (opened)="store.openDataset($event)" (metadataRequested)="store.setSelectedDataset($event)" (deleted)="store.deleteDataset($event)" (refreshed)="store.refreshUserDatasets()" (renamed)="store.renameDataset($event.name,$event.newName)" (metadataSaved)="store.saveMetadata($event)"/>
<section class="console-card source-stats-card" aria-labelledby="source-stats-title"><h2 id="source-stats-title">Dataset statistics</h2>@if (store.selectedDataset()) { <pre>{{ store.datasetStats() }}</pre> } @else { <div class="compact-placeholder"><span class="empty-state-icon" aria-hidden="true">▦</span><p>Select a dataset to see row and column summaries.</p></div> }</section>
</div>`})
export class SourcePageComponent { protected readonly store=inject(CoreWorkspaceStore); protected fileChanged(event:Event):void { const file=(event.target as HTMLInputElement).files?.[0]; if(file) { this.store.setPendingFile(file); void this.store.uploadPendingDataset(); } } }
