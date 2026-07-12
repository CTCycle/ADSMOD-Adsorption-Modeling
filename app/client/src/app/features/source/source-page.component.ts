import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { DatasetManagementComponent } from './dataset-management.component';
@Component({ selector:'adsmod-source-page', standalone:true, imports:[CommonModule, DatasetManagementComponent], template:`
<div class="user-data-page">
<section class="console-card source-upload-card" aria-labelledby="source-upload-title">
<div class="card-title-row"><div><p class="eyebrow">Step 1</p><h2 id="source-upload-title">Add an experimental dataset</h2></div></div>
<label class="source-upload-dropzone" for="source-file-input"><span class="source-upload-icon" aria-hidden="true">↑</span><strong>Choose a dataset file</strong><span>CSV, TXT, XLSX, or JSON</span><span class="source-upload-cta">Browse files</span></label>
<input id="source-file-input" class="source-file-input" type="file" accept=".csv,.txt,.xlsx,.json" [disabled]="store.isDatasetUploading()" (change)="fileChanged($event)"/>
@if (store.pendingFile(); as file) { <p class="source-selected-file" aria-live="polite"><span aria-hidden="true">✓</span> {{ file.name }} <small>{{ file.size | number }} bytes</small></p> } @else { <p class="upload-helper">Upload one file at a time. Your dataset will appear in the workspace below.</p> }
<button class="button primary source-upload-submit" type="button" [disabled]="store.isDatasetUploading() || !store.pendingFile()" (click)="store.uploadPendingDataset()">{{ store.isDatasetUploading() ? 'Uploading…' : 'Upload dataset' }}</button>
</section>
@if (store.managementStatus()) { <p class="dataset-status error" role="alert">{{ store.managementStatus() }}</p> }
<adsmod-dataset-management [datasets]="store.userDatasets()" [selected]="store.selectedDataset()" [page]="store.editorPage()" (opened)="store.openDataset($event)" (deleted)="store.deleteDataset($event)" (refreshed)="store.refreshUserDatasets()" (renamed)="store.renameDataset($event.name,$event.newName)" (metadataSaved)="store.saveMetadata($event)" (paged)="store.loadDatasetPage($event)" (saved)="store.saveMutations($event)"/>
<section class="console-card source-stats-card" aria-labelledby="source-stats-title"><h2 id="source-stats-title">Dataset statistics</h2>@if (store.selectedDataset()) { <pre>{{ store.datasetStats() }}</pre> } @else { <div class="compact-placeholder"><span class="empty-state-icon" aria-hidden="true">▦</span><p>Select a dataset to see row and column summaries.</p></div> }</section>
</div>`})
export class SourcePageComponent { protected readonly store=inject(CoreWorkspaceStore); protected fileChanged(event:Event):void { const file=(event.target as HTMLInputElement).files?.[0]; if(file) this.store.setPendingFile(file); } }
