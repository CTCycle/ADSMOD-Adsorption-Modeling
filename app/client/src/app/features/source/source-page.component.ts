import { Component, inject } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { NistCollectionRowsComponent } from './nist-collection-rows.component';

@Component({
    selector: 'adsmod-source-page',
    standalone: true,
    imports: [NistCollectionRowsComponent],
    template: `
        <div class="source-console-grid">
            <section class="console-card source-upload-card" aria-label="Dataset source section">
                <div class="card-title-row">
                    <h2>Load Experimental Data</h2>
                    <button class="card-info-button" type="button" aria-label="Upload help">i</button>
                </div>
                <label class="drop-zone" for="source-file-input">
                    <input
                        id="source-file-input"
                        type="file"
                        accept=".csv,.txt,.xlsx,.json"
                        [disabled]="store.isDatasetUploading()"
                        (change)="handleFileChange($event)"
                    />
                    <svg class="upload-cloud" aria-hidden="true" viewBox="0 0 64 48" fill="none">
                        <path d="M18 36H14C8.5 36 4 31.5 4 26s4.5-10 10-10c1.3 0 2.5.2 3.6.7C20.4 9.2 27.4 4 35.5 4 46.3 4 55 12.7 55 23.5V24h1c4.4 0 8 3.6 8 8s-3.6 8-8 8H45" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
                        <path d="M32 42V22m0 0-9 9m9-9 9 9" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    <strong>Drag and drop files here</strong>
                    <span>CSV, TXT, XLSX, JSON</span>
                    <span>Max file size: 200 MB</span>
                </label>
                <div class="upload-divider"><span></span><em>or</em><span></span></div>
                <button
                    class="reference-upload-button"
                    type="button"
                    [disabled]="store.isDatasetUploading()"
                    (click)="uploadSelectedFile()"
                >
                    <span aria-hidden="true">↥</span>
                    {{ store.isDatasetUploading() ? 'Uploading...' : 'Upload Files' }}
                </button>
                <p class="upload-helper">You can upload multiple files.</p>
            </section>

            <section class="console-card source-nist-card" aria-label="NIST source section">
                <div class="card-title-row">
                    <h2>NIST-A Collection</h2>
                    <button class="card-info-button" type="button" aria-label="NIST help">i</button>
                </div>
                <adsmod-nist-collection-rows (statusUpdate)="store.setNistStatusMessage($event)" />
            </section>

            <section class="console-card source-stats-card" aria-label="Uploaded dataset statistics">
                <div class="card-title-row">
                    <h2>Uploaded Data Statistics</h2>
                    <button class="copy-icon-button" type="button" aria-label="Copy statistics">□</button>
                </div>
                <pre class="reference-markdown">{{ store.datasetStats() }}</pre>
            </section>

            <section class="console-card source-log-card" aria-label="NIST status updates">
                <div class="card-title-row">
                    <h2>NIST-A Status Updates</h2>
                    <button class="clear-log-button" type="button">⌫ Clear</button>
                </div>
                <pre class="reference-log">{{ store.nistStatusMessage() }}</pre>
            </section>
        </div>
    `,
})
export class SourcePageComponent {
    protected readonly store = inject(CoreWorkspaceStore);

    protected get datasetBadge(): string {
        return this.store.datasetName() || 'No dataset loaded';
    }

    protected get sampleBadge(): string {
        const samples = this.store.datasetSamples();
        return samples > 0 ? `${samples} samples` : '0 samples';
    }

    protected get datasetDisplayName(): string {
        return this.store.datasetName() || this.store.pendingFile()?.name || 'N.A.';
    }

    protected get datasetDisplaySize(): string {
        return this.store.datasetSizeKb() || this.store.pendingFileSize() || 'N.A.';
    }

    protected handleFileChange(event: Event): void {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        if (file) {
            this.store.setPendingFile(file);
        }
        input.value = '';
    }

    protected async uploadSelectedFile(): Promise<void> {
        await this.store.uploadPendingDataset();
    }
}
