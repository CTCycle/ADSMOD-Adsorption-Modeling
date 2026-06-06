import { Component, inject } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { FileUploadComponent } from '../../shared/components/file-upload/file-upload.component';
import { MarkdownRendererComponent } from '../../shared/components/markdown-renderer/markdown-renderer.component';
import { NistCollectionRowsComponent } from './nist-collection-rows.component';

@Component({
    selector: 'adsmod-source-page',
    standalone: true,
    imports: [FileUploadComponent, MarkdownRendererComponent, NistCollectionRowsComponent],
    template: `
        <div class="source-page">
            <div class="source-columns">
                <section class="source-column" aria-label="Dataset source section">
                    <div class="source-column-row source-column-header">
                        <div class="section-title">Load Experimental Data</div>
                        <div class="section-caption">
                            Upload adsorption data from local CSV or Excel files.
                        </div>
                        <div class="section-caption section-caption-journey">
                            Load, validate, and prepare your baseline dataset before fitting and training.
                        </div>
                        <div class="source-inline-labels">
                            <span class="inline-pill">{{ datasetBadge }}</span>
                            <span class="inline-pill">{{ sampleBadge }}</span>
                        </div>
                    </div>

                    <div class="source-column-row source-column-widget">
                        <div class="dataset-upload-toolbar">
                            <adsmod-file-upload
                                label="Load dataset"
                                accept=".csv,.xls,.xlsx"
                                [autoUpload]="false"
                                [disabled]="store.isDatasetUploading()"
                                (fileSelected)="store.setPendingFile($event)"
                            />
                            <button
                                class="button primary dataset-upload-button"
                                type="button"
                                [disabled]="!store.pendingFile() || store.isDatasetUploading()"
                                (click)="uploadSelectedFile()"
                            >
                                {{ store.isDatasetUploading() ? 'Uploading...' : 'Upload' }}
                            </button>
                        </div>
                        <div class="source-inline-labels dataset-upload-meta">
                            <span class="inline-pill">Dataset: {{ datasetDisplayName }}</span>
                            <span class="inline-pill">Size: {{ datasetDisplaySize }}</span>
                        </div>
                    </div>

                    <div class="source-column-row source-column-log">
                        <div class="panel-title">Uploaded Data Statistics</div>
                        <div class="source-markdown-scroll markdown-content compact-stats">
                            <adsmod-markdown-renderer [content]="store.datasetStats()" />
                        </div>
                    </div>
                </section>

                <section class="source-column" aria-label="NIST source section">
                    <div class="source-column-row source-column-header">
                        <div class="section-title">NIST-A Collection</div>
                        <div class="section-caption">
                            Fetch NIST-A records into the local database using sampling fractions.
                        </div>
                        <div class="section-caption section-caption-journey">
                            Use NIST data to benchmark coverage before moving to fitting and training.
                        </div>
                    </div>

                    <div class="source-column-row source-column-widget">
                        <adsmod-nist-collection-rows (statusUpdate)="store.setNistStatusMessage($event)" />
                    </div>

                    <div class="source-column-row source-column-log">
                        <div class="panel-title">NIST-A Status Updates</div>
                        <div class="source-markdown-scroll markdown-content">
                            <adsmod-markdown-renderer [content]="store.nistStatusMessage()" />
                        </div>
                    </div>
                </section>
            </div>
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

    protected async uploadSelectedFile(): Promise<void> {
        await this.store.uploadPendingDataset();
    }
}
