import { Component, inject } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { FileUploadComponent } from '../../shared/components/file-upload/file-upload.component';
import { MarkdownRendererComponent } from '../../shared/components/markdown-renderer/markdown-renderer.component';
import { HeaderTabsComponent } from '../../layout/header-tabs.component';
import { NistCollectionRowsComponent } from './nist-collection-rows.component';

@Component({
    selector: 'adsmod-source-page',
    standalone: true,
    imports: [FileUploadComponent, MarkdownRendererComponent, NistCollectionRowsComponent, HeaderTabsComponent],
    template: `
        <div class="route-workspace route-workspace-source">
            <aside class="route-rail route-rail-source" aria-label="Source overview">
                <div class="route-rail-brand">
                    <div class="route-rail-logo" aria-hidden="true">AD</div>
                    <div class="route-rail-wordmark">ADSMOD</div>
                </div>
                <div class="route-rail-copy">
                    <h1>Source</h1>
                    <p>Prepare and manage experimental datasets.</p>
                </div>
            </aside>

            <section class="route-canvas route-canvas-source">
                <div class="route-tabs-row" aria-label="Source navigation header">
                    <adsmod-header-tabs />
                </div>

                <div class="source-card-grid">
                    <section class="source-card source-card-primary" aria-label="Dataset source section">
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
                    </section>

                    <section class="source-card source-card-secondary" aria-label="NIST source section">
                        <div class="source-card-topline">
                            <div>
                                <div class="section-title">NIST-A Collection</div>
                                <div class="section-caption">
                                    Fetch NIST-A records into the local database using sampling fractions, then use NIST data to benchmark coverage before moving to fitting and training.
                                </div>
                            </div>
                        </div>
                        <adsmod-nist-collection-rows (statusUpdate)="store.setNistStatusMessage($event)" />
                    </section>

                    <section class="source-card source-card-tertiary" aria-label="Uploaded dataset statistics">
                        <div class="panel-title">Uploaded Data Statistics</div>
                        <div class="source-card-text">
                            <adsmod-markdown-renderer [content]="store.datasetStats()" />
                        </div>
                    </section>

                    <section class="source-card source-card-tertiary" aria-label="NIST status updates">
                        <div class="panel-title">NIST-A Status Updates</div>
                        <div class="source-card-text">
                            <adsmod-markdown-renderer [content]="store.nistStatusMessage()" />
                        </div>
                    </section>
                </div>
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

    protected async uploadSelectedFile(): Promise<void> {
        await this.store.uploadPendingDataset();
    }
}
