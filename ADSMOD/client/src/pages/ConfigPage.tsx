import React from 'react';
import { FileUpload } from '../components/UIComponents';
import { NistCollectionRows } from '../components/NistCollectionRows';
import { MarkdownRenderer } from '../components/MarkdownRenderer';

interface ConfigPageProps {
    datasetStats: string;
    nistStatusMessage: string;
    datasetName: string | null;
    datasetSizeKb: string | null;
    datasetSamples: number;
    pendingFileName: string | null;
    pendingFileSize: string | null;
    onDatasetPreload: (file: File) => void;
    onDatasetUpload: () => void;
    isDatasetUploading: boolean;
    onNistStatusUpdate: (message: string) => void;
}

export const ConfigPage: React.FC<ConfigPageProps> = ({
    datasetStats,
    nistStatusMessage,
    datasetName,
    datasetSizeKb,
    datasetSamples,
    pendingFileName,
    pendingFileSize,
    onDatasetPreload,
    onDatasetUpload,
    isDatasetUploading,
    onNistStatusUpdate,
}) => {
    const datasetBadge = datasetName || 'No dataset loaded';
    const sampleBadge = datasetSamples > 0 ? `${datasetSamples} samples` : '0 samples';
    const datasetDisplayName = datasetName || pendingFileName || 'N.A.';
    const datasetDisplaySize = datasetSizeKb || pendingFileSize || 'N.A.';

    return (
        <div className="source-page">
            <div className="source-columns">
                <section className="source-column" aria-label="Dataset source section">
                    <div className="source-column-row source-column-header">
                        <div className="section-title">Load Experimental Data</div>
                        <div className="section-caption">
                            Upload adsorption data from local CSV or Excel files.
                        </div>
                        <div className="section-caption section-caption-journey">
                            Load, validate, and prepare your baseline dataset before fitting and training.
                        </div>
                        <div className="source-inline-labels">
                            <span className="inline-pill">{datasetBadge}</span>
                            <span className="inline-pill">{sampleBadge}</span>
                        </div>
                    </div>

                    <div className="source-column-row source-column-widget">
                        <div className="dataset-upload-toolbar">
                            <FileUpload
                                label="Load dataset"
                                accept=".csv,.xls,.xlsx"
                                onSelect={onDatasetPreload}
                                autoUpload={false}
                                disabled={isDatasetUploading}
                            />
                            <button
                                className="button primary dataset-upload-button"
                                onClick={onDatasetUpload}
                                disabled={!pendingFileName || isDatasetUploading}
                                type="button"
                            >
                                {isDatasetUploading ? 'Uploading...' : 'Upload'}
                            </button>
                        </div>
                        <div className="source-inline-labels dataset-upload-meta">
                            <span className="inline-pill">Dataset: {datasetDisplayName}</span>
                            <span className="inline-pill">Size: {datasetDisplaySize}</span>
                        </div>
                    </div>

                    <div className="source-column-row source-column-log">
                        <div className="panel-title">Uploaded Data Statistics</div>
                        <div className="source-markdown-scroll markdown-content compact-stats">
                            <MarkdownRenderer content={datasetStats} />
                        </div>
                    </div>
                </section>

                <section className="source-column" aria-label="NIST source section">
                    <div className="source-column-row source-column-header">
                        <div className="section-title">NIST-A Collection</div>
                        <div className="section-caption">
                            Fetch NIST-A records into the local database using sampling fractions.
                        </div>
                        <div className="section-caption section-caption-journey">
                            Use NIST data to benchmark coverage before moving to fitting and training.
                        </div>
                    </div>

                    <div className="source-column-row source-column-widget">
                        <NistCollectionRows onStatusUpdate={onNistStatusUpdate} />
                    </div>

                    <div className="source-column-row source-column-log">
                        <div className="panel-title">NIST-A Status Updates</div>
                        <div className="source-markdown-scroll markdown-content">
                            <MarkdownRenderer content={nistStatusMessage} />
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
};
