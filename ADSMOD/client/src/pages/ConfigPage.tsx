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
        <div className="config-page">
            <div className="config-rows">
                {/* First Row: Dataset Upload + Data Statistics */}
                <div className="config-row">
                    <div className="config-row-info">
                        <div className="section-heading">
                            <div className="section-title">Load Experimental Data</div>
                            <div className="section-caption">
                                Upload adsorption data from local CSV or Excel files.
                                This data is stored separately from the NIST-A collection
                                and can be processed independently for model fitting.
                            </div>
                            <div className="panel-subtitle" style={{ marginTop: '1rem' }}>
                                {datasetBadge} | {sampleBadge}
                            </div>
                        </div>
                    </div>

                    <div className="config-row-main">
                        <div className="config-open-split dataset-open-layout">
                            <div className="dataset-open-left">
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
                                    >
                                        {isDatasetUploading ? 'Uploading...' : 'Upload'}
                                    </button>
                                </div>
                                <div className="dataset-inline dataset-upload-meta">
                                    <span className="inline-pill">Dataset: {datasetDisplayName}</span>
                                    <span className="inline-pill">Size: {datasetDisplaySize}</span>
                                </div>
                            </div>
                            <div className="dataset-open-right">
                                <div className="panel-title split-card-title">Uploaded Data Statistics</div>
                                <div className="dataset-stats-body stats-scroll markdown-content compact-stats">
                                    <MarkdownRenderer content={datasetStats} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Second Row: NIST Collection */}
                <div className="config-row">
                    <div className="config-row-info">
                        <div className="section-heading">
                            <div className="section-title">NIST-A Collection</div>
                            <div className="section-caption">
                                Fetch NIST-A isotherms and materials into the local database.
                                Use fractions to sample the catalog.
                            </div>
                        </div>
                    </div>

                    <div className="config-row-main">
                        <div className="nist-open-layout">
                            <div className="nist-open-left">
                                <NistCollectionRows onStatusUpdate={onNistStatusUpdate} />
                            </div>
                            <div className="panel nist-status-panel nist-open-right">
                                <div className="panel-header">
                                    <div className="panel-title">NIST-A Status Updates</div>
                                </div>
                                <div className="panel-body stats-scroll markdown-content nist-status-body">
                                    <MarkdownRenderer content={nistStatusMessage} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
