import React from 'react';
import { FileUpload } from './UIComponents';
import { NistCollectCard, NistPropertiesCard, useNistStatus } from './CollectDataCard';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ConfigPageProps {
    datasetStats: string;
    nistStatusMessage: string;
    datasetName: string | null;
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
    datasetSamples,
    pendingFileName,
    pendingFileSize,
    onDatasetPreload,
    onDatasetUpload,
    isDatasetUploading,
    onNistStatusUpdate,
}) => {
    const nistStatusState = useNistStatus();
    const datasetBadge = datasetName || 'No dataset loaded';
    const sampleBadge = datasetSamples > 0 ? `${datasetSamples} samples` : '0 samples';
    const pendingLabel = pendingFileName ? `Selected: ${pendingFileName}` : 'No file selected';
    const pendingSize = pendingFileSize || '-- kb';

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
                        <div className="card split-card">
                            <div className="split-card-content">
                                <div className="split-card-left">
                                    <FileUpload
                                        label="Load dataset"
                                        accept=".csv,.xls,.xlsx"
                                        onSelect={onDatasetPreload}
                                        autoUpload={false}
                                        disabled={isDatasetUploading}
                                    />
                                    <div className="dataset-inline" style={{ marginTop: '0.75rem' }}>
                                        <span className="inline-pill">{pendingLabel}</span>
                                        <span className="inline-separator">|</span>
                                        <span className="inline-pill">{pendingSize}</span>
                                    </div>
                                    <div className="nist-actions" style={{ marginTop: '1rem' }}>
                                        <button
                                            className="button primary"
                                            onClick={onDatasetUpload}
                                            style={{ justifyContent: 'center' }}
                                            disabled={!pendingFileName || isDatasetUploading}
                                        >
                                            {isDatasetUploading ? 'Uploading...' : 'Upload data'}
                                        </button>
                                    </div>
                                </div>
                                <div className="split-card-right">
                                    <div className="panel-title split-card-title">Uploaded Data Statistics</div>
                                    <div className="panel-body stats-scroll markdown-content compact-stats">
                                        <MarkdownRenderer content={datasetStats} />
                                    </div>
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
                        <div className="config-row-stack">
                            <NistCollectCard onStatusUpdate={onNistStatusUpdate} nistStatusState={nistStatusState} />
                            <NistPropertiesCard onStatusUpdate={onNistStatusUpdate} nistStatusState={nistStatusState} />
                        </div>
                        <div className="panel nist-status-panel">
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
    );
};
