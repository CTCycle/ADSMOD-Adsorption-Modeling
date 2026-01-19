import React from 'react';
import { FileUpload } from './UIComponents';
import { NistCollectCard, NistPropertiesCard } from './CollectDataCard';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ConfigPageProps {
    datasetStats: string;
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
    datasetName,
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
    const pendingLabel = pendingFileName ? `Selected: ${pendingFileName}` : 'No file selected';
    const pendingSize = pendingFileSize || '-- kb';

    return (
        <div className="config-page">
            <div className="config-rows">
                {/* First Row: Dataset Upload - Centered and Constrained */}
                <div className="config-row" style={{ justifyContent: 'center' }}>
                    <div className="config-row-info">
                        <div className="section-heading">
                            <div className="section-title">Load Experimental Data</div>
                            <div className="section-caption">
                                Upload adsorption data from local CSV or Excel files.
                                This data is stored separately from the NIST-A collection
                                and can be processed independently for model fitting.
                            </div>
                        </div>
                    </div>

                    <div className="config-row-center" style={{ flex: '0 0 auto', width: '100%', maxWidth: '600px' }}>
                        <div className="card">
                            <div className="card-content">
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
                        </div>
                    </div>
                    {/* Empty spacer to balance layout if needed, or just let center take space */}
                    <div className="config-row-right" style={{ visibility: 'hidden', height: 0 }} />
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

                    <div className="config-row-center">
                        <NistCollectCard onStatusUpdate={onNistStatusUpdate} />
                    </div>

                    <div className="config-row-right">
                        <NistPropertiesCard onStatusUpdate={onNistStatusUpdate} />
                    </div>
                </div>

                {/* Third Row: Shared Statistics */}
                <div className="config-row" style={{ marginTop: '2rem' }}>
                    <div className="config-row-info">
                        <div className="section-heading">
                            <div className="section-title">Data Verification</div>
                            <div className="section-caption">
                                Summary statistics and structure of the currently loaded dataset or NIST collection sample.
                            </div>
                            <div className="panel-subtitle" style={{ marginTop: '1rem' }}>
                                {datasetBadge} | {sampleBadge}
                            </div>
                        </div>
                    </div>

                    <div className="config-row-center" style={{ flexGrow: 2 }}>
                        <div className="panel dataset-panel" style={{ height: '100%', minHeight: '300px' }}>
                            <div className="panel-header">
                                <div className="panel-title">Dataset Statistics</div>
                            </div>
                            <div className="panel-body stats-scroll markdown-content compact-stats">
                                <MarkdownRenderer content={datasetStats} />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};


