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
                {/* First Row: Dataset Upload */}
                <div className="config-row">
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

                    <div className="config-row-center">
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

                    <div className="config-row-right">
                        <div className="panel dataset-panel" style={{ height: '100%' }}>
                            <div className="panel-header">
                                <div>
                                    <div className="panel-title">Dataset Statistics</div>
                                    <div className="panel-subtitle">
                                        {datasetBadge} | {sampleBadge}
                                    </div>
                                </div>
                            </div>
                            <div className="panel-body stats-scroll markdown-content compact-stats">
                                <MarkdownRenderer content={datasetStats} />
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
                        <NistStatusIndicator onStatusUpdate={onNistStatusUpdate} />
                    </div>

                    <div className="config-row-center">
                        <NistCollectCard onStatusUpdate={onNistStatusUpdate} />
                    </div>

                    <div className="config-row-right">
                        <NistPropertiesCard onStatusUpdate={onNistStatusUpdate} />
                    </div>
                </div>
            </div>
        </div>
    );
};

/** 
 * Standalone LED status indicator for the left column.
 * Fetches status independently to show availability.
 */
const NistStatusIndicator: React.FC<{ onStatusUpdate: (msg: string) => void }> = () => {
    const [dataAvailable, setDataAvailable] = React.useState(false);
    const [isLoading, setIsLoading] = React.useState(true);
    const [hasError, setHasError] = React.useState(false);

    React.useEffect(() => {
        const checkStatus = async () => {
            try {
                const response = await fetch('/api/nist/status');
                if (response.ok) {
                    const data = await response.json();
                    setDataAvailable(Boolean(data.data_available));
                    setHasError(false);
                } else {
                    setHasError(true);
                }
            } catch {
                setHasError(true);
            } finally {
                setIsLoading(false);
            }
        };
        void checkStatus();
    }, []);

    let statusLabel = 'Not ready';
    if (isLoading) {
        statusLabel = 'Checking';
    } else if (hasError) {
        statusLabel = 'Unavailable';
    } else if (dataAvailable) {
        statusLabel = 'Ready';
    }

    return (
        <div className="nist-status-footer" style={{ marginTop: 'auto' }}>
            <div className="nist-status-indicator">
                <span className={`nist-status-led ${dataAvailable ? 'available' : 'unavailable'}`} />
                <span className="nist-status-label">{statusLabel}</span>
            </div>
        </div>
    );
};
