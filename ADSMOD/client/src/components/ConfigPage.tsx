import React from 'react';
import { FileUpload } from './UIComponents';
import { CollectDataCard } from './CollectDataCard';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ConfigPageProps {
    datasetStats: string;
    datasetName: string | null;
    datasetSamples: number;
    onDatasetUpload: (file: File) => void;
    onNistStatusUpdate: (message: string) => void;
}

export const ConfigPage: React.FC<ConfigPageProps> = ({
    datasetStats,
    datasetName,
    datasetSamples,
    onDatasetUpload,
    onNistStatusUpdate,
}) => {
    const datasetBadge = datasetName || 'No dataset loaded';
    const sampleBadge = datasetSamples > 0 ? `${datasetSamples} samples` : '0 samples';

    return (
        <div className="config-page">
            <div className="config-grid-v2">
                <section className="controls-column">
                    <div className="form-stack">
                        <div className="field-block">
                            <div className="section-heading">
                                <div className="section-title">Load Experimental Data</div>
                                <div className="section-caption">
                                    Upload adsorption data from local CSV or Excel files.
                                    This data is stored separately from the NIST-A collection
                                    and can be processed independently for model fitting.
                                </div>
                            </div>
                            <FileUpload
                                label="Load dataset"
                                accept=".csv,.xls,.xlsx"
                                onUpload={onDatasetUpload}
                            />
                        </div>

                        <div className="dataset-inline">
                            <span className="inline-pill">{datasetBadge}</span>
                            <span className="inline-separator">|</span>
                            <span className="inline-pill">{sampleBadge}</span>
                        </div>

                        <div className="divider" />

                        <div className="section-heading">
                            <div className="section-title">NIST-A Collection</div>
                            <div className="section-caption">
                                Fetch NIST-A isotherms and materials into the local database. Use fractions to sample the catalog.
                            </div>
                        </div>
                        <CollectDataCard onStatusUpdate={onNistStatusUpdate} />
                    </div>
                </section>

                <section className="panels-column">
                    <div className="panel dataset-panel resizable-panel" style={{ height: '100%' }}>
                        <div className="panel-header">
                            <div>
                                <div className="panel-title">Dataset Statistics</div>
                                <div className="panel-subtitle">
                                    {datasetBadge} | {sampleBadge}
                                </div>
                            </div>
                        </div>
                        <div className="panel-body stats-scroll markdown-content">
                            <MarkdownRenderer content={datasetStats} />
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
};
