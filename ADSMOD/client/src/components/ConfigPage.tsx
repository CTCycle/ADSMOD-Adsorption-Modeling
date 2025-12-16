import React from 'react';
import { FileUpload } from './UIComponents';
import { CollectDataCard } from './CollectDataCard';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ConfigPageProps {
    datasetStats: string;
    datasetName: string | null;
    datasetSamples: number;
    onDatasetUpload: (file: File) => void;
}

export const ConfigPage: React.FC<ConfigPageProps> = ({
    datasetStats,
    datasetName,
    datasetSamples,
    onDatasetUpload,
}) => {
    const datasetBadge = datasetName || 'No dataset loaded';
    const sampleBadge = datasetSamples > 0 ? `${datasetSamples} samples` : '0 samples';

    return (
        <div className="config-page">
            <div className="config-grid-v2">
                <section className="controls-column">
                    <div className="form-stack">
                        <div className="section-heading">
                            <div className="section-title">Data Source</div>
                            <div className="section-caption">Load existing data or collect from NIST-A.</div>
                        </div>

                        <div className="field-block">
                            <div className="section-heading">
                                <div className="section-title">Load existing data</div>
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
                        </div>
                        <CollectDataCard />
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
