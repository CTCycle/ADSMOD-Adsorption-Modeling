import React from 'react';
import { NumberInput, FileUpload } from './UIComponents';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ControlsPanelProps {
    maxIterations: number;
    onMaxIterationsChange: (value: number) => void;
    datasetStats: string;
    fittingStatus: string;
    onDatasetUpload: (file: File) => void;
    onStartFitting: () => void;
}

export const ControlsPanel: React.FC<ControlsPanelProps> = ({
    maxIterations,
    onMaxIterationsChange,
    datasetStats,
    fittingStatus,
    onDatasetUpload,
    onStartFitting,
}) => {
    return (
        <div className="card" style={{ flex: 1, minWidth: '320px' }}>
            <div className="flex flex-col gap-4 w-full">
                <NumberInput
                    label="Max iterations"
                    value={maxIterations}
                    onChange={onMaxIterationsChange}
                    min={1}
                    max={1000000}
                    step={1}
                    precision={0}
                />

                <div>
                    <h4 style={{ marginBottom: '0.5rem', fontSize: '1rem', fontWeight: 600 }}>Dataset statistics</h4>
                    <div className="status-area markdown-content">
                        <MarkdownRenderer content={datasetStats} />
                    </div>
                </div>

                <div>
                    <label>Fitting status</label>
                    <textarea
                        value={fittingStatus}
                        readOnly
                        style={{ minHeight: '260px' }}
                        placeholder="Fitting status will appear here..."
                    />
                </div>

                <FileUpload label="Load dataset" accept=".csv,.xls,.xlsx" onUpload={onDatasetUpload} />

                <button className="primary" onClick={onStartFitting}>
                    Start fitting
                </button>
            </div>
        </div>
    );
};
