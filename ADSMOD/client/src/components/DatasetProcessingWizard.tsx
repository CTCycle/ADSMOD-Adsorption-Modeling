import React, { useState, useEffect } from 'react';
import { NumberInput } from './UIComponents';
import type {
    DatasetBuildConfig,
    DatasetSourceInfo,
    DatasetSelection,
} from '../types';

interface DatasetProcessingWizardProps {
    selectedDatasets: DatasetSourceInfo[];
    onClose: () => void;
    onBuildStart: (config: DatasetBuildConfig) => void;
}

const buildDatasetKey = (dataset: DatasetSourceInfo): string =>
    `${dataset.source}:${dataset.dataset_name}`;

export const DatasetProcessingWizard: React.FC<DatasetProcessingWizardProps> = ({
    selectedDatasets,
    onClose,
    onBuildStart,
}) => {
    // Wizard page state (0: Settings, 1: Summary + Naming)
    const [currentPage, setCurrentPage] = useState(0);

    // Build configuration state
    const [sampleSize, setSampleSize] = useState(1.0);
    const [validationSize, setValidationSize] = useState(0.2);
    const [minMeasurements, setMinMeasurements] = useState(1);
    const [maxMeasurements, setMaxMeasurements] = useState(30);
    const [smileSequenceSize, setSmileSequenceSize] = useState(20);
    const [maxPressure, setMaxPressure] = useState(10000);
    const [maxUptake, setMaxUptake] = useState(20);
    const [datasetName, setDatasetName] = useState('');

    // Generate default dataset name based on timestamp
    useEffect(() => {
        if (currentPage === 1 && !datasetName) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            setDatasetName(`dataset_${timestamp}`);
        }
    }, [currentPage, datasetName]);

    const handleBuildDataset = () => {
        if (selectedDatasets.length === 0) {
            return;
        }

        const datasets: DatasetSelection[] = selectedDatasets.map((dataset) => ({
            source: dataset.source,
            dataset_name: dataset.dataset_name,
        }));

        const config: DatasetBuildConfig = {
            sample_size: sampleSize,
            validation_size: validationSize,
            min_measurements: minMeasurements,
            max_measurements: maxMeasurements,
            smile_sequence_size: smileSequenceSize,
            max_pressure: maxPressure,
            max_uptake: maxUptake,
            datasets,
            dataset_label: datasetName || undefined,
        };

        // Close modal immediately and start build in background
        onClose();
        onBuildStart(config);
    };

    const handleNext = () => {
        setCurrentPage((prev) => Math.min(prev + 1, 1));
    };

    const handlePrevious = () => {
        setCurrentPage((prev) => Math.max(prev - 1, 0));
    };

    const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    return (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={handleBackdropClick}>
            <div className="wizard-modal">
                <div className="wizard-header">
                    <h4>Dataset Processing Wizard</h4>
                    <p>Configure processing settings for your selected datasets.</p>
                    <div className="wizard-page-indicator">
                        <span className={`wizard-dot ${currentPage === 0 ? 'active' : ''}`}>1</span>
                        <span className={`wizard-dot-line ${currentPage > 0 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 1 ? 'active' : ''}`}>2</span>
                    </div>
                </div>

                <div className="wizard-body">
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-card">
                                <div className="wizard-card-header">
                                    <span className="wizard-card-icon">⚙️</span>
                                    <span>Processing Settings</span>
                                </div>
                                <p className="wizard-card-description">
                                    Configure the parameters for dataset preprocessing. These settings control
                                    how the raw adsorption data is filtered, sampled, and split for training.
                                    Use the sample size to reduce dataset complexity during initial experiments,
                                    and adjust measurement bounds to filter outliers.
                                </p>
                                <div className="wizard-card-body">
                                    <div className="wizard-settings-grid">
                                        <NumberInput
                                            label="Sample Size"
                                            value={sampleSize}
                                            onChange={setSampleSize}
                                            min={0.01}
                                            max={1.0}
                                            step={0.01}
                                            precision={2}
                                        />
                                        <NumberInput
                                            label="Validation %"
                                            value={validationSize}
                                            onChange={setValidationSize}
                                            min={0.05}
                                            max={0.5}
                                            step={0.05}
                                            precision={2}
                                        />
                                        <NumberInput
                                            label="SMILE Length"
                                            value={smileSequenceSize}
                                            onChange={setSmileSequenceSize}
                                            min={5}
                                            max={100}
                                            step={5}
                                            precision={0}
                                        />
                                        <NumberInput
                                            label="Min Measurements"
                                            value={minMeasurements}
                                            onChange={setMinMeasurements}
                                            min={1}
                                            max={50}
                                            step={1}
                                            precision={0}
                                        />
                                        <NumberInput
                                            label="Max Measurements"
                                            value={maxMeasurements}
                                            onChange={setMaxMeasurements}
                                            min={5}
                                            max={500}
                                            step={5}
                                            precision={0}
                                        />
                                        <NumberInput
                                            label="Max Pressure (kPa)"
                                            value={maxPressure}
                                            onChange={setMaxPressure}
                                            min={100}
                                            max={100000}
                                            step={1000}
                                            precision={0}
                                        />
                                        <NumberInput
                                            label="Max Uptake (mol/g)"
                                            value={maxUptake}
                                            onChange={setMaxUptake}
                                            min={1}
                                            max={1000}
                                            step={1}
                                            precision={1}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 1 && (
                        <div className="wizard-page">
                            <div className="wizard-card" style={{ marginBottom: '1rem', border: '1px solid var(--primary-200)' }}>
                                <div className="wizard-card-header">
                                    <span className="wizard-card-icon">🏷️</span>
                                    <span>Dataset Name</span>
                                </div>
                                <div className="wizard-card-body">
                                    <div style={{ padding: '0.5rem 0' }}>
                                        <label className="field-label" style={{ marginBottom: '0.5rem', display: 'block' }}>
                                            Custom Name
                                        </label>
                                        <input
                                            type="text"
                                            value={datasetName}
                                            onChange={(e) => setDatasetName(e.target.value)}
                                            placeholder="e.g. my_dataset_v1"
                                            className="number-input-field"
                                            style={{
                                                width: '100%',
                                                textAlign: 'left',
                                                padding: '0.5rem 0.75rem',
                                                fontSize: '0.95rem',
                                                borderRadius: '8px',
                                                border: '1px solid var(--slate-300)',
                                                height: 'auto'
                                            }}
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="wizard-summary">
                                <div className="wizard-summary-section">
                                    <h5>Selected Datasets</h5>
                                    <ul>
                                        {selectedDatasets.map((dataset) => (
                                            <li key={buildDatasetKey(dataset)}>
                                                <strong>{dataset.display_name}</strong>
                                                <span className="wizard-summary-meta">
                                                    {dataset.source} • {dataset.row_count} rows
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="wizard-summary-section">
                                    <h5>Processing Settings</h5>
                                    <div className="wizard-summary-grid">
                                        <span>Sample size</span>
                                        <strong>{sampleSize}</strong>
                                        <span>Validation split</span>
                                        <strong>{validationSize}</strong>
                                        <span>SMILE length</span>
                                        <strong>{smileSequenceSize}</strong>
                                        <span>Min measurements</span>
                                        <strong>{minMeasurements}</strong>
                                        <span>Max measurements</span>
                                        <strong>{maxMeasurements}</strong>
                                        <span>Max pressure (kPa)</span>
                                        <strong>{maxPressure}</strong>
                                        <span>Max uptake (mol/g)</span>
                                        <strong>{maxUptake}</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="wizard-footer">
                    <button className="secondary" onClick={onClose}>
                        Cancel
                    </button>
                    {currentPage === 0 ? (
                        <button className="primary" onClick={handleNext}>
                            Next →
                        </button>
                    ) : (
                        <>
                            <button className="secondary" onClick={handlePrevious}>
                                ← Previous
                            </button>
                            <button className="primary" onClick={handleBuildDataset}>
                                ✓ Build Dataset
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DatasetProcessingWizard;
