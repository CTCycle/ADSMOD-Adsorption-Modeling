import React, { useState, useEffect } from 'react';
import { NumberInput } from './UIComponents';
import { fetchProcessedDatasets } from '../services';
import type {
    DatasetBuildConfig,
    DatasetSourceInfo,
    DatasetSelection,
    ProcessedDatasetInfo,
} from '../types';

interface DatasetProcessingWizardProps {
    selectedDatasets: DatasetSourceInfo[];
    onClose: () => void;
    onBuildStart: (config: DatasetBuildConfig) => void;
    onDatasetSelect?: (dataset: ProcessedDatasetInfo) => void;
}

const buildDatasetKey = (dataset: DatasetSourceInfo): string =>
    `${dataset.source}:${dataset.dataset_name}`;

export const DatasetProcessingWizard: React.FC<DatasetProcessingWizardProps> = ({
    selectedDatasets,
    onClose,
    onBuildStart,
    onDatasetSelect,
}) => {
    // Wizard page state (0: Select/Create, 1: Settings, 2: Summary + Naming)
    const [currentPage, setCurrentPage] = useState(0);

    // Processed datasets state
    const [processedDatasets, setProcessedDatasets] = useState<ProcessedDatasetInfo[]>([]);
    const [selectedProcessedDataset, setSelectedProcessedDataset] = useState<string | null>(null);
    const [loadingProcessed, setLoadingProcessed] = useState(true);

    // Build configuration state
    const [sampleSize, setSampleSize] = useState(1.0);
    const [validationSize, setValidationSize] = useState(0.2);
    const [minMeasurements, setMinMeasurements] = useState(1);
    const [maxMeasurements, setMaxMeasurements] = useState(30);
    const [smileSequenceSize, setSmileSequenceSize] = useState(20);
    const [maxPressure, setMaxPressure] = useState(10000);
    const [maxUptake, setMaxUptake] = useState(20);
    const [datasetName, setDatasetName] = useState('');

    // Load processed datasets on mount
    useEffect(() => {
        const loadProcessedDatasets = async () => {
            setLoadingProcessed(true);
            const { datasets, error } = await fetchProcessedDatasets();
            if (!error) {
                setProcessedDatasets(datasets);
            }
            setLoadingProcessed(false);
        };
        loadProcessedDatasets();
    }, []);

    // Generate default dataset name based on timestamp
    useEffect(() => {
        if (currentPage === 2 && !datasetName) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            setDatasetName(`dataset_${timestamp}`);
        }
    }, [currentPage, datasetName]);

    const handleSelectProcessedDataset = () => {
        const dataset = processedDatasets.find((d) => d.dataset_label === selectedProcessedDataset);
        if (dataset && onDatasetSelect) {
            onDatasetSelect(dataset);
            onClose();
        }
    };

    const handleCreateNew = () => {
        setCurrentPage(1);
    };

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
        setCurrentPage((prev) => Math.min(prev + 1, 2));
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
                    <p>Select an existing dataset or create a new one from your sources.</p>
                    <div className="wizard-page-indicator">
                        <span className={`wizard-dot ${currentPage === 0 ? 'active' : ''}`}>1</span>
                        <span className={`wizard-dot-line ${currentPage > 0 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 1 ? 'active' : ''}`}>2</span>
                        <span className={`wizard-dot-line ${currentPage > 1 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 2 ? 'active' : ''}`}>3</span>
                    </div>
                </div>

                <div className="wizard-body">
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-card">
                                <div className="wizard-card-header">
                                    <span className="wizard-card-icon">📂</span>
                                    <span>Select or Create Dataset</span>
                                </div>
                                <p className="wizard-card-description">
                                    Choose an existing processed dataset for training, or create a new one
                                    from the selected source datasets.
                                </p>
                                <div className="wizard-card-body">
                                    {loadingProcessed ? (
                                        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--slate-500)' }}>
                                            Loading processed datasets...
                                        </div>
                                    ) : processedDatasets.length === 0 ? (
                                        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--slate-500)' }}>
                                            No processed datasets available yet. Click "Create New" to build your first dataset.
                                        </div>
                                    ) : (
                                        <div className="dataset-table">
                                            <div className="dataset-table-header">
                                                <span>Name</span>
                                                <span>Train Samples</span>
                                                <span>Validation Samples</span>
                                            </div>
                                            <div className="dataset-table-body">
                                                {processedDatasets.map((dataset) => (
                                                    <div
                                                        key={dataset.dataset_label}
                                                        className={`dataset-row ${selectedProcessedDataset === dataset.dataset_label ? 'selected' : ''}`}
                                                        onClick={() => setSelectedProcessedDataset(dataset.dataset_label)}
                                                        role="button"
                                                        tabIndex={0}
                                                        onKeyDown={(event) => {
                                                            if (event.key === 'Enter' || event.key === ' ') {
                                                                setSelectedProcessedDataset(dataset.dataset_label);
                                                            }
                                                        }}
                                                    >
                                                        <span>{dataset.dataset_label}</span>
                                                        <span className="dataset-count">{dataset.train_samples}</span>
                                                        <span className="dataset-count">{dataset.validation_samples}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 1 && (
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

                    {currentPage === 2 && (
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
                    {currentPage === 0 ? (
                        <>
                            <button className="secondary" onClick={onClose}>
                                Cancel
                            </button>
                            {processedDatasets.length > 0 && (
                                <button
                                    className="secondary"
                                    onClick={handleSelectProcessedDataset}
                                    disabled={!selectedProcessedDataset}
                                >
                                    Use Selected
                                </button>
                            )}
                            <button className="primary" onClick={handleCreateNew}>
                                Create New →
                            </button>
                        </>
                    ) : currentPage === 1 ? (
                        <>
                            <button className="secondary" onClick={handlePrevious}>
                                ← Previous
                            </button>
                            <button className="primary" onClick={handleNext}>
                                Next →
                            </button>
                        </>
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
