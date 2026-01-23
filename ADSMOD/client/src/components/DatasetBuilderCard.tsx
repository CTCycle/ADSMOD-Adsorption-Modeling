import React, { useState, useEffect, useMemo } from 'react';
import { NumberInput } from './UIComponents';
import {
    buildTrainingDataset,
    clearTrainingDataset,
    fetchDatasetSources,
    getTrainingDatasetInfo,
} from '../services';
import type {
    DatasetBuildConfig,
    DatasetFullInfo,
    DatasetSelection,
    DatasetSourceInfo,
} from '../types';

interface DatasetBuilderCardProps {
    onDatasetBuilt?: () => void;
}

const buildDatasetKey = (dataset: DatasetSourceInfo): string =>
    `${dataset.source}:${dataset.dataset_name}`;

export const DatasetBuilderCard: React.FC<DatasetBuilderCardProps> = ({ onDatasetBuilt }) => {
    // Build configuration state
    const [sampleSize, setSampleSize] = useState(1.0);
    const [validationSize, setValidationSize] = useState(0.2);
    const [minMeasurements, setMinMeasurements] = useState(1);
    const [maxMeasurements, setMaxMeasurements] = useState(30);
    const [smileSequenceSize, setSmileSequenceSize] = useState(20);
    const [maxPressure, setMaxPressure] = useState(10000);
    const [maxUptake, setMaxUptake] = useState(20);

    // Dataset sources and selection state
    const [datasetSources, setDatasetSources] = useState<DatasetSourceInfo[]>([]);
    const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());

    // UI state
    const [isBuilding, setIsBuilding] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [statusTone, setStatusTone] = useState<'info' | 'success' | 'error'>('info');
    const [datasetInfo, setDatasetInfo] = useState<DatasetFullInfo | null>(null);
    const [isReviewOpen, setIsReviewOpen] = useState(false);
    const [jobProgress, setJobProgress] = useState<number | null>(null);

    const selectedDatasets = useMemo(
        () => datasetSources.filter((dataset) => selectedKeys.has(buildDatasetKey(dataset))),
        [datasetSources, selectedKeys]
    );

    // Load initial dataset info and sources
    useEffect(() => {
        loadDatasetInfo();
        loadDatasetSources();
    }, []);

    const loadDatasetInfo = async () => {
        const info = await getTrainingDatasetInfo();
        setDatasetInfo(info);
    };

    const loadDatasetSources = async () => {
        const result = await fetchDatasetSources();
        if (result.error) {
            setDatasetSources([]);
            setStatusTone('error');
            setStatusMessage(`ERROR: ${result.error}`);
            return;
        }
        setDatasetSources(result.datasets);
        setStatusMessage(null);
    };

    const toggleDataset = (dataset: DatasetSourceInfo) => {
        const key = buildDatasetKey(dataset);
        setSelectedKeys((prev) => {
            const next = new Set(prev);
            if (next.has(key)) {
                next.delete(key);
            } else {
                next.add(key);
            }
            return next;
        });
    };

    const handleBuildDataset = async () => {
        if (selectedDatasets.length === 0) {
            setStatusTone('error');
            setStatusMessage('ERROR: Select at least one dataset before building.');
            return;
        }

        setIsReviewOpen(false);
        setIsBuilding(true);
        setJobProgress(0);
        setStatusTone('info');
        setStatusMessage('Starting dataset build job...');

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
        };

        const result = await buildTrainingDataset(config, (status) => {
            setJobProgress(status.progress);
        });

        if (result.success) {
            setStatusTone('success');
            setStatusMessage(
                `OK: ${result.message} (${result.train_samples} train, ${result.validation_samples} val)`
            );
            await loadDatasetInfo();
            onDatasetBuilt?.();
        } else {
            setStatusTone('error');
            setStatusMessage(`ERROR: ${result.message}`);
        }

        setIsBuilding(false);
        setJobProgress(null);
    };

    const handleClearDataset = async () => {
        const result = await clearTrainingDataset();
        if (result.success) {
            setStatusTone('info');
            setStatusMessage('Dataset cleared');
            setDatasetInfo(null);
            await loadDatasetInfo();
        } else {
            setStatusTone('error');
            setStatusMessage(`ERROR: ${result.message}`);
        }
    };

    return (
        <div className="dataset-processing-panel">
            <div className="dataset-processing-header">
                <div>
                    <h3>Dataset Processing</h3>
                    <p>Compose training-ready data from your available sources.</p>
                </div>
                <div className="dataset-processing-actions">
                    {datasetInfo?.available && (
                        <div className="dataset-current-info">
                            <span>Train: <strong>{datasetInfo.train_samples}</strong></span>
                            <span>Val: <strong>{datasetInfo.validation_samples}</strong></span>
                            <button
                                className="ghost-button"
                                onClick={handleClearDataset}
                                title="Clear dataset"
                            >
                                Clear
                            </button>
                        </div>
                    )}
                    <button className="secondary" onClick={loadDatasetSources}>
                        Refresh
                    </button>
                </div>
            </div>

            <div className="wizard-steps">
                <div className={`wizard-step ${selectedDatasets.length > 0 ? 'complete' : 'active'}`}>
                    <span className="wizard-index">1</span>
                    <span>Select datasets</span>
                </div>
                <div className="wizard-step">
                    <span className="wizard-index">2</span>
                    <span>Configure processing</span>
                </div>
                <div className="wizard-step">
                    <span className="wizard-index">3</span>
                    <span>Review and build</span>
                </div>
            </div>

            <div className="dataset-processing-grid">
                <div className="dataset-selection-panel">
                    <div className="panel-title">Available datasets</div>
                    <div className="dataset-table">
                        <div className="dataset-table-header">
                            <span>Name</span>
                            <span>Source</span>
                            <span>Rows</span>
                        </div>
                        <div className="dataset-table-body">
                            {datasetSources.length === 0 && (
                                <div className="dataset-table-empty">
                                    No datasets available yet.
                                </div>
                            )}
                            {datasetSources.map((dataset) => {
                                const key = buildDatasetKey(dataset);
                                const isSelected = selectedKeys.has(key);
                                return (
                                    <div
                                        key={key}
                                        className={`dataset-row ${isSelected ? 'selected' : ''}`}
                                        onClick={() => toggleDataset(dataset)}
                                        role="button"
                                        tabIndex={0}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter' || event.key === ' ') {
                                                toggleDataset(dataset);
                                            }
                                        }}
                                    >
                                        <span>{dataset.display_name}</span>
                                        <span className="dataset-source">{dataset.source}</span>
                                        <span className="dataset-count">{dataset.row_count}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="dataset-action-row">
                        <div className="dataset-action-text">
                            Builds a single training + validation split from the selected datasets.
                        </div>
                        <button
                            className="primary"
                            onClick={() => setIsReviewOpen(true)}
                            disabled={selectedDatasets.length === 0 || isBuilding}
                        >
                            {isBuilding ? 'Building...' : 'Build Dataset'}
                        </button>
                        <div className="dataset-selection-summary">
                            {selectedDatasets.length} selected
                        </div>
                    </div>
                </div>

                <div className="dataset-settings-panel">
                    <div className="panel-title">Processing settings</div>
                    <div className="dataset-settings-grid">
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

            {statusMessage && (
                <div className={`dataset-status ${statusTone}`}>
                    {statusMessage}
                    {isBuilding && jobProgress !== null && (
                        <span className="dataset-progress">{Math.round(jobProgress)}%</span>
                    )}
                </div>
            )}

            {isReviewOpen && (
                <div className="modal-backdrop" role="dialog" aria-modal="true">
                    <div className="modal-card">
                        <div className="modal-header">
                            <h4>Review dataset build</h4>
                            <p>Confirm your selections before starting the job.</p>
                        </div>
                        <div className="modal-section">
                            <h5>Selected datasets</h5>
                            <ul>
                                {selectedDatasets.map((dataset) => (
                                    <li key={buildDatasetKey(dataset)}>
                                        {dataset.display_name} ({dataset.source}, {dataset.row_count} rows)
                                    </li>
                                ))}
                            </ul>
                        </div>
                        <div className="modal-section">
                            <h5>Processing settings</h5>
                            <div className="modal-grid">
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
                        <div className="modal-actions">
                            <button className="secondary" onClick={() => setIsReviewOpen(false)}>
                                Go back
                            </button>
                            <button className="primary" onClick={handleBuildDataset} disabled={isBuilding}>
                                {isBuilding ? 'Creating...' : 'Create dataset'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DatasetBuilderCard;
