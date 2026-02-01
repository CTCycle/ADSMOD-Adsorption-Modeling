import React, { useState, useEffect, useMemo } from 'react';
import { DatasetProcessingWizard } from './DatasetProcessingWizard';
import {
    buildTrainingDataset,
    clearTrainingDataset,
    deleteDatasetSource,
    fetchDatasetSources,
    getTrainingDatasetInfo,
} from '../services';
import type {
    DatasetBuildConfig,
    DatasetFullInfo,
    DatasetSourceInfo,
} from '../types';

interface DatasetBuilderCardProps {
    onDatasetBuilt?: () => void;
}

const buildDatasetKey = (dataset: DatasetSourceInfo): string =>
    `${dataset.source}:${dataset.dataset_name}`;

export const DatasetBuilderCard: React.FC<DatasetBuilderCardProps> = ({ onDatasetBuilt }) => {
    // Dataset sources and selection state
    const [datasetSources, setDatasetSources] = useState<DatasetSourceInfo[]>([]);
    const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());

    // Wizard visibility
    const [isWizardOpen, setIsWizardOpen] = useState(false);

    // UI state
    const [isBuilding, setIsBuilding] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [statusTone, setStatusTone] = useState<'info' | 'success' | 'error'>('info');
    const [datasetInfo, setDatasetInfo] = useState<DatasetFullInfo | null>(null);
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
        // Clear selection on refresh
        setSelectedKeys(new Set());
        setStatusMessage(null);

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

    const handleBuildStart = async (config: DatasetBuildConfig) => {
        setIsBuilding(true);
        setJobProgress(0);
        setStatusTone('info');
        setStatusMessage('Building dataset...');

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

    const handleDeleteSourceDataset = async (dataset: DatasetSourceInfo) => {
        if (dataset.source !== 'uploaded') {
            return;
        }
        if (!window.confirm(`Are you sure you want to delete dataset '${dataset.display_name}'?`)) {
            return;
        }
        const { success, message } = await deleteDatasetSource(
            dataset.source,
            dataset.dataset_name
        );
        if (success) {
            await loadDatasetSources();
        } else {
            alert(`Failed to delete dataset: ${message}`);
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
                    <button className="secondary" onClick={loadDatasetSources}>
                        Refresh
                    </button>
                </div>
            </div>

            <div className="dataset-split-layout">
                {/* Left side: Dataset grid (60%) */}
                <div className="dataset-split-left">

                    <div className="dataset-table">
                        <div className="dataset-table-header">
                            <span>Name</span>
                            <span>Source</span>
                            <span>Rows</span>
                            <span className="dataset-actions-header">Actions</span>
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
                                        <span className="dataset-actions-cell">
                                            <button
                                                onClick={(event) => {
                                                    event.stopPropagation();
                                                    handleDeleteSourceDataset(dataset);
                                                }}
                                                title={
                                                    dataset.source === 'uploaded'
                                                        ? 'Delete Dataset'
                                                        : 'NIST datasets cannot be removed'
                                                }
                                                disabled={dataset.source !== 'uploaded'}
                                                style={{
                                                    background: 'none',
                                                    border: 'none',
                                                    cursor: dataset.source === 'uploaded' ? 'pointer' : 'not-allowed',
                                                    fontSize: '1.1rem',
                                                    padding: '4px',
                                                    lineHeight: 1,
                                                    opacity: dataset.source === 'uploaded' ? 1 : 0.4,
                                                }}
                                            >
                                                🗑️
                                            </button>
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                    <div className="dataset-selection-summary">
                        {selectedDatasets.length} selected
                    </div>
                </div>

                {/* Right side: Info + Button (40%) */}
                <div className="dataset-split-right">
                    <div className="dataset-info-text">
                        <p>
                            Select one or more datasets from the grid, then configure processing
                            settings to build a unified training dataset. The wizard will guide you
                            through sampling, filtering, and validation split options.
                            {datasetInfo?.available && (
                                <>
                                    <br /><br />
                                    <span className="text-sm text-slate-500 font-semibold">
                                        {' '}Currently: {datasetInfo.train_samples} training samples, {datasetInfo.validation_samples} validation samples.
                                    </span>
                                </>
                            )}
                        </p>
                    </div>
                    <div className="dataset-action-center">
                        <button
                            className="secondary"
                            onClick={handleClearDataset}
                            disabled={!datasetInfo?.available || isBuilding}
                            title="Clear current training dataset"
                        >
                            Clear
                        </button>
                        <button
                            className="primary"
                            onClick={() => setIsWizardOpen(true)}
                            disabled={selectedDatasets.length === 0 || isBuilding}
                        >
                            {isBuilding ? 'Building...' : 'Configure Processing'}
                        </button>
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

            {isWizardOpen && (
                <DatasetProcessingWizard
                    selectedDatasets={selectedDatasets}
                    onClose={() => setIsWizardOpen(false)}
                    onBuildStart={handleBuildStart}
                />
            )}
        </div>
    );
};

export default DatasetBuilderCard;
