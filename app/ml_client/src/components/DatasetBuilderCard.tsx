import React, { useState, useEffect, useMemo } from 'react';
import { DatasetProcessingWizard } from './DatasetProcessingWizard';
import { SplitSelectionCard } from '../features/training/components/SplitSelectionCard';
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
    showSectionHeading?: boolean;
}

const buildDatasetKey = (dataset: DatasetSourceInfo): string =>
    `${dataset.source}:${dataset.dataset_name}`;

export const DatasetBuilderCard: React.FC<DatasetBuilderCardProps> = ({
    onDatasetBuilt,
    showSectionHeading = true,
}) => {
    const [datasetSources, setDatasetSources] = useState<DatasetSourceInfo[]>([]);
    const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());
    const [isWizardOpen, setIsWizardOpen] = useState(false);
    const [isBuilding, setIsBuilding] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [statusTone, setStatusTone] = useState<'info' | 'success' | 'error'>('info');
    const [datasetInfo, setDatasetInfo] = useState<DatasetFullInfo | null>(null);
    const [jobProgress, setJobProgress] = useState<number | null>(null);

    const selectedDatasets = useMemo(
        () => datasetSources.filter((dataset) => selectedKeys.has(buildDatasetKey(dataset))),
        [datasetSources, selectedKeys]
    );

    useEffect(() => {
        loadDatasetInfo();
        loadDatasetSources();
    }, []);

    const loadDatasetInfo = async () => {
        const info = await getTrainingDatasetInfo();
        setDatasetInfo(info);
    };

    const loadDatasetSources = async () => {
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
            setStatusTone('error');
            setStatusMessage(`ERROR: Failed to delete dataset: ${message}`);
        }
    };

    return (
        <div className="section-container">
            <SplitSelectionCard
                title="Dataset Processing"
                subtitle="Compose training-ready data from your available sources."
                onRefresh={loadDatasetSources}
                hideHeader={!showSectionHeading}
                leftContent={(
                    <div className="dataset-table dataset-table-flat">
                        <div className="dataset-table-header dataset-table-header-sticky">
                            <span>Name</span>
                            <span>Source</span>
                            <span>Rows</span>
                            <span className="dataset-actions-header">Actions</span>
                        </div>
                        <div className="dataset-table-body dataset-table-body-unbounded">
                            {datasetSources.length === 0 && (
                                <div className="dataset-table-empty dataset-table-empty-lg">
                                    No datasets available yet.
                                </div>
                            )}
                            {datasetSources.map((dataset) => {
                                const key = buildDatasetKey(dataset);
                                const isSelected = selectedKeys.has(key);
                                return (
                                    <div
                                        key={key}
                                        className={`dataset-row dataset-row-flat ${isSelected ? 'selected' : ''}`}
                                        onClick={() => toggleDataset(dataset)}
                                        role="button"
                                        tabIndex={0}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter' || event.key === ' ') {
                                                event.preventDefault();
                                                toggleDataset(dataset);
                                            }
                                        }}
                                    >
                                        <span className="dataset-name-cell">{dataset.display_name}</span>
                                        <span className="dataset-source">{dataset.source}</span>
                                        <span className="dataset-count">{dataset.row_count}</span>
                                        <span className="dataset-actions-cell">
                                            <button
                                                className="icon-action-button"
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
                                                type="button"
                                            >
                                                🗑️
                                            </button>
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
                rightContent={(
                    <>
                        <div className="split-selection-card-header-row">
                            <div className="split-selection-card-icon-wrap">
                                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                                    <path d="M12 12v9" />
                                    <path d="m8 17 4 4 4-4" />
                                </svg>
                            </div>
                            <h4 className="split-selection-card-title">Build Training Dataset</h4>
                        </div>

                        <p className="split-selection-card-description">
                            Merge uploaded and NIST adsorption data into a machine-learning-ready dataset.
                            {datasetInfo?.available && (
                                <span className="split-selection-card-ready-note">
                                    Ready: {datasetInfo.train_samples}T / {datasetInfo.validation_samples}V
                                </span>
                            )}
                        </p>

                        <div>
                            {selectedKeys.size > 0 ? (
                                <div className="split-selection-card-selection">
                                    <span className="split-selection-card-selection-label">Selection</span>
                                    <div className="split-selection-card-selection-value">{selectedKeys.size} datasets selected</div>
                                </div>
                            ) : (
                                <div className="split-selection-card-selection-placeholder"></div>
                            )}
                        </div>

                        <div className="split-selection-card-actions">
                            <button
                                className="primary split-selection-card-action-button"
                                onClick={() => setIsWizardOpen(true)}
                                disabled={selectedDatasets.length === 0 || isBuilding}
                                type="button"
                            >
                                {isBuilding ? 'Building...' : 'Configure Dataset Build'}
                            </button>
                            <button
                                className="secondary split-selection-card-action-button"
                                onClick={handleClearDataset}
                                disabled={!datasetInfo?.available || isBuilding}
                                title="Clear current training dataset"
                                type="button"
                            >
                                Clear Dataset
                            </button>
                        </div>
                    </>
                )}
            />

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
