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
        <div className="section-container" style={{ marginTop: '20px' }}>
            <SplitSelectionCard
                title="Dataset Processing"
                subtitle="Compose training-ready data from your available sources."
                onRefresh={loadDatasetSources}
                leftContent={(
                    <div className="dataset-table" style={{ border: 'none', borderRadius: 0 }}>
                        <div className="dataset-table-header" style={{ position: 'sticky', top: 0, zIndex: 10, backgroundColor: 'white', borderBottom: '1px solid var(--slate-100)' }}>
                            <span>Name</span>
                            <span>Source</span>
                            <span>Rows</span>
                            <span className="dataset-actions-header">Actions</span>
                        </div>
                        <div className="dataset-table-body" style={{ maxHeight: 'none' }}>
                            {datasetSources.length === 0 && (
                                <div className="dataset-table-empty" style={{ padding: '40px' }}>
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
                                        style={{
                                            borderTop: 'none',
                                            borderBottom: '1px solid var(--slate-50)',
                                            backgroundColor: isSelected ? 'var(--primary-50, #eff6ff)' : 'transparent',
                                        }}
                                    >
                                        <span style={{ fontWeight: 500, color: 'var(--slate-700)' }}>{dataset.display_name}</span>
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
                        <div style={{ marginBottom: '16px', color: 'var(--primary-600)' }}>
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                                <path d="M12 12v9" />
                                <path d="m8 17 4 4 4-4" />
                            </svg>
                        </div>
                        <h4 style={{ margin: '0 0 8px 0', fontSize: '1.2rem', color: 'var(--slate-800)', fontWeight: 600 }}>Training Pipeline</h4>
                        <p style={{ margin: '0 0 24px 0', fontSize: '0.9rem', color: 'var(--slate-500)', lineHeight: '1.5' }}>
                            Select datasets to build a unified training source.
                            {datasetInfo?.available && (
                                <span style={{ display: 'block', marginTop: '8px', fontSize: '0.8rem', fontWeight: 600, color: 'var(--slate-400)' }}>
                                    Ready: {datasetInfo.train_samples}T / {datasetInfo.validation_samples}V
                                </span>
                            )}
                        </p>

                        <div style={{ width: '100%', marginBottom: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {selectedKeys.size > 0 ? (
                                <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '6px', border: '1px solid var(--primary-200)', textAlign: 'left' }}>
                                    <span style={{ fontSize: '0.7rem', color: 'var(--slate-400)', textTransform: 'uppercase', fontWeight: 600, display: 'block' }}>Selection</span>
                                    <div style={{ color: 'var(--primary-700)', fontWeight: 500, fontSize: '0.9rem' }}>{selectedKeys.size} datasets selected</div>
                                </div>
                            ) : (
                                <div style={{ height: '58px' }}></div>
                            )}
                        </div>

                        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            <button
                                className="primary"
                                onClick={() => setIsWizardOpen(true)}
                                disabled={selectedDatasets.length === 0 || isBuilding}
                                style={{
                                    width: '100%',
                                    padding: '10px',
                                    fontSize: '0.95rem',
                                    borderRadius: '6px',
                                    opacity: (selectedDatasets.length === 0 || isBuilding) ? 0.6 : 1,
                                    cursor: (selectedDatasets.length === 0 || isBuilding) ? 'not-allowed' : 'pointer'
                                }}
                            >
                                {isBuilding ? 'Building...' : 'Configure Processing'}
                            </button>
                            <button
                                className="secondary"
                                onClick={handleClearDataset}
                                disabled={!datasetInfo?.available || isBuilding}
                                title="Clear current training dataset"
                                style={{
                                    width: '100%',
                                    padding: '10px',
                                    fontSize: '0.9rem',
                                    borderRadius: '6px',
                                    border: '1px solid var(--slate-300)',
                                    backgroundColor: 'white',
                                    color: 'var(--slate-700)'
                                }}
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
