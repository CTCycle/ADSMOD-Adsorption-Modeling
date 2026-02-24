
import React, { useState } from 'react';
import { ProcessedDatasetInfo, CheckpointInfo } from '../types';

interface TrainingSetupRowProps {
    onNewTrainingClick: (datasetLabel: string) => void;
    onResumeTrainingClick: (checkpointName: string) => void;
    datasetAvailable: boolean;
    checkpointsAvailable: boolean;
    processedDatasets: ProcessedDatasetInfo[];
    checkpoints: CheckpointInfo[];
    isTraining: boolean;
    onDeleteDataset: (label: string) => void;
    onViewDatasetMetadata: (label: string) => void;
    onDeleteCheckpoint: (name: string) => void;
    onViewCheckpointDetails: (name: string) => void;
    onRefreshDatasets?: () => void;
    onRefreshCheckpoints?: () => void;
}

export const TrainingSetupRow: React.FC<TrainingSetupRowProps> = ({
    onNewTrainingClick,
    onResumeTrainingClick,
    processedDatasets,
    checkpoints,
    isTraining,
    onDeleteDataset,
    onViewDatasetMetadata,
    onDeleteCheckpoint,
    onViewCheckpointDetails,
    onRefreshDatasets,
    onRefreshCheckpoints,
}) => {
    const [selectedDatasetLabel, setSelectedDatasetLabel] = useState<string | null>(null);
    const [selectedCheckpointName, setSelectedCheckpointName] = useState<string | null>(null);

    const handleDatasetRowClick = (label: string) => {
        setSelectedDatasetLabel(prev => prev === label ? null : label);
    };

    const handleCheckpointRowClick = (name: string) => {
        setSelectedCheckpointName(prev => prev === name ? null : name);
    };

    const isNewTrainingReady = selectedDatasetLabel !== null && !isTraining;
    const isResumeTrainingReady = selectedCheckpointName !== null && !isTraining;

    return (
        <div className="training-setup-container">

            {/* 1. New Training Section */}
            <div className="section-container">
                {/* Section Header */}
                <h3 className="split-selection-title">Available Datasets</h3>

                {/* Unified Card */}
                <div className="split-selection-card">

                    {/* LEFT: Table Area (70%) */}
                    <div className="split-selection-card-left">
                        {/* Table Controls / Header */}
                        <div className="split-selection-card-toolbar">
                            {onRefreshDatasets && (
                                <button
                                    onClick={(e) => { e.stopPropagation(); onRefreshDatasets(); }}
                                    title="Refresh Datasets"
                                    className="split-selection-refresh-button"
                                    type="button"
                                >
                                    🔄 Refresh
                                </button>
                            )}
                        </div>

                        {/* Scrollable Table Container */}
                        <div className="split-selection-card-content">
                            <table className="split-table">
                                <thead className="split-table-head">
                                    <tr className="split-table-header-row">
                                        <th className="split-table-header-cell col-name">Name</th>
                                        <th className="split-table-header-cell col-train">Train</th>
                                        <th className="split-table-header-cell col-val">Val</th>
                                        <th className="split-table-header-cell col-actions">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {processedDatasets.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} className="split-table-empty-cell">
                                                No datasets available.
                                            </td>
                                        </tr>
                                    ) : (
                                        processedDatasets.map((ds) => (
                                            <tr
                                                key={ds.dataset_label}
                                                onClick={() => handleDatasetRowClick(ds.dataset_label)}
                                                className={`split-table-row ${selectedDatasetLabel === ds.dataset_label ? 'selected' : ''}`}
                                                role="button"
                                                tabIndex={0}
                                                onKeyDown={(event) => {
                                                    if (event.key === 'Enter' || event.key === ' ') {
                                                        event.preventDefault();
                                                        handleDatasetRowClick(ds.dataset_label);
                                                    }
                                                }}
                                            >
                                                <td className="split-table-cell split-table-cell-strong split-table-cell-ellipsis">
                                                    {ds.dataset_label}
                                                </td>
                                                <td className="split-table-cell">
                                                    {ds.train_samples}
                                                </td>
                                                <td className="split-table-cell">
                                                    {ds.validation_samples}
                                                </td>
                                                <td className="split-table-cell split-table-cell-right">
                                                    <div className="split-table-actions-wrap">
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onViewDatasetMetadata(ds.dataset_label); }}
                                                            title="View Metadata"
                                                            className="icon-action-button"
                                                            type="button"
                                                        >
                                                            ℹ️
                                                        </button>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onDeleteDataset(ds.dataset_label); }}
                                                            title="Delete Dataset"
                                                            className="icon-action-button"
                                                            type="button"
                                                        >
                                                            🗑️
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* RIGHT: Action Area (30%) */}
                    <div className="split-selection-card-right">
                        <div className="split-selection-card-icon-wrap">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="3" />
                                <circle cx="6" cy="6" r="2" />
                                <circle cx="6" cy="18" r="2" />
                                <circle cx="18" cy="6" r="2" />
                                <circle cx="18" cy="18" r="2" />
                                <path d="M8 7.5L10.5 10" />
                                <path d="M8 16.5L10.5 14" />
                                <path d="M13.5 10L16 7.5" />
                                <path d="M13.5 14L16 16.5" />
                            </svg>
                        </div>
                        <h4 className="split-selection-card-title">Initialize New Training Session</h4>
                        <p className="split-selection-card-description split-selection-card-description-wide">
                            Select a dataset from the list to configure the parameters for a new machine learning model training run.
                        </p>

                        {selectedDatasetLabel ? (
                            <div className="split-selection-card-selection">
                                <span className="split-selection-card-selection-label">Selected</span>
                                <div className="split-selection-card-selection-value">{selectedDatasetLabel}</div>
                            </div>
                        ) : (
                            <div className="split-selection-card-selection-placeholder"></div>
                        )}

                        <button
                            className="primary split-selection-card-action-button"
                            onClick={() => selectedDatasetLabel && onNewTrainingClick(selectedDatasetLabel)}
                            disabled={!isNewTrainingReady}
                            type="button"
                        >
                            Configure Training
                        </button>
                    </div>
                </div>
            </div>

            {/* 2. Resume Training Section */}
            <div className="section-container">
                {/* Section Header */}
                <h3 className="split-selection-title">Available Checkpoints</h3>

                {/* Unified Card */}
                <div className="split-selection-card">

                    {/* LEFT: Table Area (70%) */}
                    <div className="split-selection-card-left">
                        {/* Table Controls */}
                        <div className="split-selection-card-toolbar">
                            {onRefreshCheckpoints && (
                                <button
                                    onClick={(e) => { e.stopPropagation(); onRefreshCheckpoints(); }}
                                    title="Refresh Checkpoints"
                                    className="split-selection-refresh-button"
                                    type="button"
                                >
                                    🔄 Refresh
                                </button>
                            )}
                        </div>

                        {/* Scrollable Table Container */}
                        <div className="split-selection-card-content">
                            <table className="split-table">
                                <thead className="split-table-head">
                                    <tr className="split-table-header-row">
                                        <th className="split-table-header-cell col-name">Name</th>
                                        <th className="split-table-header-cell col-epochs">Epochs</th>
                                        <th className="split-table-header-cell col-loss">Loss</th>
                                        <th className="split-table-header-cell col-actions">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {checkpoints.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} className="split-table-empty-cell">
                                                No checkpoints available.
                                            </td>
                                        </tr>
                                    ) : (
                                        checkpoints.map((cp) => (
                                            <tr
                                                key={cp.name}
                                                onClick={() => handleCheckpointRowClick(cp.name)}
                                                className={`split-table-row ${selectedCheckpointName === cp.name ? 'selected' : ''}`}
                                                role="button"
                                                tabIndex={0}
                                                onKeyDown={(event) => {
                                                    if (event.key === 'Enter' || event.key === ' ') {
                                                        event.preventDefault();
                                                        handleCheckpointRowClick(cp.name);
                                                    }
                                                }}
                                            >
                                                <td className="split-table-cell split-table-cell-strong split-table-cell-ellipsis">
                                                    {cp.name}
                                                </td>
                                                <td className="split-table-cell">
                                                    {cp.epochs_trained ?? '-'}
                                                </td>
                                                <td className="split-table-cell">
                                                    {cp.final_loss?.toFixed(4) ?? '-'}
                                                </td>
                                                <td className="split-table-cell split-table-cell-right">
                                                    <div className="split-table-actions-wrap">
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onViewCheckpointDetails(cp.name); }}
                                                            title="View Details"
                                                            className="icon-action-button"
                                                            type="button"
                                                        >
                                                            ℹ️
                                                        </button>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onDeleteCheckpoint(cp.name); }}
                                                            title="Delete Checkpoint"
                                                            className="icon-action-button"
                                                            type="button"
                                                        >
                                                            🗑️
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* RIGHT: Action Area (30%) */}
                    <div className="split-selection-card-right">
                        <div className="split-selection-card-emoji-icon">📂</div>
                        <h4 className="split-selection-card-title">Resume Training</h4>
                        <p className="split-selection-card-description">
                            Resume a previous training session from a checkpoint.
                        </p>

                        {selectedCheckpointName ? (
                            <div className="split-selection-card-selection">
                                <span className="split-selection-card-selection-label">Selected</span>
                                <div className="split-selection-card-selection-value">{selectedCheckpointName}</div>
                            </div>
                        ) : (
                            <div className="split-selection-card-selection-placeholder"></div>
                        )}

                        <button
                            className="secondary split-selection-card-action-button"
                            onClick={() => selectedCheckpointName && onResumeTrainingClick(selectedCheckpointName)}
                            disabled={!isResumeTrainingReady}
                            type="button"
                        >
                            Resume Training
                        </button>
                    </div>
                </div>
            </div>

        </div>
    );
};

export default TrainingSetupRow;
