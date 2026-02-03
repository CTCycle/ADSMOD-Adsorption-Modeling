
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
        <div className="training-setup-container" style={{ display: 'flex', flexDirection: 'column', gap: '40px', marginTop: '20px', paddingBottom: '40px' }}>

            {/* 1. New Training Section */}
            <div className="section-container">
                {/* Section Header */}
                <h3 style={{ margin: '0 0 16px 0', fontSize: '1.25rem', color: 'var(--slate-800)', fontWeight: 600 }}>Available Datasets</h3>

                {/* Unified Card */}
                <div className="training-unified-card" style={{
                    display: 'flex',
                    backgroundColor: 'var(--white, #fff)',
                    borderRadius: '12px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    border: '1px solid var(--slate-200, #e2e8f0)',
                    overflow: 'hidden',
                    height: '400px' // Fixed total height for consistency
                }}>

                    {/* LEFT: Table Area (70%) */}
                    <div style={{ flex: '0 0 70%', display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--slate-200)', minWidth: 0 }}>
                        {/* Table Controls / Header */}
                        <div style={{
                            padding: '12px 16px',
                            display: 'flex',
                            justifyContent: 'flex-end',
                            borderBottom: '1px solid var(--slate-100)',
                            backgroundColor: 'var(--slate-50)'
                        }}>
                            {onRefreshDatasets && (
                                <button
                                    onClick={(e) => { e.stopPropagation(); onRefreshDatasets(); }}
                                    title="Refresh Datasets"
                                    style={{
                                        background: 'white',
                                        border: '1px solid var(--slate-300)',
                                        borderRadius: '4px',
                                        padding: '4px 8px',
                                        cursor: 'pointer',
                                        fontSize: '0.9rem',
                                        color: 'var(--slate-600)'
                                    }}
                                >
                                    🔄 Refresh
                                </button>
                            )}
                        </div>

                        {/* Scrollable Table Container */}
                        <div style={{ flex: 1, overflowY: 'auto', position: 'relative' }}>
                            <table style={{ width: '100%', tableLayout: 'fixed', borderCollapse: 'collapse' }}>
                                <thead style={{ position: 'sticky', top: 0, zIndex: 10, backgroundColor: 'white', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
                                    <tr style={{ color: 'var(--slate-500)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                        <th style={{ padding: '12px 16px', textAlign: 'left', width: '40%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Name</th>
                                        <th style={{ padding: '12px 16px', textAlign: 'left', width: '20%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Train</th>
                                        <th style={{ padding: '12px 16px', textAlign: 'left', width: '20%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Val</th>
                                        <th style={{ padding: '12px 16px', textAlign: 'right', width: '20%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {processedDatasets.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} style={{ padding: '40px', textAlign: 'center', color: 'var(--slate-400)' }}>
                                                No datasets available.
                                            </td>
                                        </tr>
                                    ) : (
                                        processedDatasets.map((ds) => (
                                            <tr
                                                key={ds.dataset_label}
                                                onClick={() => handleDatasetRowClick(ds.dataset_label)}
                                                style={{
                                                    cursor: 'pointer',
                                                    backgroundColor: selectedDatasetLabel === ds.dataset_label ? 'var(--primary-50, #eff6ff)' : 'transparent',
                                                    borderBottom: '1px solid var(--slate-50)',
                                                    transition: 'background-color 0.1s'
                                                }}
                                            >
                                                <td style={{ padding: '10px 16px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem', color: 'var(--slate-700)', fontWeight: 500 }}>
                                                    {ds.dataset_label}
                                                </td>
                                                <td style={{ padding: '10px 16px', fontSize: '0.9rem', color: 'var(--slate-600)' }}>
                                                    {ds.train_samples}
                                                </td>
                                                <td style={{ padding: '10px 16px', fontSize: '0.9rem', color: 'var(--slate-600)' }}>
                                                    {ds.validation_samples}
                                                </td>
                                                <td style={{ padding: '10px 16px', textAlign: 'right' }}>
                                                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onViewDatasetMetadata(ds.dataset_label); }}
                                                            title="View Metadata"
                                                            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', padding: '4px', lineHeight: 1 }}
                                                        >
                                                            ℹ️
                                                        </button>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onDeleteDataset(ds.dataset_label); }}
                                                            title="Delete Dataset"
                                                            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', padding: '4px', lineHeight: 1 }}
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
                    <div style={{ flex: '0 0 30%', backgroundColor: 'var(--slate-50)', padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                        <div style={{ marginBottom: '16px', color: 'var(--primary-600)' }}>
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
                        <h4 style={{ margin: '0 0 8px 0', fontSize: '1.2rem', color: 'var(--slate-800)', fontWeight: 600 }}>Initialize New Training Session</h4>
                        <p style={{ margin: '0 0 24px 0', fontSize: '0.9rem', color: 'var(--slate-500)', lineHeight: '1.5', maxWidth: '80%' }}>
                            Select a dataset from the list to configure the parameters for a new machine learning model training run.
                        </p>

                        {selectedDatasetLabel ? (
                            <div style={{ width: '100%', marginBottom: '16px', padding: '10px', backgroundColor: 'white', borderRadius: '6px', border: '1px solid var(--primary-200)', textAlign: 'left' }}>
                                <span style={{ fontSize: '0.7rem', color: 'var(--slate-400)', textTransform: 'uppercase', fontWeight: 600, display: 'block' }}>Selected</span>
                                <div style={{ color: 'var(--primary-700)', fontWeight: 500, fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{selectedDatasetLabel}</div>
                            </div>
                        ) : (
                            <div style={{ width: '100%', height: '58px', marginBottom: '16px' }}></div>
                        )}

                        <button
                            className="primary"
                            onClick={() => selectedDatasetLabel && onNewTrainingClick(selectedDatasetLabel)}
                            disabled={!isNewTrainingReady}
                            style={{
                                width: '100%',
                                padding: '10px',
                                fontSize: '0.95rem',
                                borderRadius: '6px',
                                opacity: isNewTrainingReady ? 1 : 0.6,
                                cursor: isNewTrainingReady ? 'pointer' : 'not-allowed'
                            }}
                        >
                            Configure Training
                        </button>
                    </div>
                </div>
            </div>

            {/* 2. Resume Training Section */}
            <div className="section-container">
                {/* Section Header */}
                <h3 style={{ margin: '0 0 16px 0', fontSize: '1.25rem', color: 'var(--slate-800)', fontWeight: 600 }}>Available Checkpoints</h3>

                {/* Unified Card */}
                <div className="training-unified-card" style={{
                    display: 'flex',
                    backgroundColor: 'var(--white, #fff)',
                    borderRadius: '12px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    border: '1px solid var(--slate-200, #e2e8f0)',
                    overflow: 'hidden',
                    height: '400px'
                }}>

                    {/* LEFT: Table Area (70%) */}
                    <div style={{ flex: '0 0 70%', display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--slate-200)', minWidth: 0 }}>
                        {/* Table Controls */}
                        <div style={{
                            padding: '12px 16px',
                            display: 'flex',
                            justifyContent: 'flex-end',
                            borderBottom: '1px solid var(--slate-100)',
                            backgroundColor: 'var(--slate-50)'
                        }}>
                            {onRefreshCheckpoints && (
                                <button
                                    onClick={(e) => { e.stopPropagation(); onRefreshCheckpoints(); }}
                                    title="Refresh Checkpoints"
                                    style={{
                                        background: 'white',
                                        border: '1px solid var(--slate-300)',
                                        borderRadius: '4px',
                                        padding: '4px 8px',
                                        cursor: 'pointer',
                                        fontSize: '0.9rem',
                                        color: 'var(--slate-600)'
                                    }}
                                >
                                    🔄 Refresh
                                </button>
                            )}
                        </div>

                        {/* Scrollable Table Container */}
                        <div style={{ flex: 1, overflowY: 'auto', position: 'relative' }}>
                            <table style={{ width: '100%', tableLayout: 'fixed', borderCollapse: 'collapse' }}>
                                <thead style={{ position: 'sticky', top: 0, zIndex: 10, backgroundColor: 'white', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
                                    <tr style={{ color: 'var(--slate-500)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                        <th style={{ padding: '12px 16px', textAlign: 'left', width: '40%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Name</th>
                                        <th style={{ padding: '12px 16px', textAlign: 'left', width: '20%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Epochs</th>
                                        <th style={{ padding: '12px 16px', textAlign: 'left', width: '20%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Loss</th>
                                        <th style={{ padding: '12px 16px', textAlign: 'right', width: '20%', fontWeight: 600, borderBottom: '1px solid var(--slate-100)' }}>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {checkpoints.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} style={{ padding: '40px', textAlign: 'center', color: 'var(--slate-400)' }}>
                                                No checkpoints available.
                                            </td>
                                        </tr>
                                    ) : (
                                        checkpoints.map((cp) => (
                                            <tr
                                                key={cp.name}
                                                onClick={() => handleCheckpointRowClick(cp.name)}
                                                style={{
                                                    cursor: 'pointer',
                                                    backgroundColor: selectedCheckpointName === cp.name ? 'var(--primary-50, #eff6ff)' : 'transparent',
                                                    borderBottom: '1px solid var(--slate-50)',
                                                    transition: 'background-color 0.1s'
                                                }}
                                            >
                                                <td style={{ padding: '10px 16px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem', color: 'var(--slate-700)', fontWeight: 500 }}>
                                                    {cp.name}
                                                </td>
                                                <td style={{ padding: '10px 16px', fontSize: '0.9rem', color: 'var(--slate-600)' }}>
                                                    {cp.epochs_trained ?? '-'}
                                                </td>
                                                <td style={{ padding: '10px 16px', fontSize: '0.9rem', color: 'var(--slate-600)' }}>
                                                    {cp.final_loss?.toFixed(4) ?? '-'}
                                                </td>
                                                <td style={{ padding: '10px 16px', textAlign: 'right' }}>
                                                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onViewCheckpointDetails(cp.name); }}
                                                            title="View Details"
                                                            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', padding: '4px', lineHeight: 1 }}
                                                        >
                                                            ℹ️
                                                        </button>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); onDeleteCheckpoint(cp.name); }}
                                                            title="Delete Checkpoint"
                                                            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', padding: '4px', lineHeight: 1 }}
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
                    <div style={{ flex: '0 0 30%', backgroundColor: 'var(--slate-50)', padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📂</div>
                        <h4 style={{ margin: '0 0 8px 0', fontSize: '1.1rem', color: 'var(--slate-800)' }}>Resume Training</h4>
                        <p style={{ margin: '0 0 24px 0', fontSize: '0.85rem', color: 'var(--slate-500)', lineHeight: '1.4' }}>
                            Resume a previous training session from a checkpoint.
                        </p>

                        {selectedCheckpointName ? (
                            <div style={{ width: '100%', marginBottom: '16px', padding: '10px', backgroundColor: 'white', borderRadius: '6px', border: '1px solid var(--primary-200)', textAlign: 'left' }}>
                                <span style={{ fontSize: '0.7rem', color: 'var(--slate-400)', textTransform: 'uppercase', fontWeight: 600, display: 'block' }}>Selected</span>
                                <div style={{ color: 'var(--primary-700)', fontWeight: 500, fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{selectedCheckpointName}</div>
                            </div>
                        ) : (
                            <div style={{ width: '100%', height: '58px', marginBottom: '16px' }}></div>
                        )}

                        <button
                            className="secondary"
                            onClick={() => selectedCheckpointName && onResumeTrainingClick(selectedCheckpointName)}
                            disabled={!isResumeTrainingReady}
                            style={{
                                width: '100%',
                                padding: '10px',
                                fontSize: '0.95rem',
                                borderRadius: '6px',
                                opacity: isResumeTrainingReady ? 1 : 0.6,
                                cursor: isResumeTrainingReady ? 'pointer' : 'not-allowed',
                                border: '1px solid var(--slate-300)',
                                backgroundColor: 'white',
                                color: 'var(--slate-700)'
                            }}
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
