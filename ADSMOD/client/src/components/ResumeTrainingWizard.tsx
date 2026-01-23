import React, { useState } from 'react';
import { NumberInput } from './UIComponents';
import type { CheckpointInfo, ResumeTrainingConfig } from '../types';

interface ResumeTrainingWizardProps {
    checkpoints: CheckpointInfo[];
    config: ResumeTrainingConfig;
    onConfigChange: (config: ResumeTrainingConfig) => void;
    onClose: () => void;
    onConfirm: () => void;
    isLoading: boolean;
}

const formatMetricValue = (value: number | null) => {
    if (value === null || Number.isNaN(value)) {
        return '--';
    }
    return value.toFixed(4);
};

const formatEpochs = (value: number | null) => {
    if (value === null || Number.isNaN(value)) {
        return '--';
    }
    return value.toString();
};

export const ResumeTrainingWizard: React.FC<ResumeTrainingWizardProps> = ({
    checkpoints,
    config,
    onConfigChange,
    onClose,
    onConfirm,
    isLoading,
}) => {
    const [currentPage, setCurrentPage] = useState(1);

    const updateConfig = <K extends keyof ResumeTrainingConfig>(key: K, value: ResumeTrainingConfig[K]) => {
        onConfigChange({ ...config, [key]: value });
    };

    const handleNext = () => {
        setCurrentPage((prev) => Math.min(prev + 1, 2));
    };

    const handlePrevious = () => {
        setCurrentPage((prev) => Math.max(prev - 1, 1));
    };

    const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
        if (event.target === event.currentTarget) {
            onClose();
        }
    };

    const selectedCheckpoint = checkpoints.find((checkpoint) => checkpoint.name === config.checkpoint_name) || null;
    const hasSelection = Boolean(selectedCheckpoint);
    const compatibilityLabel = selectedCheckpoint
        ? selectedCheckpoint.is_compatible
            ? 'Compatible'
            : 'Incompatible'
        : 'Not selected';

    return (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={handleBackdropClick}>
            <div className="wizard-modal">
                <div className="wizard-header">
                    <h4>Resume Training Wizard</h4>
                    <p>Select a checkpoint and configure additional epochs before resuming training.</p>
                    <div className="wizard-page-indicator">
                        <span className={`wizard-dot ${currentPage === 1 ? 'active' : ''}`}>1</span>
                        <span className={`wizard-dot-line ${currentPage > 1 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 2 ? 'active' : ''}`}>2</span>
                    </div>
                </div>

                <div className="wizard-body">
                    {currentPage === 1 && (
                        <div className="wizard-page">
                            <div className="wizard-card">
                                <div className="wizard-card-header">
                                    <span className="wizard-card-icon">CP</span>
                                    <span>Checkpoint Selection</span>
                                </div>
                                <p className="wizard-card-description">
                                    Choose a saved checkpoint and set the number of additional epochs.
                                </p>
                                <div className="wizard-card-body">
                                    <div className="wizard-settings-grid">
                                        <NumberInput
                                            label="Additional Epochs"
                                            value={config.additional_epochs}
                                            onChange={(value) => updateConfig('additional_epochs', value)}
                                            min={1}
                                            max={100}
                                            step={1}
                                            precision={0}
                                        />
                                    </div>
                                    <div className="checkpoint-list" role="listbox" aria-label="Checkpoint list">
                                        {checkpoints.length === 0 && (
                                            <div className="checkpoint-empty">No checkpoints available.</div>
                                        )}
                                        {checkpoints.map((checkpoint) => {
                                            const isSelected = checkpoint.name === config.checkpoint_name;
                                            return (
                                                <div
                                                    key={checkpoint.name}
                                                    className={`checkpoint-row ${isSelected ? 'selected' : ''}`}
                                                    role="option"
                                                    aria-selected={isSelected}
                                                    tabIndex={0}
                                                    onClick={() => updateConfig('checkpoint_name', checkpoint.name)}
                                                    onKeyDown={(event) => {
                                                        if (event.key === 'Enter' || event.key === ' ') {
                                                            event.preventDefault();
                                                            updateConfig('checkpoint_name', checkpoint.name);
                                                        }
                                                    }}
                                                >
                                                    <div className="checkpoint-main">
                                                        <span
                                                            className={`led-indicator ${
                                                                checkpoint.is_compatible ? 'compatible' : 'incompatible'
                                                            }`}
                                                        />
                                                        <span className="checkpoint-name">{checkpoint.name}</span>
                                                    </div>
                                                    <div className="checkpoint-stats">
                                                        <div className="checkpoint-stat">
                                                            <span>Epochs</span>
                                                            <strong>{formatEpochs(checkpoint.epochs_trained)}</strong>
                                                        </div>
                                                        <div className="checkpoint-stat">
                                                            <span>Loss</span>
                                                            <strong>{formatMetricValue(checkpoint.final_loss)}</strong>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 2 && (
                        <div className="wizard-page">
                            <div className="wizard-summary">
                                <div className="wizard-summary-section">
                                    <h5>Resume Configuration</h5>
                                    <div className="wizard-summary-grid">
                                        <span>Checkpoint</span>
                                        <strong>{selectedCheckpoint?.name || 'Not selected'}</strong>
                                        <span>Additional epochs</span>
                                        <strong>{config.additional_epochs}</strong>
                                        <span>Compatibility</span>
                                        <strong>{compatibilityLabel}</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="wizard-footer">
                    <button className="secondary" onClick={onClose} disabled={isLoading}>
                        Cancel
                    </button>
                    {currentPage > 1 && (
                        <button className="secondary" onClick={handlePrevious} disabled={isLoading}>
                            Previous
                        </button>
                    )}
                    {currentPage < 2 && (
                        <button className="primary" onClick={handleNext} disabled={!hasSelection || isLoading}>
                            Next
                        </button>
                    )}
                    {currentPage === 2 && (
                        <button className="primary" onClick={onConfirm} disabled={!hasSelection || isLoading}>
                            {isLoading ? 'Resuming...' : 'Confirm Resume'}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ResumeTrainingWizard;
