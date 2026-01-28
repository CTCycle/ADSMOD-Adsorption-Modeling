
import React, { useState, useEffect } from 'react';
import { NumberInput } from './UIComponents';
import type { CheckpointInfo, ResumeTrainingConfig } from '../types';

interface ResumeTrainingWizardProps {
    checkpoints: CheckpointInfo[];
    config: ResumeTrainingConfig;
    onConfigChange: (config: ResumeTrainingConfig) => void;
    onClose: () => void;
    onConfirm: () => void;
    isLoading: boolean;
    selectedCheckpointName: string; // Now required and pre-selected
}

export const ResumeTrainingWizard: React.FC<ResumeTrainingWizardProps> = ({
    checkpoints,
    config,
    onConfigChange,
    onClose,
    onConfirm,
    isLoading,
    selectedCheckpointName,
}) => {
    const [currentPage, setCurrentPage] = useState(0);

    // Ensure the checkpoint name is set in the config on mount
    useEffect(() => {
        if (config.checkpoint_name !== selectedCheckpointName) {
            onConfigChange({ ...config, checkpoint_name: selectedCheckpointName });
        }
    }, [selectedCheckpointName, config, onConfigChange]);

    const updateConfig = <K extends keyof ResumeTrainingConfig>(key: K, value: ResumeTrainingConfig[K]) => {
        onConfigChange({ ...config, [key]: value });
    };

    const handleNext = () => {
        setCurrentPage((prev) => Math.min(prev + 1, 1));
    };

    const handlePrevious = () => {
        setCurrentPage((prev) => Math.max(prev - 1, 0));
    };

    const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
        if (event.target === event.currentTarget) {
            onClose();
        }
    };

    const selectedCheckpoint = checkpoints.find((checkpoint) => checkpoint.name === selectedCheckpointName) || null;
    const compatibilityLabel = selectedCheckpoint
        ? selectedCheckpoint.is_compatible
            ? 'Compatible'
            : 'Incompatible'
        : 'Unknown';

    return (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={handleBackdropClick}>
            <div className="wizard-modal">
                <div className="wizard-header">
                    <h4>Resume Training Wizard</h4>
                    <p>Resuming from checkpoint: <strong>{selectedCheckpointName}</strong></p>
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
                                    <span className="wizard-card-icon">CP</span>
                                    <span>Configuration</span>
                                </div>
                                <p className="wizard-card-description">
                                    Set the number of additional epochs to train.
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
                                    <div style={{ marginTop: '20px', padding: '15px', backgroundColor: 'var(--slate-50)', borderRadius: '8px' }}>
                                        <strong>Checkpoint Details:</strong>
                                        <ul style={{ listStyle: 'none', padding: 0, marginTop: '10px' }}>
                                            <li>Compatibility: {compatibilityLabel}</li>
                                            <li>Epochs Trained: {selectedCheckpoint?.epochs_trained ?? '--'}</li>
                                            <li>Final Loss: {selectedCheckpoint?.final_loss?.toFixed(4) ?? '--'}</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 1 && (
                        <div className="wizard-page">
                            <div className="wizard-summary">
                                <div className="wizard-summary-section">
                                    <h5>Resume Configuration</h5>
                                    <div className="wizard-summary-grid">
                                        <span>Checkpoint</span>
                                        <strong>{selectedCheckpointName}</strong>
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
                    {currentPage > 0 && (
                        <button className="secondary" onClick={handlePrevious} disabled={isLoading}>
                            Previous
                        </button>
                    )}
                    {currentPage < 1 && (
                        <button className="primary" onClick={handleNext} disabled={isLoading}>
                            Next
                        </button>
                    )}
                    {currentPage === 1 && (
                        <button className="primary" onClick={onConfirm} disabled={isLoading}>
                            {isLoading ? 'Resuming...' : 'Confirm Resume'}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ResumeTrainingWizard;
