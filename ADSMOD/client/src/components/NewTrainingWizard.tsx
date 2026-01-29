
import React, { useState, useEffect } from 'react';
import { Checkbox, NumberInput, Switch } from './UIComponents';
import type { TrainingConfig } from '../types';

interface NewTrainingWizardProps {
    config: TrainingConfig;
    onConfigChange: (config: TrainingConfig) => void;
    onClose: () => void;
    onConfirm: () => void;
    isLoading: boolean;
    selectedDatasetLabel: string; // Now required and must be pre-selected
}

const getModelType = (value: string): TrainingConfig['selected_model'] => {
    return value === 'SCADS Atomic' ? 'SCADS Atomic' : 'SCADS Series';
};

export const NewTrainingWizard: React.FC<NewTrainingWizardProps> = ({
    config,
    onConfigChange,
    onClose,
    onConfirm,
    isLoading,
    selectedDatasetLabel,
}) => {
    const [currentPage, setCurrentPage] = useState(0);

    // Ensure the dataset label is set in the config on mount
    useEffect(() => {
        if (config.dataset_label !== selectedDatasetLabel) {
            onConfigChange({ ...config, dataset_label: selectedDatasetLabel });
        }
    }, [selectedDatasetLabel, config, onConfigChange]);

    const updateConfig = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
        onConfigChange({ ...config, [key]: value });
    };

    const handleNext = () => {
        setCurrentPage((prev) => Math.min(prev + 1, 3));
    };

    const handlePrevious = () => {
        setCurrentPage((prev) => Math.max(prev - 1, 0));
    };

    const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
        if (event.target === event.currentTarget) {
            onClose();
        }
    };

    return (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={handleBackdropClick}>
            <div className="wizard-modal">
                <div className="wizard-header">
                    <h4>New Training Wizard</h4>
                    <p>Configure training settings for dataset: <strong>{selectedDatasetLabel}</strong></p>
                    <div className="wizard-page-indicator">
                        <span className={`wizard-dot ${currentPage === 0 ? 'active' : ''}`}>1</span>
                        <span className={`wizard-dot-line ${currentPage > 0 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 1 ? 'active' : ''}`}>2</span>
                        <span className={`wizard-dot-line ${currentPage > 1 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 2 ? 'active' : ''}`}>3</span>
                        <span className={`wizard-dot-line ${currentPage > 2 ? 'active' : ''}`} />
                        <span className={`wizard-dot ${currentPage === 3 ? 'active' : ''}`}>4</span>
                    </div>
                </div>

                <div className="wizard-body">
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-card">
                                <div className="wizard-card-header">
                                    <span className="wizard-card-icon">📊</span>
                                    <span>Dataset Configuration</span>
                                </div>
                                <p className="wizard-card-description">
                                    Control how the training dataset is shuffled during training.
                                    Use buffer size to control memory usage.
                                </p>
                                <div className="wizard-card-body">
                                    <div className="wizard-toggle-row">
                                        <div style={{ flex: 1, minWidth: '140px' }}>
                                            <label>Shuffle Buffered</label>
                                            <div style={{ height: '42px', display: 'flex', alignItems: 'center' }}>
                                                <Switch
                                                    checked={config.shuffle_dataset}
                                                    onChange={(value) => updateConfig('shuffle_dataset', value)}
                                                />
                                            </div>
                                        </div>
                                        {config.shuffle_dataset && (
                                            <NumberInput
                                                label="Buffer Size"
                                                value={config.shuffle_size}
                                                onChange={(value) => updateConfig('shuffle_size', value)}
                                                min={100}
                                                max={10000}
                                                step={100}
                                                precision={0}
                                            />
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}


                    {
                        currentPage === 1 && (
                            <div className="wizard-page">
                                <div className="wizard-card">
                                    <div className="wizard-card-header">
                                        <span className="wizard-card-icon">🧠</span>
                                        <span>Model Configuration</span>
                                    </div>
                                    <p className="wizard-card-description">
                                        Define the architecture and embedding dimensions for the SCADS model
                                        family. Higher values improve capacity but increase training cost.
                                    </p>
                                    <div className="wizard-card-body">
                                        <div className="wizard-settings-grid">
                                            <NumberInput
                                                label="Encoders"
                                                value={config.num_encoders}
                                                onChange={(value) => updateConfig('num_encoders', value)}
                                                min={1}
                                                max={12}
                                                step={1}
                                                precision={0}
                                            />
                                            <NumberInput
                                                label="Attention Heads"
                                                value={config.num_attention_heads}
                                                onChange={(value) => updateConfig('num_attention_heads', value)}
                                                min={1}
                                                max={16}
                                                step={1}
                                                precision={0}
                                            />
                                            <NumberInput
                                                label="Embedding Dims"
                                                value={config.molecular_embedding_size}
                                                onChange={(value) => updateConfig('molecular_embedding_size', value)}
                                                min={64}
                                                max={1024}
                                                step={64}
                                                precision={0}
                                            />
                                            <NumberInput
                                                label="Dropout Rate"
                                                value={config.dropout_rate}
                                                onChange={(value) => updateConfig('dropout_rate', value)}
                                                min={0}
                                                max={0.5}
                                                step={0.05}
                                                precision={2}
                                            />
                                            <div style={{ minWidth: '160px', gridColumn: 'span 2', width: '100%' }}>
                                                <label className="field-label">Model Type</label>
                                                <select
                                                    value={config.selected_model}
                                                    onChange={(event) =>
                                                        updateConfig('selected_model', getModelType(event.target.value))
                                                    }
                                                    className="select-input"
                                                    style={{ width: '100%' }}
                                                >
                                                    <option value="SCADS Series">SCADS Series</option>
                                                    <option value="SCADS Atomic">SCADS Atomic</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )
                    }

                    {
                        currentPage === 2 && (
                            <div className="wizard-page">
                                <div className="wizard-card">
                                    <div className="wizard-card-header">
                                        <span className="wizard-card-icon">⚙️</span>
                                        <span>Training Configuration</span>
                                    </div>
                                    <p className="wizard-card-description">
                                        Define the training schedule, checkpointing, and learning rate behavior.
                                        Scheduler settings apply only when the LR scheduler is enabled.
                                    </p>
                                    <div className="wizard-card-body">
                                        <div className="wizard-settings-grid">
                                            <NumberInput
                                                label="Epochs"
                                                value={config.epochs}
                                                onChange={(value) => updateConfig('epochs', value)}
                                                min={1}
                                                max={500}
                                                step={1}
                                                precision={0}
                                            />
                                            <NumberInput
                                                label="Batch Size"
                                                value={config.batch_size}
                                                onChange={(value) => updateConfig('batch_size', value)}
                                                min={1}
                                                max={256}
                                                step={1}
                                                precision={0}
                                            />
                                            <div className="wizard-toggle-column">
                                                <Checkbox
                                                    label="Use GPU"
                                                    checked={config.use_device_GPU}
                                                    onChange={(value) => updateConfig('use_device_GPU', value)}
                                                />
                                                <Checkbox
                                                    label="Save Checkpoints"
                                                    checked={config.save_checkpoints}
                                                    onChange={(value) => updateConfig('save_checkpoints', value)}
                                                />
                                                <Checkbox
                                                    label="LR Scheduler"
                                                    checked={config.use_lr_scheduler}
                                                    onChange={(value) => updateConfig('use_lr_scheduler', value)}
                                                />
                                            </div>
                                        </div>
                                        {config.use_lr_scheduler && (
                                            <div className="wizard-settings-grid wizard-settings-grid-tight">
                                                <NumberInput
                                                    label="Initial LR"
                                                    value={config.initial_lr}
                                                    onChange={(value) => updateConfig('initial_lr', value)}
                                                    min={1e-6}
                                                    max={1e-2}
                                                    step={1e-5}
                                                    precision={6}
                                                />
                                                <NumberInput
                                                    label="Target LR"
                                                    value={config.target_lr}
                                                    onChange={(value) => updateConfig('target_lr', value)}
                                                    min={1e-7}
                                                    max={1e-3}
                                                    step={1e-6}
                                                    precision={7}
                                                />
                                                <NumberInput
                                                    label="Constant Steps"
                                                    value={config.constant_steps}
                                                    onChange={(value) => updateConfig('constant_steps', value)}
                                                    min={0}
                                                    max={50}
                                                    step={1}
                                                    precision={0}
                                                />
                                                <NumberInput
                                                    label="Decay Steps"
                                                    value={config.decay_steps}
                                                    onChange={(value) => updateConfig('decay_steps', value)}
                                                    min={1}
                                                    max={100}
                                                    step={1}
                                                    precision={0}
                                                />
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )
                    }

                    {
                        currentPage === 3 && (
                            <div className="wizard-page">
                                <div className="wizard-card" style={{ marginBottom: '1rem', border: '1px solid var(--primary-200)' }}>
                                    <div className="wizard-card-header">
                                        <span className="wizard-card-icon">🏷️</span>
                                        <span>Training Name</span>
                                    </div>
                                    <div className="wizard-card-body">
                                        <div style={{ padding: '0.5rem 0' }}>
                                            <label className="field-label" style={{ marginBottom: '0.5rem', display: 'block' }}>
                                                Custom Name (Optional)
                                            </label>
                                            <input
                                                type="text"
                                                value={config.custom_name || ''}
                                                onChange={(e) => updateConfig('custom_name', e.target.value)}
                                                placeholder="e.g. Experiment_A"
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
                                        <h5>Selected Dataset</h5>
                                        <div className="wizard-summary-grid">
                                            <span>Dataset</span>
                                            <strong>{selectedDatasetLabel}</strong>
                                        </div>
                                    </div>
                                    <div className="wizard-summary-section">
                                        <h5>Dataset Configuration</h5>
                                        <div className="wizard-summary-grid">
                                            <span>Shuffle buffered</span>
                                            <strong>{config.shuffle_dataset ? 'Enabled' : 'Disabled'}</strong>
                                            {config.shuffle_dataset && (
                                                <>
                                                    <span>Buffer size</span>
                                                    <strong>{config.shuffle_size}</strong>
                                                </>
                                            )}


                                        </div>
                                    </div>
                                    <div className="wizard-summary-section">
                                        <h5>Model Configuration</h5>
                                        <div className="wizard-summary-grid">
                                            <span>Encoders</span>
                                            <strong>{config.num_encoders}</strong>
                                            <span>Attention heads</span>
                                            <strong>{config.num_attention_heads}</strong>
                                            <span>Embedding dims</span>
                                            <strong>{config.molecular_embedding_size}</strong>
                                            <span>Dropout rate</span>
                                            <strong>{config.dropout_rate}</strong>
                                            <span>Model type</span>
                                            <strong>{config.selected_model}</strong>
                                        </div>
                                    </div>
                                    <div className="wizard-summary-section">
                                        <h5>Training Configuration</h5>
                                        <div className="wizard-summary-grid">
                                            <span>Epochs</span>
                                            <strong>{config.epochs}</strong>
                                            <span>Batch size</span>
                                            <strong>{config.batch_size}</strong>
                                            <span>Save checkpoints</span>
                                            <strong>{config.save_checkpoints ? 'Enabled' : 'Disabled'}</strong>
                                            <span>Use GPU</span>
                                            <strong>{config.use_device_GPU ? 'Enabled' : 'Disabled'}</strong>
                                            <span>LR scheduler</span>
                                            <strong>{config.use_lr_scheduler ? 'Enabled' : 'Disabled'}</strong>
                                            {config.use_lr_scheduler && (
                                                <>
                                                    <span>Initial LR</span>
                                                    <strong>{config.initial_lr}</strong>
                                                    <span>Target LR</span>
                                                    <strong>{config.target_lr}</strong>
                                                    <span>Constant steps</span>
                                                    <strong>{config.constant_steps}</strong>
                                                    <span>Decay steps</span>
                                                    <strong>{config.decay_steps}</strong>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )
                    }
                </div >

                <div className="wizard-footer">
                    <button className="secondary" onClick={onClose} disabled={isLoading}>
                        Cancel
                    </button>
                    {currentPage > 0 && (
                        <button className="secondary" onClick={handlePrevious} disabled={isLoading}>
                            Previous
                        </button>
                    )}
                    {currentPage < 3 && (
                        <button
                            className="primary"
                            onClick={handleNext}
                            disabled={isLoading}
                        >
                            Next
                        </button>
                    )}
                    {currentPage === 3 && (
                        <button className="primary" onClick={onConfirm} disabled={isLoading}>
                            {isLoading ? 'Starting...' : 'Confirm Training'}
                        </button>
                    )}
                </div>
            </div >
        </div >
    );
};

export default NewTrainingWizard;
