import React, { useState, useEffect, useCallback } from 'react';
import { NumberInput, Switch, Checkbox } from './UIComponents';
import { DatasetBuilderCard } from './DatasetBuilderCard';
import type { TrainingConfig, TrainingDatasetInfo, CheckpointInfo, TrainingStatus } from '../types';
import {
    fetchTrainingDatasets,
    fetchCheckpoints,
    startTraining,
    resumeTraining,
    stopTraining,
} from '../services';

// Default training configuration based on legacy app
const DEFAULT_CONFIG: TrainingConfig = {
    // Dataset settings
    sample_size: 1.0,
    validation_size: 0.2,
    batch_size: 16,
    shuffle_dataset: true,
    shuffle_size: 256,

    // Model settings
    selected_model: 'SCADS Series',
    dropout_rate: 0.1,
    num_attention_heads: 2,
    num_encoders: 2,
    molecular_embedding_size: 64,

    // Training settings
    epochs: 2,

    // LR scheduler settings
    use_lr_scheduler: true,
    initial_lr: 1e-4,
    target_lr: 1e-5,
    constant_steps: 5,
    decay_steps: 10,

    // Callbacks
    save_checkpoints: true,
    checkpoints_frequency: 5,
};

// Collapsible Section Component
interface CollapsibleSectionProps {
    title: string;
    icon: string;
    defaultOpen?: boolean;
    children: React.ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, icon, defaultOpen = true, children }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className="collapsible-section">
            <div className="collapsible-header" onClick={() => setIsOpen(!isOpen)}>
                <div className="collapsible-title">
                    <span className="collapsible-icon">{icon}</span>
                    <span>{title}</span>
                </div>
                <span className={`collapsible-chevron ${isOpen ? 'open' : ''}`}>▾</span>
            </div>
            {isOpen && <div className="collapsible-content">{children}</div>}
        </div>
    );
};

// Metric Card Component
interface MetricCardProps {
    label: string;
    value: string;
    icon?: string;
    color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, icon, color = 'var(--primary-600)' }) => (
    <div className="metric-card">
        <div className="metric-label">{icon && <span className="metric-icon">{icon}</span>}{label}</div>
        <div className="metric-value" style={{ color }}>{value}</div>
    </div>
);

export const MachineLearningPage: React.FC = () => {
    // Training configuration state
    const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);

    // Data availability state
    const [datasetInfo, setDatasetInfo] = useState<TrainingDatasetInfo>({ available: false });
    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);

    // Training state
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
        is_training: false,
        current_epoch: 0,
        total_epochs: 0,
        progress: 0
    });
    const [trainingLog, setTrainingLog] = useState<string>('Ready to start training...');
    const [isLoading, setIsLoading] = useState(false);

    // Training metrics for dashboard (will be updated via WebSocket)
    const [metrics, _setMetrics] = useState({
        trainLoss: 0,
        valLoss: 0,
        trainAcc: 0,
        valAcc: 0,
    });

    // Additional epochs for resume training
    const [additionalEpochs, setAdditionalEpochs] = useState(10);

    // Load initial data
    useEffect(() => {
        loadDatasetInfo();
        loadCheckpoints();
    }, []);

    const loadDatasetInfo = async () => {
        const result = await fetchTrainingDatasets();
        if (!result.error) {
            setDatasetInfo(result.data);
        }
    };

    const loadCheckpoints = async () => {
        const result = await fetchCheckpoints();
        if (!result.error && result.checkpoints.length > 0) {
            setCheckpoints(result.checkpoints);
            setSelectedCheckpoint(result.checkpoints[0].name);
        }
    };

    const updateConfig = useCallback(<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    }, []);

    const appendLog = useCallback((message: string) => {
        setTrainingLog(prev => prev + '\n' + message);
    }, []);

    const handleStartTraining = async () => {
        if (!datasetInfo.available) {
            appendLog('[ERROR] No training dataset available. Please build dataset first.');
            return;
        }

        setIsLoading(true);
        setTrainingLog('[INFO] Starting training...');

        try {
            const result = await startTraining(config);
            if (result.status === 'started') {
                appendLog(`[INFO] ${result.message}`);
                setTrainingStatus(prev => ({ ...prev, is_training: true, total_epochs: config.epochs }));
            } else {
                appendLog(`[ERROR] ${result.message}`);
            }
        } catch (error) {
            appendLog(`[ERROR] Failed to start training: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleResumeTraining = async () => {
        if (!selectedCheckpoint) {
            appendLog('[ERROR] No checkpoint selected for resuming training.');
            return;
        }

        setIsLoading(true);
        setTrainingLog(`[INFO] Resuming training from ${selectedCheckpoint}...`);

        try {
            const result = await resumeTraining(selectedCheckpoint, additionalEpochs);
            if (result.status === 'started') {
                appendLog(`[INFO] ${result.message}`);
                setTrainingStatus(prev => ({ ...prev, is_training: true }));
            } else {
                appendLog(`[ERROR] ${result.message}`);
            }
        } catch (error) {
            appendLog(`[ERROR] Failed to resume training: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStopTraining = async () => {
        setIsLoading(true);
        appendLog('[INFO] Stopping training...');

        try {
            const result = await stopTraining();
            appendLog(`[INFO] ${result.message}`);
            setTrainingStatus(prev => ({ ...prev, is_training: false }));
        } catch (error) {
            appendLog(`[ERROR] Failed to stop training: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRefreshCheckpoints = async () => {
        await loadCheckpoints();
        appendLog('[INFO] Checkpoints list refreshed.');
    };

    return (
        <div className="ml-page">
            {/* Page Header */}
            <div className="ml-header">
                <h2>SCADS Model Training</h2>
                <p className="ml-subtitle">Configure and monitor your training sessions</p>
            </div>

            {/* Dataset Processing Section */}
            <DatasetBuilderCard onDatasetBuilt={loadDatasetInfo} />

            {/* New Training Session */}
            <CollapsibleSection title="New Training Session" icon="▷" defaultOpen={true}>
                <div className="training-config-grid">
                    {/* Model Architecture Card */}
                    <div className="config-card">
                        <div className="config-card-header">
                            <span className="config-card-icon">🧠</span>
                            <span>Model Architecture</span>
                        </div>
                        <div className="config-card-body">
                            <div className="config-fields-row">
                                <NumberInput
                                    label="Encoders"
                                    value={config.num_encoders}
                                    onChange={(v) => updateConfig('num_encoders', v)}
                                    min={1}
                                    max={12}
                                    step={1}
                                    precision={0}
                                />
                                <NumberInput
                                    label="Attention Heads"
                                    value={config.num_attention_heads}
                                    onChange={(v) => updateConfig('num_attention_heads', v)}
                                    min={1}
                                    max={16}
                                    step={1}
                                    precision={0}
                                />
                                <NumberInput
                                    label="Embedding Dims"
                                    value={config.molecular_embedding_size}
                                    onChange={(v) => updateConfig('molecular_embedding_size', v)}
                                    min={64}
                                    max={1024}
                                    step={64}
                                    precision={0}
                                />
                            </div>
                            <div className="config-fields-row">
                                <NumberInput
                                    label="Dropout Rate"
                                    value={config.dropout_rate}
                                    onChange={(v) => updateConfig('dropout_rate', v)}
                                    min={0}
                                    max={0.5}
                                    step={0.05}
                                    precision={2}
                                />
                                <div style={{ flex: 1 }}>
                                    <label className="field-label">Model Type</label>
                                    <select
                                        value={config.selected_model}
                                        onChange={(e) => updateConfig('selected_model', e.target.value as 'SCADS Series' | 'SCADS Atomic')}
                                        className="select-input"
                                    >
                                        <option value="SCADS Series">SCADS Series</option>
                                        <option value="SCADS Atomic">SCADS Atomic</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Dataset Config Card */}
                    <div className="config-card">
                        <div className="config-card-header">
                            <span className="config-card-icon">📊</span>
                            <span>Dataset Config</span>
                            <span className={`status-badge-small ${datasetInfo.available ? 'available' : 'unavailable'}`}>
                                {datasetInfo.available ? 'Ready' : 'No Data'}
                            </span>
                        </div>
                        <div className="config-card-body">
                            <div className="toggle-row">
                                <Switch
                                    checked={config.shuffle_dataset}
                                    onChange={(v) => updateConfig('shuffle_dataset', v)}
                                    label="Shuffle Buffered"
                                />
                            </div>
                            {config.shuffle_dataset && (
                                <div className="config-fields-row" style={{ marginTop: '1rem' }}>
                                    <NumberInput
                                        label="Buffer Size"
                                        value={config.shuffle_size}
                                        onChange={(v) => updateConfig('shuffle_size', v)}
                                        min={100}
                                        max={10000}
                                        step={100}
                                        precision={0}
                                    />
                                </div>
                            )}
                            <div className="config-fields-row" style={{ marginTop: '1rem' }}>
                                <NumberInput
                                    label="Sample Size"
                                    value={config.sample_size}
                                    onChange={(v) => updateConfig('sample_size', v)}
                                    min={0.01}
                                    max={1.0}
                                    step={0.01}
                                    precision={2}
                                />
                                <NumberInput
                                    label="Validation Split"
                                    value={config.validation_size}
                                    onChange={(v) => updateConfig('validation_size', v)}
                                    min={0.05}
                                    max={0.5}
                                    step={0.05}
                                    precision={2}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Training Parameters */}
                <div className="training-params-section">
                    <div className="section-label">
                        <span className="section-icon">✨</span>
                        <span>Training Parameters</span>
                    </div>
                    <div className="config-fields-row">
                        <NumberInput
                            label="Epochs"
                            value={config.epochs}
                            onChange={(v) => updateConfig('epochs', v)}
                            min={1}
                            max={500}
                            step={1}
                            precision={0}
                        />
                        <NumberInput
                            label="Batch Size"
                            value={config.batch_size}
                            onChange={(v) => updateConfig('batch_size', v)}
                            min={1}
                            max={256}
                            step={1}
                            precision={0}
                        />
                    </div>
                    <div className="toggles-row">
                        <Checkbox
                            label="Save Checkpoints"
                            checked={config.save_checkpoints}
                            onChange={(v) => updateConfig('save_checkpoints', v)}
                        />
                        <Checkbox
                            label="LR Scheduler"
                            checked={config.use_lr_scheduler}
                            onChange={(v) => updateConfig('use_lr_scheduler', v)}
                        />
                    </div>

                    {config.use_lr_scheduler && (
                        <div className="lr-scheduler-params">
                            <div className="config-fields-row">
                                <NumberInput
                                    label="Initial LR"
                                    value={config.initial_lr}
                                    onChange={(v) => updateConfig('initial_lr', v)}
                                    min={1e-6}
                                    max={1e-2}
                                    step={1e-5}
                                    precision={6}
                                />
                                <NumberInput
                                    label="Target LR"
                                    value={config.target_lr}
                                    onChange={(v) => updateConfig('target_lr', v)}
                                    min={1e-7}
                                    max={1e-3}
                                    step={1e-6}
                                    precision={7}
                                />
                                <NumberInput
                                    label="Constant Steps"
                                    value={config.constant_steps}
                                    onChange={(v) => updateConfig('constant_steps', v)}
                                    min={0}
                                    max={50}
                                    step={1}
                                    precision={0}
                                />
                                <NumberInput
                                    label="Decay Steps"
                                    value={config.decay_steps}
                                    onChange={(v) => updateConfig('decay_steps', v)}
                                    min={1}
                                    max={100}
                                    step={1}
                                    precision={0}
                                />
                            </div>
                        </div>
                    )}

                    <div className="start-training-row">
                        <button
                            className="primary start-btn"
                            onClick={handleStartTraining}
                            disabled={!datasetInfo.available || trainingStatus.is_training || isLoading}
                        >
                            {isLoading ? '⏳ Starting...' : '▷ Start Training'}
                        </button>
                    </div>
                </div>
            </CollapsibleSection>

            {/* Resume Training Session */}
            <CollapsibleSection title="Resume Training Session" icon="↺" defaultOpen={false}>
                <div className="resume-section-content">
                    {checkpoints.length > 0 ? (
                        <div className="resume-grid">
                            <div className="resume-controls">
                                <div style={{ flex: 2 }}>
                                    <label className="field-label">Select Checkpoint</label>
                                    <select
                                        value={selectedCheckpoint || ''}
                                        onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                        className="select-input"
                                    >
                                        {checkpoints.map((cp) => (
                                            <option key={cp.name} value={cp.name}>
                                                {cp.name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <NumberInput
                                    label="Additional Epochs"
                                    value={additionalEpochs}
                                    onChange={setAdditionalEpochs}
                                    min={1}
                                    max={100}
                                    step={1}
                                    precision={0}
                                />
                                <div className="resume-buttons">
                                    <button className="secondary" onClick={handleRefreshCheckpoints}>
                                        🔄 Refresh
                                    </button>
                                    <button
                                        className="primary"
                                        onClick={handleResumeTraining}
                                        disabled={!selectedCheckpoint || trainingStatus.is_training || isLoading}
                                    >
                                        ↺ Resume
                                    </button>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <p className="no-data-message">
                            No checkpoints available. Train a model first to create checkpoints.
                        </p>
                    )}
                </div>
            </CollapsibleSection>

            {/* Training Dashboard */}
            <div className="training-dashboard">
                <div className="dashboard-header">
                    <div className="dashboard-title">
                        <span className="dashboard-icon">✨</span>
                        <span>Training Dashboard</span>
                    </div>
                    <span className={`training-status-badge ${trainingStatus.is_training ? 'active' : 'idle'}`}>
                        {trainingStatus.is_training ? '● Training in Progress' : '○ Idle'}
                    </span>
                </div>

                {/* Metrics Row */}
                <div className="metrics-row">
                    <MetricCard
                        label="EPOCH"
                        value={`${trainingStatus.current_epoch} / ${trainingStatus.total_epochs || config.epochs}`}
                        color="var(--slate-800)"
                    />
                    <MetricCard
                        label="TRAIN LOSS"
                        icon="↘"
                        value={metrics.trainLoss.toFixed(4)}
                        color="#f59e0b"
                    />
                    <MetricCard
                        label="VAL LOSS"
                        icon="↘"
                        value={metrics.valLoss.toFixed(4)}
                        color="#f59e0b"
                    />
                    <MetricCard
                        label="TRAIN ACC"
                        icon="◉"
                        value={`${(metrics.trainAcc * 100).toFixed(2)}%`}
                        color="#22c55e"
                    />
                    <MetricCard
                        label="VAL ACC"
                        icon="◉"
                        value={`${(metrics.valAcc * 100).toFixed(2)}%`}
                        color="#22c55e"
                    />
                </div>

                {/* Progress Bar */}
                <div className="dashboard-progress">
                    <div className="progress-label">
                        <span>% Progress: {Math.round(trainingStatus.progress)}%</span>
                    </div>
                    <div className="progress-bar-container">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${trainingStatus.progress}%` }}
                        />
                    </div>
                    <div className="progress-actions">
                        <button
                            className="stop-btn"
                            onClick={handleStopTraining}
                            disabled={!trainingStatus.is_training || isLoading}
                        >
                            ⏹ Stop Training
                        </button>
                    </div>
                </div>

                {/* Charts Placeholder */}
                <div className="charts-container">
                    <div className="chart-panel">
                        <div className="chart-title">LOSS</div>
                        <div className="chart-placeholder">
                            <p>Loss chart will appear during training</p>
                            <small>Train Loss ● Val Loss</small>
                        </div>
                    </div>
                    <div className="chart-panel">
                        <div className="chart-title">ACCURACY</div>
                        <div className="chart-placeholder">
                            <p>Accuracy chart will appear during training</p>
                            <small>Train Accuracy ● Val Accuracy</small>
                        </div>
                    </div>
                </div>

                {/* Training Log */}
                <div className="log-panel">
                    <div className="log-header">
                        <span>Training Log</span>
                        <button
                            className="ghost-button"
                            onClick={() => setTrainingLog('Ready to start training...')}
                        >
                            Clear
                        </button>
                    </div>
                    <pre className="training-log-light">{trainingLog}</pre>
                </div>
            </div>
        </div>
    );
};

export default MachineLearningPage;
