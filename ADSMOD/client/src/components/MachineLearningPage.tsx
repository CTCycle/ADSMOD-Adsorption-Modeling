import React, { useState, useEffect, useCallback } from 'react';
import { DatasetBuilderCard } from './DatasetBuilderCard';
import { NewTrainingWizard } from './NewTrainingWizard';
import { ResumeTrainingWizard } from './ResumeTrainingWizard';
import { TrainingSetupRow } from './TrainingSetupRow';
import type {
    TrainingConfig,
    TrainingDatasetInfo,
    CheckpointInfo,
    TrainingStatus,
    ResumeTrainingConfig,
} from '../types';
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

    // Training state
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
        is_training: false,
        current_epoch: 0,
        total_epochs: 0,
        progress: 0
    });
    const [trainingLog, setTrainingLog] = useState<string>('Ready to start training...');
    const [isLoading, setIsLoading] = useState(false);
    const [showNewTrainingWizard, setShowNewTrainingWizard] = useState(false);
    const [showResumeTrainingWizard, setShowResumeTrainingWizard] = useState(false);
    const [resumeConfig, setResumeConfig] = useState<ResumeTrainingConfig>({
        checkpoint_name: '',
        additional_epochs: 10,
    });

    // Training metrics for dashboard (will be updated via WebSocket)
    const [metrics, _setMetrics] = useState({
        trainLoss: 0,
        valLoss: 0,
        trainAcc: 0,
        valAcc: 0,
    });

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
        if (!result.error) {
            setCheckpoints(result.checkpoints);
        }
    };

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

    const handleConfirmTraining = async () => {
        await handleStartTraining();
        setShowNewTrainingWizard(false);
    };

    const handleResumeTrainingClick = () => {
        const compatibleCheckpoint = checkpoints.find((checkpoint) => checkpoint.is_compatible);
        const fallbackCheckpoint = compatibleCheckpoint || checkpoints[0];
        setResumeConfig((prev) => ({
            checkpoint_name: fallbackCheckpoint ? fallbackCheckpoint.name : prev.checkpoint_name,
            additional_epochs: prev.additional_epochs || 10,
        }));
        setShowResumeTrainingWizard(true);
    };

    const handleConfirmResume = async () => {
        if (!resumeConfig.checkpoint_name) {
            appendLog('[ERROR] Select a checkpoint before resuming training.');
            return;
        }

        setIsLoading(true);
        setTrainingLog('[INFO] Resuming training...');

        try {
            const result = await resumeTraining(resumeConfig);
            if (result.status === 'started') {
                const selectedCheckpoint = checkpoints.find(
                    (checkpoint) => checkpoint.name === resumeConfig.checkpoint_name
                );
                const trainedEpochs = selectedCheckpoint?.epochs_trained ?? 0;
                const baseEpochs = typeof trainedEpochs === 'number' ? trainedEpochs : 0;
                const totalEpochs = baseEpochs + resumeConfig.additional_epochs;

                appendLog(`[INFO] ${result.message}`);
                const resumeProgress = totalEpochs > 0 ? (baseEpochs / totalEpochs) * 100 : 0;
                setTrainingStatus({
                    is_training: true,
                    current_epoch: baseEpochs,
                    total_epochs: totalEpochs,
                    progress: resumeProgress,
                });
                setShowResumeTrainingWizard(false);
            } else {
                appendLog(`[ERROR] ${result.message}`);
            }
        } catch (error) {
            appendLog(`[ERROR] Failed to resume training: ${error}`);
        } finally {
            setIsLoading(false);
        }
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

            {/* Training Setup Cards */}
            <TrainingSetupRow
                onNewTrainingClick={() => setShowNewTrainingWizard(true)}
                onResumeTrainingClick={handleResumeTrainingClick}
                datasetAvailable={datasetInfo.available}
                checkpointsAvailable={checkpoints.length > 0}
                isTraining={trainingStatus.is_training}
            />

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

            {showNewTrainingWizard && (
                <NewTrainingWizard
                    config={config}
                    onConfigChange={setConfig}
                    onClose={() => setShowNewTrainingWizard(false)}
                    onConfirm={handleConfirmTraining}
                    isLoading={isLoading}
                />
            )}

            {showResumeTrainingWizard && (
                <ResumeTrainingWizard
                    checkpoints={checkpoints}
                    config={resumeConfig}
                    onConfigChange={setResumeConfig}
                    onClose={() => setShowResumeTrainingWizard(false)}
                    onConfirm={handleConfirmResume}
                    isLoading={isLoading}
                />
            )}
        </div>
    );
};

export default MachineLearningPage;
