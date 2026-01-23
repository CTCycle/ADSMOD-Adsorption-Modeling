import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';
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
    getTrainingStatus,
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
    use_device_GPU: false,
    use_mixed_precision: false,

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
        progress: 0,
        metrics: {},
        history: [],
        log: [],
    });

    // UI state
    const [isLoading, setIsLoading] = useState(false);
    const [showNewTrainingWizard, setShowNewTrainingWizard] = useState(false);
    const [showResumeTrainingWizard, setShowResumeTrainingWizard] = useState(false);
    const [resumeConfig, setResumeConfig] = useState<ResumeTrainingConfig>({
        checkpoint_name: '',
        additional_epochs: 10,
    });

    // Polling ref
    const pollIntervalRef = useRef<number | null>(null);
    const logContainerRef = useRef<HTMLPreElement>(null);

    // Load initial data
    useEffect(() => {
        loadDatasetInfo();
        loadCheckpoints();
        // Initial status check to catch up if page is refreshed
        checkStatus();
        return () => stopPolling();
    }, []);

    // Auto-scroll logs
    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [trainingStatus.log]);

    // Poll status when training is active
    useEffect(() => {
        if (trainingStatus.is_training) {
            startPolling();
        } else {
            stopPolling();
        }
    }, [trainingStatus.is_training]);

    const startPolling = () => {
        if (pollIntervalRef.current) return;
        pollIntervalRef.current = window.setInterval(checkStatus, 1000);
    };

    const stopPolling = () => {
        if (pollIntervalRef.current) {
            window.clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
        }
    };

    const checkStatus = async () => {
        const status = await getTrainingStatus();
        if (status.error) {
            console.error('Failed to poll status:', status.error);
            return;
        }

        // Merge logs to avoid full overwrite if needed, but for now full replace is fine as backend sends window
        setTrainingStatus({
            is_training: status.is_training,
            current_epoch: status.current_epoch,
            total_epochs: status.total_epochs,
            progress: status.progress,
            metrics: status.metrics || {},
            history: status.history || [],
            log: status.log || [],
        });

        // Refresh checkpoints if training just finished
        if (!status.is_training && trainingStatus.is_training) {
            loadCheckpoints();
        }
    };

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

    // Derived metrics for UI
    const metrics = trainingStatus.metrics || {};

    // Handlers for training actions
    const handleConfirmTraining = useCallback(async () => {
        setIsLoading(true);
        const result = await startTraining(config);
        setIsLoading(false);
        if (result.status === 'started') {
            setShowNewTrainingWizard(false);
            setTrainingStatus(prev => ({
                ...prev,
                log: [...(prev.log || []), result.message]
            }));
            checkStatus(); // Immediately check status to update UI
        } else {
            console.error('Failed to start training:', result.message);
            alert(`Failed to start training: ${result.message}`);
        }
    }, [config]);

    const handleResumeTrainingClick = useCallback(() => {
        if (checkpoints.length > 0) {
            setResumeConfig(prev => ({ ...prev, checkpoint_name: checkpoints[0].name }));
        }
        setShowResumeTrainingWizard(true);
    }, [checkpoints]);

    const handleConfirmResume = useCallback(async () => {
        setIsLoading(true);
        const result = await resumeTraining(resumeConfig);
        setIsLoading(false);
        if (result.status === 'started') {
            setShowResumeTrainingWizard(false);
            setTrainingStatus(prev => ({
                ...prev,
                log: [...(prev.log || []), result.message]
            }));
            checkStatus(); // Immediately check status to update UI
        } else {
            console.error('Failed to resume training:', result.message);
            alert(`Failed to resume training: ${result.message}`);
        }
    }, [resumeConfig]);

    const handleStopTraining = useCallback(async () => {
        setIsLoading(true);
        const result = await stopTraining();
        setIsLoading(false);
        if (result.status === 'stopped') {
            setTrainingStatus(prev => ({
                ...prev,
                log: [...(prev.log || []), result.message]
            }));
            checkStatus(); // Immediately check status to update UI
        } else {
            console.error('Failed to stop training:', result.message);
            alert(`Failed to stop training: ${result.message}`);
        }
    }, []);

    return (
        <div className="ml-page">
            <div className="ml-header">
                <h2>SCADS Model Training</h2>
                <p className="ml-subtitle">Configure and monitor your training sessions</p>
            </div>

            <DatasetBuilderCard onDatasetBuilt={loadDatasetInfo} />

            <TrainingSetupRow
                onNewTrainingClick={() => setShowNewTrainingWizard(true)}
                onResumeTrainingClick={handleResumeTrainingClick}
                datasetAvailable={datasetInfo.available}
                checkpointsAvailable={checkpoints.length > 0}
                isTraining={trainingStatus.is_training}
            />

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

                <div className="metrics-row">
                    <MetricCard
                        label="EPOCH"
                        value={`${trainingStatus.current_epoch} / ${trainingStatus.total_epochs || config.epochs}`}
                        color="var(--slate-800)"
                    />
                    <MetricCard
                        label="TRAIN LOSS"
                        icon="↘"
                        value={metrics.loss?.toFixed(4) || '0.0000'}
                        color="#f59e0b"
                    />
                    <MetricCard
                        label="VAL LOSS"
                        icon="↘"
                        value={metrics.val_loss?.toFixed(4) || '0.0000'}
                        color="#f59e0b"
                    />
                    <MetricCard
                        label="TRAIN ACC"
                        icon="◉"
                        value={`${(metrics.accuracy ? metrics.accuracy * 100 : 0).toFixed(2)}%`}
                        color="#22c55e"
                    />
                    <MetricCard
                        label="VAL ACC"
                        icon="◉"
                        value={`${(metrics.val_accuracy ? metrics.val_accuracy * 100 : 0).toFixed(2)}%`}
                        color="#22c55e"
                    />
                </div>

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

                <div className="charts-container">
                    <div className="chart-panel">
                        <div className="chart-title">LOSS</div>
                        <div className="chart-wrapper" style={{ width: '100%', height: 250 }}>
                            <ResponsiveContainer>
                                <LineChart data={trainingStatus.history}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                    <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                                    <YAxis tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '4px', border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
                                    />
                                    <Line type="monotone" dataKey="loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Train Loss" />
                                    <Line type="monotone" dataKey="val_loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Val Loss" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <div className="chart-panel">
                        <div className="chart-title">ACCURACY</div>
                        <div className="chart-wrapper" style={{ width: '100%', height: 250 }}>
                            <ResponsiveContainer>
                                <LineChart data={trainingStatus.history}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                    <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                                    <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '4px', border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
                                    />
                                    <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} dot={false} name="Train Acc" />
                                    <Line type="monotone" dataKey="val_accuracy" stroke="#3b82f6" strokeWidth={2} dot={false} name="Val Acc" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                <div className="log-panel">
                    <div className="log-header">
                        <span>Training Log</span>
                        <button
                            className="ghost-button"
                            onClick={() => setTrainingStatus(prev => ({ ...prev, log: ['Ready to start training...'] }))}
                        >
                            Clear
                        </button>
                    </div>
                    <pre className="training-log-light" ref={logContainerRef}>
                        {(trainingStatus.log || []).join('\n')}
                    </pre>
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
