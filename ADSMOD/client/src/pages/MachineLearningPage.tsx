import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts';
import { DatasetBuilderCard } from '../components/DatasetBuilderCard';
import { NewTrainingWizard } from '../components/NewTrainingWizard';
import { ResumeTrainingWizard } from '../components/ResumeTrainingWizard';
import { TrainingSetupRow } from '../components/TrainingSetupRow';
import { InfoModal } from '../components/InfoModal'; // Import InfoModal
import type {
    CheckpointFullDetails,
    TrainingConfig,
    CheckpointInfo,
    TrainingStatus,
    ResumeTrainingConfig,
    TrainingHistoryPoint,
    ProcessedDatasetInfo,
    DatasetFullInfo,
    InfoModalData,
} from '../types';
import {
    fetchCheckpoints,
    startTraining,
    resumeTraining,
    stopTraining,
    getTrainingStatus,
    fetchProcessedDatasets,
    deleteDataset, // Import deleteDataset
    getTrainingDatasetInfo, // Import getTrainingDatasetInfo if needed for view metadata
    deleteCheckpoint, // Import deleteCheckpoint
    fetchCheckpointDetails,
} from '../services';

// Default training configuration based on legacy app
const DEFAULT_CONFIG: TrainingConfig = {
    // Dataset settings
    batch_size: 16,
    shuffle_dataset: true,
    max_buffer_size: 256,

    // Model settings
    selected_model: 'SCADS Series',
    dropout_rate: 0.1,
    num_attention_heads: 2,
    num_encoders: 2,
    molecular_embedding_size: 64,

    // Training settings
    epochs: 2,
    dataloader_workers: 0,
    prefetch_factor: 1,
    pin_memory: true,
    use_device_GPU: true,
    device_ID: 0,
    use_mixed_precision: false,
    use_jit: false,
    jit_backend: 'inductor',

    // LR scheduler settings
    use_lr_scheduler: false,
    initial_lr: 1e-4,
    target_lr: 1e-5,
    constant_steps: 5,
    decay_steps: 10,

    // Callbacks
    save_checkpoints: false,
    checkpoints_frequency: 5,
    custom_name: '',
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

const CHART_COLORS = {
    loss: '#f59e0b',
    valLoss: '#2563eb', // Blue for high contrast against Orange
    metric: '#16a34a',
    valMetric: '#9333ea', // Purple for high contrast against Green
};

const formatLossValue = (value: number | undefined): string => {
    const safeValue = typeof value === 'number' ? value : 0;
    return safeValue.toFixed(4);
};

const formatMetricValue = (value: number | undefined, asPercent: boolean): string => {
    const safeValue = typeof value === 'number' ? value : 0;
    if (asPercent) {
        return `${(safeValue * 100).toFixed(2)}%`;
    }
    return safeValue.toFixed(4);
};

const buildDatasetMetadataModalData = (info: DatasetFullInfo): InfoModalData => ({
    'Dataset Label': info.dataset_label,
    'Created At': info.created_at,
    'Total Samples': info.total_samples,
    'Train Samples': info.train_samples,
    'Validation Samples': info.validation_samples,
    'Sample Fraction': info.sample_size,
    'Validation Fraction': info.validation_size,
    'Min Measurements': info.min_measurements,
    'Max Measurements': info.max_measurements,
    'SMILES Length': info.smile_sequence_size,
    'Max Pressure': info.max_pressure,
    'Max Uptake': info.max_uptake,
    'SMILES Vocabulary': info.smile_vocabulary_size,
    'Adsorbents Count': info.adsorbent_vocabulary_size,
    'Normalization': info.normalization_stats,
});

const buildCheckpointDetailsModalData = (details: CheckpointFullDetails): InfoModalData => ({
    'Name': details.name,
    'Epochs Trained': details.epochs_trained,
    'Final Loss': details.final_loss?.toFixed(6) ?? 'N/A',
    'Is Compatible': details.is_compatible ? 'Yes' : 'No',
    'Created At': details.created_at || 'Unknown',
});

export const MachineLearningPage: React.FC = () => {
    // Training configuration state
    const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);


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

    // Processed datasets for training wizard
    const [processedDatasets, setProcessedDatasets] = useState<ProcessedDatasetInfo[]>([]);
    const [selectedDatasetLabel, setSelectedDatasetLabel] = useState<string | null>(null);
    const [selectedDatasetHash, setSelectedDatasetHash] = useState<string | null>(null);

    // Info Modal State
    const [infoModalOpen, setInfoModalOpen] = useState(false);
    const [infoModalTitle, setInfoModalTitle] = useState('');
    const [infoModalData, setInfoModalData] = useState<InfoModalData | null>(null);

    // Polling ref
    const pollIntervalRef = useRef<number | null>(null);
    const pollIntervalSecondsRef = useRef<number | null>(null);
    const logContainerRef = useRef<HTMLPreElement>(null);

    // Load initial data
    useEffect(() => {
        loadCheckpoints();
        loadProcessedDatasets();
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
             // We don't want to restart polling aggressively here if already running
             // checkStatus handles regular updates. 
             // But if we just loaded the page and is_training=true, we default to 1s
             // until next checkStatus updates us? 
             // Actually, useEffect on is_training is good for initial load state restoration
             if (!pollIntervalRef.current) {
                 const intervalSeconds = trainingStatus.poll_interval
                     ?? pollIntervalSecondsRef.current
                     ?? 1.0;
                 startPolling(intervalSeconds);
             }
        } else {
            stopPolling();
        }
    }, [trainingStatus.is_training]);

    const normalizePollingInterval = (intervalSeconds: number | null | undefined) => {
        if (typeof intervalSeconds !== 'number' || Number.isNaN(intervalSeconds)) {
            return null;
        }
        return intervalSeconds < 0 ? 0 : intervalSeconds;
    };

    const startPolling = (intervalSeconds: number = 1.0) => {
        const normalizedInterval = normalizePollingInterval(intervalSeconds) ?? 1.0;
        if (pollIntervalRef.current) {
            window.clearInterval(pollIntervalRef.current);
        }
        pollIntervalSecondsRef.current = normalizedInterval;
        pollIntervalRef.current = window.setInterval(checkStatus, normalizedInterval * 1000);
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
            poll_interval: status.poll_interval,
        });

        // Update polling interval if backend changed it
        const nextInterval = normalizePollingInterval(status.poll_interval);
        if (nextInterval !== null && pollIntervalSecondsRef.current !== nextInterval) {
            if (status.is_training) {
                startPolling(nextInterval);
            } else {
                pollIntervalSecondsRef.current = nextInterval;
            }
        }

        // Refresh checkpoints if training just finished
        if (!status.is_training && trainingStatus.is_training) {
            loadCheckpoints();
        }
    };



    const loadCheckpoints = async () => {
        const result = await fetchCheckpoints();
        if (!result.error) {
            setCheckpoints(result.checkpoints);
        }
    };

    const loadProcessedDatasets = async () => {
        const { datasets, error } = await fetchProcessedDatasets();
        if (!error) {
            setProcessedDatasets(datasets);
        }
    };

    const handleDeleteDataset = async (label: string) => {
        if (window.confirm(`Are you sure you want to delete dataset '${label}'?`)) {
            const { success, message } = await deleteDataset(label);
            if (success) {
                loadProcessedDatasets();
            } else {
                alert(`Failed to delete dataset: ${message}`);
            }
        }
    };

    const handleViewDatasetMetadata = async (label: string) => {
        setIsLoading(true);
        const info = await getTrainingDatasetInfo(label);
        setIsLoading(false);

        if (info && info.available) {
            setInfoModalTitle('Dataset Metadata');
            setInfoModalData(buildDatasetMetadataModalData(info));
            setInfoModalOpen(true);
        } else {
            alert(`Could not fetch details for dataset '${label}'`);
        }
    };

    const handleDeleteCheckpoint = async (name: string) => {
        if (window.confirm(`Are you sure you want to delete checkpoint '${name}'?`)) {
            const { success, error } = await deleteCheckpoint(name);
            if (success) {
                loadCheckpoints();
            } else {
                alert(`Failed to delete checkpoint: ${error}`);
            }
        }
    };

    const handleViewCheckpointDetails = async (name: string) => {
        const { details, error } = await fetchCheckpointDetails(name);
        if (error) {
            // Error falls back to alert or could use a toast
            alert(`Failed to load details: ${error}`);
        } else if (details) {
            setInfoModalTitle('Checkpoint Details');
            setInfoModalData(buildCheckpointDetailsModalData(details));
            setInfoModalOpen(true);
        }
    };

    const handleNewTrainingClick = (datasetLabel: string) => {
        handleDatasetSelect(datasetLabel);
        setShowNewTrainingWizard(true);
    };

    const handleResumeTrainingClickWithSelection = (checkpointName: string) => {
        setResumeConfig(prev => ({ ...prev, checkpoint_name: checkpointName }));
        setShowResumeTrainingWizard(true);
    };

    // Derived metrics for UI
    const metrics: Record<string, number> = trainingStatus.metrics || {};
    const history: TrainingHistoryPoint[] = trainingStatus.history || [];
    const hasAccuracyMetric = history.some((entry) =>
        typeof entry.accuracy === 'number' || typeof entry.val_accuracy === 'number'
    ) || typeof metrics.accuracy === 'number' || typeof metrics.val_accuracy === 'number';
    const metricKey = hasAccuracyMetric ? 'accuracy' : 'masked_r2';
    const valMetricKey = hasAccuracyMetric ? 'val_accuracy' : 'val_masked_r2';
    const metricLabel = hasAccuracyMetric ? 'ACC' : 'R2';
    const metricTitle = hasAccuracyMetric ? 'ACCURACY' : 'R2 SCORE';
    const metricAsPercent = hasAccuracyMetric;
    const hasHistory = history.length > 0;

    // Handlers for training actions
    const handleConfirmTraining = useCallback(async () => {
        setIsLoading(true);
        // Include the selected dataset label in the configuration
        const trainingConfig = {
            ...config,
            dataset_label: selectedDatasetLabel || undefined,
            dataset_hash: selectedDatasetHash || undefined,
        };
        const result = await startTraining(trainingConfig);
        setIsLoading(false);
        if (result.status === 'started') {
            setShowNewTrainingWizard(false);
            setTrainingStatus(prev => ({
                ...prev,
                log: [...(prev.log || []), result.message]
            }));
            startPolling(result.poll_interval ?? 1.0);
            checkStatus(); // Immediately check status to update UI
        } else {
            console.error('Failed to start training:', result.message);
            alert(`Failed to start training: ${result.message}`);
        }
    }, [config, selectedDatasetHash, selectedDatasetLabel]);

    const handleDatasetSelect = useCallback((label: string) => {
        setSelectedDatasetLabel(label);
        const match = processedDatasets.find((dataset) => dataset.dataset_label === label);
        setSelectedDatasetHash(match?.dataset_hash ?? null);
    }, [processedDatasets]);

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
            startPolling(result.poll_interval ?? 1.0);
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

            <DatasetBuilderCard onDatasetBuilt={() => { loadProcessedDatasets(); }} />

            <TrainingSetupRow
                onNewTrainingClick={handleNewTrainingClick}
                onResumeTrainingClick={handleResumeTrainingClickWithSelection}
                datasetAvailable={processedDatasets.length > 0}
                checkpointsAvailable={checkpoints.length > 0}
                processedDatasets={processedDatasets}
                checkpoints={checkpoints}
                isTraining={trainingStatus.is_training}
                onDeleteDataset={handleDeleteDataset}
                onViewDatasetMetadata={handleViewDatasetMetadata}
                onDeleteCheckpoint={handleDeleteCheckpoint}
                onViewCheckpointDetails={handleViewCheckpointDetails}
                onRefreshDatasets={loadProcessedDatasets}
                onRefreshCheckpoints={loadCheckpoints}
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
                        value={formatLossValue(metrics.loss)}
                        color={CHART_COLORS.loss}
                    />
                    <MetricCard
                        label="VAL LOSS"
                        icon="↘"
                        value={formatLossValue(metrics.val_loss)}
                        color={CHART_COLORS.valLoss}
                    />
                    <MetricCard
                        label={`TRAIN ${metricLabel}`}
                        icon="◉"
                        value={formatMetricValue(metrics[metricKey], metricAsPercent)}
                        color={CHART_COLORS.metric}
                    />
                    <MetricCard
                        label={`VAL ${metricLabel}`}
                        icon="◉"
                        value={formatMetricValue(metrics[valMetricKey], metricAsPercent)}
                        color={CHART_COLORS.valMetric}
                    />
                </div>

                <div className="dashboard-progress">
                    <div className="progress-label">
                        <span>% Progress: {Math.round(trainingStatus.progress)}%</span>
                    </div>
                    <div className="progress-bar-wrapper">
                        <div className="progress-bar-container">
                            <div
                                className="progress-bar-fill"
                                style={{ width: `${trainingStatus.progress}%` }}
                            />
                        </div>
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
                        {hasHistory ? (
                            <div className="chart-wrapper" style={{ width: '100%', height: 250 }}>
                                <ResponsiveContainer>
                                    <LineChart data={history}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                                        <YAxis tick={{ fontSize: 12 }} />
                                        <Tooltip
                                            labelFormatter={(value) => `Epoch ${value}`}
                                            contentStyle={{
                                                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                                                borderRadius: '4px',
                                                border: 'none',
                                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                            }}
                                        />
                                        <Legend />
                                        <Line
                                            type="monotone"
                                            dataKey="loss"
                                            stroke={CHART_COLORS.loss}
                                            strokeWidth={2}
                                            dot={false}
                                            name="Train Loss"
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="val_loss"
                                            stroke={CHART_COLORS.valLoss}
                                            strokeWidth={2}
                                            dot={false}
                                            name="Val Loss"
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        ) : (
                            <div className="chart-placeholder">
                                Waiting for training data...
                                <small>Loss metrics will appear once training starts.</small>
                            </div>
                        )}
                    </div>
                    <div className="chart-panel">
                        <div className="chart-title">{metricTitle}</div>
                        {hasHistory ? (
                            <div className="chart-wrapper" style={{ width: '100%', height: 250 }}>
                                <ResponsiveContainer>
                                    <LineChart data={history}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                                        <YAxis
                                            domain={metricAsPercent ? [0, 1] : ['auto', 'auto']}
                                            tick={{ fontSize: 12 }}
                                        />
                                        <Tooltip
                                            labelFormatter={(value) => `Epoch ${value}`}
                                            contentStyle={{
                                                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                                                borderRadius: '4px',
                                                border: 'none',
                                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                            }}
                                        />
                                        <Legend />
                                        <Line
                                            type="monotone"
                                            dataKey={metricKey}
                                            stroke={CHART_COLORS.metric}
                                            strokeWidth={2}
                                            dot={false}
                                            name={`Train ${metricLabel}`}
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey={valMetricKey}
                                            stroke={CHART_COLORS.valMetric}
                                            strokeWidth={2}
                                            dot={false}
                                            name={`Val ${metricLabel}`}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        ) : (
                            <div className="chart-placeholder">
                                Waiting for training data...
                                <small>Validation metrics will appear once training starts.</small>
                            </div>
                        )}
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
                    selectedDatasetLabel={selectedDatasetLabel || ''}
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
                    selectedCheckpointName={resumeConfig.checkpoint_name} // Pass selected checkpoint
                />
            )}

            <InfoModal
                isOpen={infoModalOpen}
                onClose={() => setInfoModalOpen(false)}
                title={infoModalTitle}
                data={infoModalData}
            />
        </div>
    );
};

export default MachineLearningPage;
