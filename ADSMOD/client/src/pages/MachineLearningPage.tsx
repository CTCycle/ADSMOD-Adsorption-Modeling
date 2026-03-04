import React, { useState, useEffect, useCallback, useRef } from 'react';
import { DatasetBuilderCard } from '../components/DatasetBuilderCard';
import { NewTrainingWizard } from '../components/NewTrainingWizard';
import { ResumeTrainingWizard } from '../components/ResumeTrainingWizard';
import { TrainingSetupRow } from '../components/TrainingSetupRow';
import { InfoModal } from '../components/InfoModal'; // Import InfoModal
import { TrainingHistoryChartPanel } from '../features/training/components/TrainingHistoryChartPanel';
import { useTrainingActionRunner } from '../features/training/hooks/useTrainingActionRunner';
import type {
    CheckpointFullDetails,
    TrainingConfig,
    CheckpointInfo,
    TrainingStatus,
    ResumeTrainingConfig,
    TrainingHistoryPoint,
    TrainingMetrics,
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
    const wasTrainingRef = useRef(false);
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

        const wasTraining = wasTrainingRef.current;
        wasTrainingRef.current = status.is_training;

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
        if (!status.is_training && wasTraining) {
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
    const metrics: TrainingMetrics = trainingStatus.metrics || {};
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

    const appendTrainingLog = useCallback((message: string) => {
        setTrainingStatus((prev) => ({
            ...prev,
            log: [...(prev.log || []), message],
        }));
    }, []);

    const { runTrainingAction } = useTrainingActionRunner({
        setLoading: setIsLoading,
        appendLog: appendTrainingLog,
    });

    // Handlers for training actions
    const handleConfirmTraining = useCallback(async () => {
        // Include the selected dataset label in the configuration
        const trainingConfig = {
            ...config,
            dataset_label: selectedDatasetLabel || undefined,
            dataset_hash: selectedDatasetHash || undefined,
        };
        await runTrainingAction({
            action: () => startTraining(trainingConfig),
            successStatus: 'started',
            actionLabel: 'start training',
            onSuccess: (result) => {
                setShowNewTrainingWizard(false);
                startPolling(result.poll_interval ?? 1.0);
                checkStatus(); // Immediately check status to update UI
            },
        });
    }, [checkStatus, config, runTrainingAction, selectedDatasetHash, selectedDatasetLabel, startPolling]);

    const handleDatasetSelect = useCallback((label: string) => {
        setSelectedDatasetLabel(label);
        const match = processedDatasets.find((dataset) => dataset.dataset_label === label);
        setSelectedDatasetHash(match?.dataset_hash ?? null);
    }, [processedDatasets]);

    const handleConfirmResume = useCallback(async () => {
        await runTrainingAction({
            action: () => resumeTraining(resumeConfig),
            successStatus: 'started',
            actionLabel: 'resume training',
            onSuccess: (result) => {
                setShowResumeTrainingWizard(false);
                startPolling(result.poll_interval ?? 1.0);
                checkStatus(); // Immediately check status to update UI
            },
        });
    }, [checkStatus, resumeConfig, runTrainingAction, startPolling]);

    const handleStopTraining = useCallback(async () => {
        await runTrainingAction({
            action: () => stopTraining(),
            successStatus: 'stopped',
            actionLabel: 'stop training',
            onSuccess: () => {
                checkStatus(); // Immediately check status to update UI
            },
        });
    }, [checkStatus, runTrainingAction]);

    return (
        <div className="ml-page">
            <div className="ml-header">
                <h2>SCADS Model Training</h2>
                <p className="ml-subtitle">Configure and monitor your training sessions</p>
            </div>

            <DatasetBuilderCard onDatasetBuilt={loadProcessedDatasets} />

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
                    <TrainingHistoryChartPanel
                        title="LOSS"
                        hasHistory={hasHistory}
                        history={history}
                        primaryLine={{
                            dataKey: 'loss',
                            color: CHART_COLORS.loss,
                            name: 'Train Loss',
                        }}
                        secondaryLine={{
                            dataKey: 'val_loss',
                            color: CHART_COLORS.valLoss,
                            name: 'Val Loss',
                        }}
                        placeholderHint="Loss metrics will appear once training starts."
                    />
                    <TrainingHistoryChartPanel
                        title={metricTitle}
                        hasHistory={hasHistory}
                        history={history}
                        primaryLine={{
                            dataKey: metricKey,
                            color: CHART_COLORS.metric,
                            name: `Train ${metricLabel}`,
                        }}
                        secondaryLine={{
                            dataKey: valMetricKey,
                            color: CHART_COLORS.valMetric,
                            name: `Val ${metricLabel}`,
                        }}
                        placeholderHint="Validation metrics will appear once training starts."
                        yAxisDomain={metricAsPercent ? [0, 1] : ['auto', 'auto']}
                    />
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
