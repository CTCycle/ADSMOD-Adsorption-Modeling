import React, { useState, useEffect } from 'react';
import { NumberInput } from './UIComponents';
import { buildTrainingDataset, getTrainingDatasetInfo, clearTrainingDataset } from '../services';
import type { DatasetBuildConfig, DatasetFullInfo } from '../types';

interface DatasetBuilderCardProps {
    onDatasetBuilt?: () => void;
}

export const DatasetBuilderCard: React.FC<DatasetBuilderCardProps> = ({ onDatasetBuilt }) => {
    // Build configuration state
    const [sampleSize, setSampleSize] = useState(1.0);
    const [validationSize, setValidationSize] = useState(0.2);
    const [minMeasurements, setMinMeasurements] = useState(1);
    const [maxMeasurements, setMaxMeasurements] = useState(30);
    const [smileSequenceSize, setSmileSequenceSize] = useState(20);
    const [maxPressure, setMaxPressure] = useState(10000);
    const [maxUptake, setMaxUptake] = useState(20);

    // UI state
    const [isBuilding, setIsBuilding] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [datasetInfo, setDatasetInfo] = useState<DatasetFullInfo | null>(null);

    // Load existing dataset info on mount
    useEffect(() => {
        loadDatasetInfo();
    }, []);

    const loadDatasetInfo = async () => {
        const info = await getTrainingDatasetInfo();
        setDatasetInfo(info);
    };

    const handleBuildDataset = async () => {
        setIsBuilding(true);
        setStatusMessage('Building dataset...');

        const config: DatasetBuildConfig = {
            sample_size: sampleSize,
            validation_size: validationSize,
            min_measurements: minMeasurements,
            max_measurements: maxMeasurements,
            smile_sequence_size: smileSequenceSize,
            max_pressure: maxPressure,
            max_uptake: maxUptake,
            source_datasets: ['SINGLE_COMPONENT_ADSORPTION'],
        };

        const result = await buildTrainingDataset(config);

        if (result.success) {
            setStatusMessage(`✓ ${result.message} (${result.train_samples} train, ${result.validation_samples} val)`);
            await loadDatasetInfo();
            onDatasetBuilt?.();
        } else {
            setStatusMessage(`✗ ${result.message}`);
        }

        setIsBuilding(false);
    };

    const handleClearDataset = async () => {
        const result = await clearTrainingDataset();
        if (result.success) {
            setStatusMessage('Dataset cleared');
            setDatasetInfo(null);
            await loadDatasetInfo();
        } else {
            setStatusMessage(`✗ ${result.message}`);
        }
    };

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-icon">📦</span>
                <h3>Dataset Builder</h3>
                {datasetInfo?.available && (
                    <span className="badge success" style={{ marginLeft: 'auto' }}>
                        {datasetInfo.total_samples} samples
                    </span>
                )}
            </div>
            <div className="card-content">
                {/* Current Dataset Info */}
                {datasetInfo?.available && (
                    <div className="dataset-info-section" style={{
                        marginBottom: '1rem',
                        padding: '0.75rem',
                        background: 'var(--surface-50)',
                        borderRadius: '8px',
                        border: '1px solid var(--border-light)'
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: 500, color: 'var(--text-secondary)' }}>Current Dataset</span>
                            <button
                                className="button-icon"
                                onClick={handleClearDataset}
                                title="Clear dataset"
                                style={{ color: 'var(--error-500)', padding: '0.25rem' }}
                            >
                                🗑️
                            </button>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.85rem' }}>
                            <span>Train: <strong>{datasetInfo.train_samples}</strong></span>
                            <span>Val: <strong>{datasetInfo.validation_samples}</strong></span>
                        </div>
                    </div>
                )}

                {/* Build Parameters */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
                    <NumberInput
                        label="Sample Size"
                        value={sampleSize}
                        onChange={setSampleSize}
                        min={0.01}
                        max={1.0}
                        step={0.01}
                        precision={2}
                    />
                    <NumberInput
                        label="Validation %"
                        value={validationSize}
                        onChange={setValidationSize}
                        min={0.05}
                        max={0.5}
                        step={0.05}
                        precision={2}
                    />
                    <NumberInput
                        label="SMILE Length"
                        value={smileSequenceSize}
                        onChange={setSmileSequenceSize}
                        min={5}
                        max={100}
                        step={5}
                        precision={0}
                    />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
                    <NumberInput
                        label="Min Measurements"
                        value={minMeasurements}
                        onChange={setMinMeasurements}
                        min={1}
                        max={50}
                        step={1}
                        precision={0}
                    />
                    <NumberInput
                        label="Max Measurements"
                        value={maxMeasurements}
                        onChange={setMaxMeasurements}
                        min={5}
                        max={500}
                        step={5}
                        precision={0}
                    />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
                    <NumberInput
                        label="Max Pressure (kPa)"
                        value={maxPressure}
                        onChange={setMaxPressure}
                        min={100}
                        max={100000}
                        step={1000}
                        precision={0}
                    />
                    <NumberInput
                        label="Max Uptake (mol/g)"
                        value={maxUptake}
                        onChange={setMaxUptake}
                        min={1}
                        max={1000}
                        step={1}
                        precision={1}
                    />
                </div>

                {/* Build Button */}
                <button
                    className="button primary"
                    onClick={handleBuildDataset}
                    disabled={isBuilding}
                    style={{ width: '100%', justifyContent: 'center' }}
                >
                    {isBuilding ? '⏳ Building...' : '🔨 Build Dataset'}
                </button>

                {/* Status Message */}
                {statusMessage && (
                    <div style={{
                        marginTop: '0.75rem',
                        padding: '0.5rem',
                        borderRadius: '6px',
                        fontSize: '0.85rem',
                        background: statusMessage.startsWith('✓') ? 'var(--success-50)' :
                            statusMessage.startsWith('✗') ? 'var(--error-50)' : 'var(--surface-100)',
                        color: statusMessage.startsWith('✓') ? 'var(--success-700)' :
                            statusMessage.startsWith('✗') ? 'var(--error-700)' : 'var(--text-secondary)',
                    }}>
                        {statusMessage}
                    </div>
                )}
            </div>
        </div>
    );
};

export default DatasetBuilderCard;
