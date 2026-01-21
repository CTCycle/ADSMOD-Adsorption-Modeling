import React, { useState, useCallback, useMemo } from 'react';
import { ModelCard } from './ModelCard';
import { NumberInput } from './UIComponents';
import { ADSORPTION_MODELS } from '../adsorptionModels';
import type { ModelParameters, FittingPayload } from '../types';

type OptimizationMethod = FittingPayload['optimization_method'];

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

interface ModelsPageProps {
    modelStates: Record<string, ModelState>;
    onParametersChange: (modelName: string, parameters: ModelParameters) => void;
    onToggle: (modelName: string, enabled: boolean) => void;
    maxIterations: number;
    onMaxIterationsChange: (value: number) => void;
    optimizationMethod: OptimizationMethod;
    onOptimizationMethodChange: (value: OptimizationMethod) => void;
    fittingStatus: string;
    onStartFitting: () => void;
    onResetFittingStatus: () => void;
    availableDatasets: string[];
    selectedDataset: string | null;
    onDatasetSelect: (name: string | null) => void;
    useNistData: boolean;
    onUseNistDataChange: (value: boolean) => void;
}

/**
 * ModelsPage (ModelsPanel): Container component that manages fitting configuration and 9 adsorption model cards.
 * 
 * New Structure:
 * 1. Fitting Panel: Max iterations, Method, Start/Reset buttons, Logs.
 * 2. Separator
 * 3. Models Grid: 3x3 grid of model cards.
 */
export const ModelsPage: React.FC<ModelsPageProps> = ({
    modelStates,
    onParametersChange,
    onToggle,
    maxIterations,
    onMaxIterationsChange,
    optimizationMethod,
    onOptimizationMethodChange,
    fittingStatus,
    onStartFitting,
    onResetFittingStatus,
    availableDatasets,
    selectedDataset,
    onDatasetSelect,
    useNistData,
    onUseNistDataChange,
}) => {
    // Single expanded card strategy: only one card can be expanded at a time
    const [expandedId, setExpandedId] = useState<string | null>(null);

    // Handle card expand/collapse toggle
    const handleCardToggle = useCallback((modelId: string) => {
        setExpandedId((prev) => (prev === modelId ? null : modelId));
    }, []);

    // Handle model enabled/disabled toggle
    const handleEnabledChange = useCallback(
        (modelName: string, enabled: boolean) => {
            onToggle(modelName, enabled);
        },
        [onToggle]
    );

    // Handle configuration change
    const handleConfigChange = useCallback(
        (modelName: string, config: ModelParameters) => {
            onParametersChange(modelName, config);
        },
        [onParametersChange]
    );

    // Create a 3x3 grid (9 cells total) using the ADSORPTION_MODELS data
    const gridCells = useMemo(() => {
        return Array(9)
            .fill(null)
            .map((_, index) => {
                const model = ADSORPTION_MODELS[index];
                if (!model) {
                    return <div key={`empty-${index}`} className="model-grid-card empty" />;
                }

                const state = modelStates[model.name] || {
                    enabled: true,
                    config: {},
                };

                return (
                    <ModelCard
                        key={model.id}
                        model={model}
                        isExpanded={expandedId === model.id}
                        isEnabled={state.enabled}
                        currentConfig={state.config}
                        onToggle={handleCardToggle}
                        onEnabledChange={handleEnabledChange}
                        onConfigChange={handleConfigChange}
                    />
                );
            });
    }, [expandedId, modelStates, handleCardToggle, handleEnabledChange, handleConfigChange]);

    return (
        <div className="models-page">
            <div className="fitting-config-panel" style={{ marginBottom: '3rem' }}>
                <div className="models-header-row">
                    <div className="models-title-block">
                        <h3>Fitting Configuration</h3>
                        <p>Configure the optimizer and run the fit.</p>
                    </div>

                </div>

                <div className="fitting-main-layout" style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 380px) 1fr', gap: '2rem', marginTop: '1.5rem' }}>
                    <div className="fitting-controls-column">
                        <div className="fitting-controls-row" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                            <div className="control-group">
                                <label className="field-label">Dataset</label>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                    <select
                                        value={selectedDataset || ''}
                                        onChange={(e) => onDatasetSelect(e.target.value || null)}
                                        className="select-input"
                                        disabled={useNistData}
                                        style={useNistData ? { opacity: 0.5, cursor: 'not-allowed', flex: 1 } : { flex: 1 }}
                                    >
                                        <option value="">
                                            {availableDatasets.length === 0 ? 'No datasets available' : 'Select a dataset'}
                                        </option>
                                        {availableDatasets.map((name) => (
                                            <option key={name} value={name}>
                                                {name}
                                            </option>
                                        ))}
                                    </select>
                                    <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                                        <input
                                            type="checkbox"
                                            checked={useNistData}
                                            onChange={(e) => onUseNistDataChange(e.target.checked)}
                                            style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                                        />
                                        NIST Data
                                    </label>
                                </div>
                            </div>
                            <div className="control-group">
                                <NumberInput
                                    label="Max iterations"
                                    value={maxIterations}
                                    onChange={onMaxIterationsChange}
                                    min={1}
                                    max={1000000}
                                    step={1}
                                    precision={0}
                                />
                            </div>
                            <div className="control-group">
                                <label className="field-label">Optimization method</label>
                                <select
                                    value={optimizationMethod}
                                    onChange={(e) => onOptimizationMethodChange(e.target.value as OptimizationMethod)}
                                    className="select-input"
                                >
                                    <option value="LSS">Least Squares (LSS)</option>
                                    <option value="BFGS">BFGS</option>
                                    <option value="L-BFGS-B">L-BFGS-B</option>
                                    <option value="Nelder-Mead">Nelder-Mead</option>
                                    <option value="Powell">Powell</option>
                                </select>
                            </div>
                            <div className="control-group">
                                <button className="primary" onClick={onStartFitting} style={{ width: '100%', justifyContent: 'center', marginBottom: '0.75rem' }}>
                                    Start Fitting
                                </button>
                                <button
                                    onClick={onResetFittingStatus}
                                    style={{
                                        width: '100%',
                                        justifyContent: 'center',
                                        padding: '0.6rem',
                                        background: 'transparent',
                                        border: '1px solid #cbd5e1',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        color: '#475569',
                                        fontSize: '0.9rem',
                                        fontWeight: 500
                                    }}
                                >
                                    Reset Log
                                </button>
                            </div>
                        </div>
                    </div>

                    <div style={{ position: 'relative' }}>
                        <div className="fitting-status-box" style={{
                            display: 'flex',
                            flexDirection: 'column',
                            position: 'absolute',
                            inset: 0,
                            height: '100%'
                        }}>
                            <div className="status-label" style={{ marginBottom: '0.5rem', fontWeight: 500, color: 'var(--slate-700)' }}>Fitting Log:</div>
                            <pre className="status-text" style={{
                                flex: 1,
                                minHeight: 0,
                                background: '#0f172a',
                                color: '#e2e8f0',
                                padding: '1rem',
                                borderRadius: 'var(--radius-md)',
                                overflowY: 'auto',
                                overflowX: 'hidden',
                                fontFamily: 'monospace',
                                fontSize: '0.9rem'
                            }}>{fittingStatus || 'Ready to start...'}</pre>
                        </div>
                    </div>
                </div>
            </div>

            <hr className="section-separator" style={{ margin: '3rem 0', opacity: 0.5 }} />

            <div className="models-grid-header" style={{ marginBottom: '2rem' }}>
                <h3>Select Adsorption Models</h3>
            </div>

            <div className="models-grid">{gridCells}</div>
        </div>
    );
};

export default ModelsPage;
