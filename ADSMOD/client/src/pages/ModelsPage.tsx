import React, { useState, useCallback, useMemo } from 'react';
import { ModelCard } from '../components/ModelCard';
import { NumberInput } from '../components/UIComponents';
import { ADSORPTION_MODELS } from '../adsorptionModels';
import type { ModelParameters, FittingPayload } from '../types';

type OptimizationMethod = FittingPayload['optimization_method'];

interface OptimizationMethodOption {
    value: OptimizationMethod;
    label: string;
}

const OPTIMIZATION_METHOD_OPTIONS: readonly OptimizationMethodOption[] = [
    { value: 'LSS', label: 'Least Squares (LSS)' },
    { value: 'BFGS', label: 'BFGS' },
    { value: 'L-BFGS-B', label: 'L-BFGS-B' },
    { value: 'Nelder-Mead', label: 'Nelder-Mead' },
    { value: 'Powell', label: 'Powell' },
];

const parseOptimizationMethod = (value: string): OptimizationMethod | null => {
    const option = OPTIMIZATION_METHOD_OPTIONS.find((candidate) => candidate.value === value);
    return option?.value ?? null;
};

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

    const handleOptimizationMethodChange = useCallback(
        (event: React.ChangeEvent<HTMLSelectElement>) => {
            const nextMethod = parseOptimizationMethod(event.target.value);
            if (nextMethod) {
                onOptimizationMethodChange(nextMethod);
            }
        },
        [onOptimizationMethodChange]
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
            <div className="fitting-config-panel">
                <div className="models-header-row">
                    <div className="models-title-block">
                        <h3>Fitting Configuration</h3>
                        <p>Configure the optimizer and run the fit.</p>
                    </div>

                </div>

                <div className="fitting-main-layout">
                    <div className="fitting-controls-column">
                        <div className="fitting-controls-row">
                            <div className="control-group">
                                <label className="field-label">Dataset</label>
                                <div className="fitting-dataset-row">
                                    <select
                                        value={selectedDataset || ''}
                                        onChange={(e) => onDatasetSelect(e.target.value || null)}
                                        className={`select-input fitting-dataset-select ${useNistData ? 'is-disabled' : ''}`}
                                        disabled={useNistData}
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
                                    <label className="fitting-dataset-checkbox">
                                        <input
                                            type="checkbox"
                                            checked={useNistData}
                                            onChange={(e) => onUseNistDataChange(e.target.checked)}
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
                                    onChange={handleOptimizationMethodChange}
                                    className="select-input"
                                >
                                    {OPTIMIZATION_METHOD_OPTIONS.map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.label}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div className="control-group">
                                <button className="primary fitting-action-primary" onClick={onStartFitting} type="button">
                                    Start Fitting
                                </button>
                                <button className="secondary fitting-action-secondary" onClick={onResetFittingStatus} type="button">
                                    Reset Log
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="fitting-status-column">
                        <div className="fitting-status-box">
                            <div className="status-label">Fitting Log:</div>
                            <pre className="status-text">{fittingStatus || 'Ready to start...'}</pre>
                        </div>
                    </div>
                </div>
            </div>

            <hr className="section-separator" />

            <div className="models-grid-header">
                <h3>Select Adsorption Models</h3>
            </div>

            <div className="models-grid">{gridCells}</div>
        </div>
    );
};

export default ModelsPage;
