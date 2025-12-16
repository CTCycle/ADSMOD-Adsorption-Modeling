import { useState, useCallback } from 'react';
import { Sidebar, PageId } from './components/Sidebar';
import { ConfigPage } from './components/ConfigPage';
import { ModelsPage } from './components/ModelsPage';
import { DatabaseBrowserPage, initialDatabaseBrowserState } from './components/DatabaseBrowserPage';
import { MachineLearningPage } from './components/MachineLearningPage';
import type { DatabaseBrowserState } from './components/DatabaseBrowserPage';
import { ADSORPTION_MODELS } from './adsorptionModels';
import { loadDataset, startFitting } from './services';
import type { DatasetPayload, FittingPayload, ModelParameters, ModelConfiguration } from './types';
import './index.css';

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

type OptimizationMethod = FittingPayload['optimization_method'];

function App() {
    const [currentPage, setCurrentPage] = useState<PageId>('config');
    const [maxIterations, setMaxIterations] = useState(10000);
    const [optimizationMethod, setOptimizationMethod] = useState<OptimizationMethod>('LSS');
    const [datasetStats, setDatasetStats] = useState('No dataset loaded.');
    const [fittingStatus, setFittingStatus] = useState('');
    const [dataset, setDataset] = useState<DatasetPayload | null>(null);
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [datasetSamples, setDatasetSamples] = useState(0);
    const [modelStates, setModelStates] = useState<Record<string, ModelState>>(() => {
        const initial: Record<string, ModelState> = {};
        ADSORPTION_MODELS.forEach((model) => {
            const config: ModelParameters = {};
            Object.entries(model.parameterDefaults).forEach(([paramName, [min, max]]) => {
                config[paramName] = { min, max };
            });
            initial[model.name] = { enabled: true, config };
        });
        return initial;
    });

    // Database browser state - lifted for persistence across page navigation
    const [databaseBrowserState, setDatabaseBrowserState] = useState<DatabaseBrowserState>(initialDatabaseBrowserState);

    const handleModelToggle = useCallback((modelName: string, enabled: boolean) => {
        setModelStates((prev) => ({
            ...prev,
            [modelName]: { ...prev[modelName], enabled },
        }));
    }, []);

    const handleParametersChange = useCallback((modelName: string, config: ModelParameters) => {
        setModelStates((prev) => ({
            ...prev,
            [modelName]: { ...prev[modelName], config },
        }));
    }, []);

    const handleDatasetUpload = useCallback(async (file: File) => {
        setDatasetStats('[INFO] Uploading dataset...');
        const result = await loadDataset(file);
        setDataset(result.dataset);

        if (result.dataset) {
            setDatasetName(file.name);
            const recordCount = Array.isArray(result.dataset.records) ? result.dataset.records.length : 0;
            setDatasetSamples(recordCount);
        } else {
            setDatasetName(null);
            setDatasetSamples(0);
        }

        setDatasetStats(result.message);
    }, []);

    const handleResetFittingStatus = useCallback(() => {
        setFittingStatus('');
    }, []);

    const handleStartFitting = useCallback(async () => {
        if (!dataset) {
            setFittingStatus('[ERROR] Please load a dataset before starting the fitting process.');
            return;
        }

        const selectedModels = Object.entries(modelStates)
            .filter(([_, state]) => state.enabled)
            .map(([name]) => name);

        if (selectedModels.length === 0) {
            setFittingStatus('[ERROR] Please select at least one model before starting the fitting process.');
            return;
        }

        // Build parameter bounds configuration
        const parameterBounds: Record<string, ModelConfiguration> = {};
        selectedModels.forEach((modelName) => {
            const state = modelStates[modelName];
            const modelConfig: ModelConfiguration = {
                min: {},
                max: {},
                initial: {},
            };

            Object.entries(state.config).forEach(([paramName, bounds]) => {
                let { min, max } = bounds;

                // Validate and swap if needed
                if (max < min) {
                    [min, max] = [max, min];
                }

                const midpoint = min + (max - min) / 2;
                modelConfig.min[paramName] = min;
                modelConfig.max[paramName] = max;
                modelConfig.initial[paramName] = midpoint;
            });

            parameterBounds[modelName] = modelConfig;
        });

        setFittingStatus('[INFO] Starting fitting process...');

        const payload: FittingPayload = {
            max_iterations: Math.max(1, Math.round(maxIterations)),
            optimization_method: optimizationMethod,
            parameter_bounds: parameterBounds,
            dataset,
        };

        const result = await startFitting(payload);
        setFittingStatus(result.message);
    }, [dataset, modelStates, maxIterations, optimizationMethod]);

    const methodLabels: Record<OptimizationMethod, string> = {
        LSS: 'Least Squares',
        BFGS: 'BFGS',
        'L-BFGS-B': 'L-BFGS-B',
        'Nelder-Mead': 'Nelder-Mead',
        Powell: 'Powell',
    };

    const optimizationLabel = methodLabels[optimizationMethod];
    const datasetLabel = datasetName || 'none';

    return (
        <div className="app-container">
            <header className="app-header">
                <div className="header-content">
                    <h1>ADSMOD Model Fitting</h1>
                    <div className="header-meta">
                        <span className="meta-chip">Dataset: {datasetLabel}</span>
                        <span className="meta-separator">|</span>
                        <span className="meta-chip">Method: {optimizationLabel}</span>
                    </div>
                </div>
            </header>

            <div className="app-layout">
                <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />

                <main className="app-main">
                    {currentPage === 'config' && (
                        <ConfigPage
                            datasetStats={datasetStats}
                            datasetName={datasetName}
                            datasetSamples={datasetSamples}
                            onDatasetUpload={handleDatasetUpload}
                        />
                    )}

                    {currentPage === 'models' && (
                        <ModelsPage
                            modelStates={modelStates}
                            onParametersChange={handleParametersChange}
                            onToggle={handleModelToggle}
                            maxIterations={maxIterations}
                            onMaxIterationsChange={setMaxIterations}
                            optimizationMethod={optimizationMethod}
                            onOptimizationMethodChange={setOptimizationMethod}
                            fittingStatus={fittingStatus}
                            onStartFitting={handleStartFitting}
                            onResetFittingStatus={handleResetFittingStatus}
                        />
                    )}

                    {currentPage === 'analysis' && (
                        <MachineLearningPage />
                    )}

                    {currentPage === 'browser' && (
                        <DatabaseBrowserPage
                            state={databaseBrowserState}
                            onStateChange={setDatabaseBrowserState}
                        />
                    )}
                </main>
            </div>
        </div>
    );
}

export default App;
