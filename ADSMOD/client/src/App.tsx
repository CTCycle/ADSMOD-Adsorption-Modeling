import { useCallback, useEffect, useState } from 'react';
import { Sidebar, PageId } from './components/Sidebar';
import { ConfigPage } from './pages/ConfigPage';
import { ModelsPage } from './pages/ModelsPage';
import { DatabaseBrowserPage, initialDatabaseBrowserState } from './pages/DatabaseBrowserPage';
import { MachineLearningPage } from './pages/MachineLearningPage';
import type { DatabaseBrowserState } from './pages/DatabaseBrowserPage';
import { ADSORPTION_MODELS } from './adsorptionModels';
import { loadDataset, startFitting, fetchDatasetNames, fetchDatasetByName, fetchNistDataForFitting } from './services';
import type { DatasetPayload, FittingPayload, ModelParameters, ModelConfiguration } from './types';
import './index.css';

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

type OptimizationMethod = FittingPayload['optimization_method'];

const initialMountedPages: Record<PageId, boolean> = {
    config: true,
    models: false,
    analysis: false,
    browser: false,
};

const formatFileSize = (bytes: number): string => {
    const kb = Math.max(1, Math.round(bytes / 1024));
    return `${kb} kb`;
};

function App() {
    const [currentPage, setCurrentPage] = useState<PageId>('config');
    const [mountedPages, setMountedPages] = useState<Record<PageId, boolean>>(initialMountedPages);
    const [maxIterations, setMaxIterations] = useState(10000);
    const [optimizationMethod, setOptimizationMethod] = useState<OptimizationMethod>('LSS');
    const [datasetStats, setDatasetStats] = useState('No dataset loaded.');
    const [nistStatusMessage, setNistStatusMessage] = useState('NIST-A updates will appear here.');
    const [fittingStatus, setFittingStatus] = useState('');
    const [dataset, setDataset] = useState<DatasetPayload | null>(null);
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [datasetSamples, setDatasetSamples] = useState(0);
    const [pendingFile, setPendingFile] = useState<File | null>(null);
    const [pendingFileSize, setPendingFileSize] = useState<string | null>(null);
    const [isDatasetUploading, setIsDatasetUploading] = useState(false);
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

    // Dataset selector state for fitting page
    const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
    const [useNistData, setUseNistData] = useState(false);

    // Load available dataset names on mount
    useEffect(() => {
        const loadDatasetNames = async () => {
            const result = await fetchDatasetNames();
            if (!result.error) {
                setAvailableDatasets(result.names);
            }
        };
        void loadDatasetNames();
    }, []);

    const handlePageChange = useCallback((page: PageId) => {
        setCurrentPage(page);
        setMountedPages((prev) => (prev[page] ? prev : { ...prev, [page]: true }));
    }, []);

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

    const handleDatasetPreload = useCallback((file: File) => {
        setPendingFile(file);
        setPendingFileSize(formatFileSize(file.size));
    }, []);

    const handleDatasetUpload = useCallback(async () => {
        if (!pendingFile) {
            setDatasetStats('[ERROR] Please select a dataset file before uploading.');
            return;
        }
        setIsDatasetUploading(true);
        setDatasetStats('[INFO] Uploading dataset...');
        const result = await loadDataset(pendingFile);

        if (result.dataset) {
            setDataset(result.dataset);
            setDatasetName(result.dataset.dataset_name);
            const recordCount = Array.isArray(result.dataset.records) ? result.dataset.records.length : 0;
            setDatasetSamples(recordCount);
            setPendingFile(null);
            setPendingFileSize(null);
        }

        setDatasetStats(result.message);
        setIsDatasetUploading(false);

        // Refresh available datasets after upload
        const namesResult = await fetchDatasetNames();
        if (!namesResult.error) {
            setAvailableDatasets(namesResult.names);
            // Auto-select the newly uploaded dataset
            if (result.dataset?.dataset_name) {
                setSelectedDataset(result.dataset.dataset_name);
            }
        }
    }, [pendingFile]);

    const handleNistStatusUpdate = useCallback((message: string) => {
        setNistStatusMessage(message);
    }, []);

    const handleResetFittingStatus = useCallback(() => {
        setFittingStatus('');
    }, []);

    const handleStartFitting = useCallback(async () => {
        // Determine which dataset to use
        let fittingDataset: DatasetPayload | null = null;

        if (useNistData) {
            setFittingStatus('[INFO] Loading NIST single-component data...');
            const nistResult = await fetchNistDataForFitting();
            if (nistResult.error || !nistResult.dataset) {
                setFittingStatus(`[ERROR] ${nistResult.error || 'Failed to load NIST data.'}`);
                return;
            }
            fittingDataset = nistResult.dataset;
        } else {
            if (selectedDataset) {
                if (!dataset || dataset.dataset_name !== selectedDataset) {
                    setFittingStatus(`[INFO] Loading dataset "${selectedDataset}"...`);
                    const lookup = await fetchDatasetByName(selectedDataset);
                    if (lookup.error || !lookup.dataset) {
                        setFittingStatus(`[ERROR] ${lookup.error || 'Failed to load dataset.'}`);
                        return;
                    }
                    fittingDataset = lookup.dataset;
                    setDataset(lookup.dataset);
                    setDatasetName(lookup.dataset.dataset_name);
                    const recordCount = Array.isArray(lookup.dataset.records) ? lookup.dataset.records.length : 0;
                    setDatasetSamples(recordCount);
                    if (lookup.summary) {
                        setDatasetStats(lookup.summary);
                    }
                } else {
                    fittingDataset = dataset;
                }
            } else {
                if (!dataset) {
                    setFittingStatus('[ERROR] Please load a dataset before starting the fitting process.');
                    return;
                }
                fittingDataset = dataset;
            }
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
            dataset: fittingDataset,
        };

        const result = await startFitting(payload);
        setFittingStatus(result.message);
    }, [dataset, modelStates, maxIterations, optimizationMethod, selectedDataset, useNistData]);

    return (
        <div className="app-container">
            <header className="app-header">
                <div className="header-content">
                    <h1>ADSMOD Adsorption Modeling</h1>
                </div>
            </header>

            <div className="app-layout">
                <Sidebar currentPage={currentPage} onPageChange={handlePageChange} />

                <main className="app-main">
                    {mountedPages.config && (
                        <section hidden={currentPage !== 'config'} aria-hidden={currentPage !== 'config'}>
                            <ConfigPage
                                datasetStats={datasetStats}
                                nistStatusMessage={nistStatusMessage}
                                datasetName={datasetName}
                                datasetSamples={datasetSamples}
                                pendingFileName={pendingFile?.name ?? null}
                                pendingFileSize={pendingFileSize}
                                onDatasetPreload={handleDatasetPreload}
                                onDatasetUpload={handleDatasetUpload}
                                isDatasetUploading={isDatasetUploading}
                                onNistStatusUpdate={handleNistStatusUpdate}
                            />
                        </section>
                    )}

                    {mountedPages.models && (
                        <section hidden={currentPage !== 'models'} aria-hidden={currentPage !== 'models'}>
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
                                availableDatasets={availableDatasets}
                                selectedDataset={selectedDataset}
                                onDatasetSelect={setSelectedDataset}
                                useNistData={useNistData}
                                onUseNistDataChange={setUseNistData}
                            />
                        </section>
                    )}

                    {mountedPages.analysis && (
                        <section hidden={currentPage !== 'analysis'} aria-hidden={currentPage !== 'analysis'}>
                            <MachineLearningPage />
                        </section>
                    )}

                    {mountedPages.browser && (
                        <section hidden={currentPage !== 'browser'} aria-hidden={currentPage !== 'browser'}>
                            <DatabaseBrowserPage
                                state={databaseBrowserState}
                                onStateChange={setDatabaseBrowserState}
                            />
                        </section>
                    )}
                </main>
            </div>
        </div>
    );
}

export default App;
