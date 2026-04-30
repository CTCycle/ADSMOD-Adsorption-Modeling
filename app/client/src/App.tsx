import { useCallback, useEffect, useState } from 'react';
import { Sidebar, PageId } from './components/Sidebar';
import { ConfigPage } from './pages/ConfigPage';
import { ModelsPage } from './pages/ModelsPage';
import { MachineLearningPage } from './pages/MachineLearningPage';
import { ADSORPTION_MODELS } from './adsorptionModels';
import {
    loadDataset,
    startFittingJob,
    pollFittingJobUntilComplete,
    fetchDatasetNames,
    fetchDatasetByName,
    fetchNistDataForFitting,
} from './services';
import type { DatasetPayload, FittingPayload, ModelParameters, ModelConfiguration } from './types';
import './index.css';

interface ModelState {
    enabled: boolean;
    config: ModelParameters;
}

type OptimizationMethod = FittingPayload['optimization_method'];

const initialMountedPages: Record<PageId, boolean> = {
    source: true,
    fitting: false,
    training: false,
};

const NIST_DATASET_OPTION = '__NIST_A_COLLECTION__';

const formatFileSize = (bytes: number): string => {
    const kb = Math.max(1, Math.round(bytes / 1024));
    return `${kb} kb`;
};

const escapeMarkdownTableCell = (value: unknown): string => {
    return String(value).replace(/\|/g, '\\|').replace(/\n/g, ' ');
};

const inferColumnDtype = (values: unknown[]): string => {
    const firstValue = values.find((value) => value !== null && value !== undefined && !(typeof value === 'number' && Number.isNaN(value)));
    if (firstValue === undefined) {
        return 'unknown';
    }
    if (typeof firstValue === 'number') {
        return Number.isInteger(firstValue) ? 'int64' : 'float64';
    }
    if (typeof firstValue === 'boolean') {
        return 'bool';
    }
    return 'str';
};

const buildDatasetSummary = (payload: DatasetPayload): string => {
    const records = Array.isArray(payload.records) ? payload.records : [];
    const providedColumns = Array.isArray(payload.columns) ? payload.columns : [];
    const discoveredColumns = new Set<string>(providedColumns);
    records.forEach((row) => {
        Object.keys(row || {}).forEach((key) => discoveredColumns.add(key));
    });
    const orderedColumns = Array.from(discoveredColumns);

    let totalMissing = 0;
    const columnLines = orderedColumns.map((column) => {
        const values = records.map((row) => row?.[column]);
        const missing = values.filter(
            (value) => value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value))
        ).length;
        totalMissing += missing;
        const dtype = inferColumnDtype(values);
        return `| \`${escapeMarkdownTableCell(column)}\` | \`${escapeMarkdownTableCell(dtype)}\` | ${missing} |`;
    });

    return [
        '### Dataset overview',
        '',
        '| Metric | Value |',
        '|---|---:|',
        `| Rows | ${records.length} |`,
        `| Columns | ${orderedColumns.length} |`,
        `| NaN cells | ${totalMissing} |`,
        '',
        '### Column details',
        '',
        '| Column | Dtype | Missing |',
        '|---|---|---:|',
        ...columnLines,
    ].join('\n');
};

function App() {
    const [currentPage, setCurrentPage] = useState<PageId>('source');
    const [mountedPages, setMountedPages] = useState<Record<PageId, boolean>>(initialMountedPages);
    const [maxIterations, setMaxIterations] = useState(10000);
    const [optimizationMethod, setOptimizationMethod] = useState<OptimizationMethod>('LSS');
    const [datasetStats, setDatasetStats] = useState('No dataset loaded.');
    const [nistStatusMessage, setNistStatusMessage] = useState('NIST-A updates will appear here.');
    const [fittingStatus, setFittingStatus] = useState('');
    const [dataset, setDataset] = useState<DatasetPayload | null>(null);
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [datasetSamples, setDatasetSamples] = useState(0);
    const [datasetSizeKb, setDatasetSizeKb] = useState<string | null>(null);
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

    // Dataset selector state for fitting page
    const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

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
        const activeFileSize = pendingFileSize || formatFileSize(pendingFile.size);
        const result = await loadDataset(pendingFile);

        if (result.dataset) {
            setDataset(result.dataset);
            setDatasetName(result.dataset.dataset_name);
            setDatasetSizeKb(activeFileSize);
            const recordCount = Array.isArray(result.dataset.records) ? result.dataset.records.length : 0;
            setDatasetSamples(recordCount);
            setDatasetStats(buildDatasetSummary(result.dataset));
            setPendingFile(null);
            setPendingFileSize(null);
        } else {
            setDatasetStats(result.message);
        }
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
    }, [pendingFile, pendingFileSize]);

    const handleNistStatusUpdate = useCallback((message: string) => {
        setNistStatusMessage(message);
    }, []);

    const handleResetFittingStatus = useCallback(() => {
        setFittingStatus('');
    }, []);

    const handleStartFitting = useCallback(async () => {
        // Determine which dataset to use
        let fittingDataset: DatasetPayload | null = null;
        const useNistData = selectedDataset === NIST_DATASET_OPTION;

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
                    setDatasetStats(buildDatasetSummary(lookup.dataset));
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

        const { jobId, pollInterval, error } = await startFittingJob(payload);
        if (error || !jobId) {
            setFittingStatus(`[ERROR] ${error || 'Failed to start job.'}`);
            return;
        }

        const result = await pollFittingJobUntilComplete(jobId, pollInterval);
        setFittingStatus(result.message);
    }, [dataset, modelStates, maxIterations, optimizationMethod, selectedDataset]);

    return (
        <div className="app-container">
            <header className="app-header">
                <div className="header-content">
                    <div className="header-brand">
                        <img className="brand-logo" src="/favicon.png" alt="ADSMOD logo" />
                        <h1 className="brand-wordmark">ADSMOD</h1>
                    </div>
                    <Sidebar currentPage={currentPage} onPageChange={handlePageChange} />
                </div>
            </header>

            <main className="app-main">
                {mountedPages.source && (
                    <section hidden={currentPage !== 'source'} aria-hidden={currentPage !== 'source'}>
                        <ConfigPage
                            datasetStats={datasetStats}
                            nistStatusMessage={nistStatusMessage}
                            datasetName={datasetName}
                            datasetSizeKb={datasetSizeKb}
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

                {mountedPages.fitting && (
                    <section hidden={currentPage !== 'fitting'} aria-hidden={currentPage !== 'fitting'}>
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
                            nistOptionValue={NIST_DATASET_OPTION}
                        />
                    </section>
                )}

                {mountedPages.training && (
                    <section hidden={currentPage !== 'training'} aria-hidden={currentPage !== 'training'}>
                        <MachineLearningPage />
                    </section>
                )}
            </main>
        </div>
    );
}

export default App;

