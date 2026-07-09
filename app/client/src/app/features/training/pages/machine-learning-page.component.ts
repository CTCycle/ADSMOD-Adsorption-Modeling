import { Component, computed, effect, inject, signal } from '@angular/core';
import { ActivatedRoute, RouterLink, RouterLinkActive } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { map } from 'rxjs';
import { TrainingViewId, TrainingWorkspaceStore } from '../../../core/state/training-workspace.store';
import { InfoModalComponent } from '../../../shared/components/info-modal/info-modal.component';
import { MetricCardComponent } from '../../../shared/components/metric-card/metric-card.component';
import { DatasetBuilderCardComponent } from '../components/dataset-builder-card.component';
import { NewTrainingWizardComponent } from '../components/new-training-wizard.component';
import { ResumeTrainingWizardComponent } from '../components/resume-training-wizard.component';
import { TrainingHistoryChartPanelComponent } from '../components/training-history-chart-panel.component';
import { TrainingSetupRowComponent } from '../components/training-setup-row.component';
import { TrainingActionRunnerService } from '../services/training-action-runner.service';
import { buildCheckpointDetailsModalData, buildDatasetMetadataModalData } from '../services/training-modal-data';
import { TrainingStatusPollingService } from '../services/training-status-polling.service';
import { TrainingViewNavigationService } from '../services/training-view-navigation.service';
import type {
    ResumeTrainingConfig,
    TrainingConfig,
    TrainingHistoryPoint,
    TrainingMetricKey,
    TrainingMetrics,
    TrainingStatus,
} from '../../../models/training.model';
import { getTrainingStatus } from '../../../services/training.service';

interface TrainingViewSpec {
    id: TrainingViewId;
    label: string;
    description: string;
}

const TRAINING_VIEWS: readonly TrainingViewSpec[] = [
    { id: 'processing', label: 'Data Processing', description: 'Prepare adsorption datasets for model training.' },
    { id: 'datasets', label: 'Train datasets', description: 'Choose a processed dataset and configure a new training run.' },
    { id: 'checkpoints', label: 'Checkpoints', description: 'Review saved checkpoints and resume a training run.' },
    { id: 'dashboard', label: 'Training Dashboard', description: 'Track training progress, metrics, and logs in real time.' },
];

const isTrainingViewId = (value: string | null): value is TrainingViewId =>
    value === 'processing' || value === 'datasets' || value === 'checkpoints' || value === 'dashboard';

@Component({
    selector: 'adsmod-machine-learning-page',
    standalone: true,
    imports: [
        RouterLink,
        RouterLinkActive,
        DatasetBuilderCardComponent,
        TrainingSetupRowComponent,
        NewTrainingWizardComponent,
        ResumeTrainingWizardComponent,
        InfoModalComponent,
        MetricCardComponent,
        TrainingHistoryChartPanelComponent,
    ],
    template: `
        <main class="route-workspace route-workspace-training">
            <section class="route-rail route-rail-training training-rail" aria-label="Training workspace navigation">
                <div class="route-rail-brand">
                    <div class="route-rail-logo" aria-hidden="true">AD</div>
                    <div class="route-rail-wordmark">ADSMOD</div>
                </div>
                <nav class="training-view-toolbar" aria-label="Training views">
                    @for (view of views; track view.id) {
                        <a class="training-view-tab" [routerLink]="['/training', view.id]" routerLinkActive="active">
                            <span class="training-view-tab-icon" aria-hidden="true">
                                @switch (view.id) {
                                    @case ('processing') {
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                            <polyline points="7 10 12 15 17 10" />
                                            <line x1="12" y1="15" x2="12" y2="3" />
                                        </svg>
                                    }
                                    @case ('datasets') {
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                            <rect x="3" y="4" width="18" height="6" rx="1" />
                                            <rect x="3" y="14" width="18" height="6" rx="1" />
                                            <line x1="7" y1="7" x2="7.01" y2="7" />
                                            <line x1="7" y1="17" x2="7.01" y2="17" />
                                        </svg>
                                    }
                                    @case ('checkpoints') {
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                            <rect x="4" y="4" width="16" height="16" rx="2" />
                                            <path d="M8 9h8" />
                                            <path d="M8 13h8" />
                                            <path d="M8 17h5" />
                                        </svg>
                                    }
                                    @case ('dashboard') {
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                            <path d="M3 3v18h18" />
                                            <path d="M19 9l-5 5-4-4-3 3" />
                                        </svg>
                                    }
                                }
                            </span>
                            <span class="training-view-tab-label">{{ view.label }}</span>
                        </a>
                    }
                </nav>
            </section>

            <section class="route-canvas route-canvas-training training-view-panel">
                <div class="training-view-description">
                    <h2>{{ activeView().label }}</h2>
                    <p>{{ activeView().description }}</p>
                </div>

                @if (trainingAvailability() === 'checking') {
                    <section class="training-unavailable-card" aria-live="polite">
                        <h3>Checking ML service...</h3>
                        <p>Training availability is being verified.</p>
                    </section>
                } @else if (trainingAvailability() === 'unavailable') {
                    <section class="training-unavailable-card" aria-live="polite">
                        <h3>Training requires the optional ML service.</h3>
                        <p>Source and Fitting remain available in core-only mode. Start or install the ML service to use training workflows.</p>
                    </section>
                } @else {
                    @switch (activeView().id) {
                        @case ('processing') {
                            <div class="training-view-widget">
                                <adsmod-dataset-builder-card
                                    [showSectionHeading]="false"
                                    (datasetBuilt)="refreshWorkspace()"
                                    (workspaceChanged)="refreshWorkspace()"
                                />
                            </div>
                        }
                        @case ('datasets') {
                            <div class="training-view-widget">
                                <adsmod-training-setup-row
                                    [processedDatasets]="store.processedDatasets()"
                                    [checkpoints]="store.checkpoints()"
                                    [isTraining]="store.trainingStatus().is_training"
                                    viewMode="datasets"
                                    [showSectionHeading]="false"
                                    (refreshDatasetsRequested)="refreshDatasets()"
                                    (refreshCheckpointsRequested)="refreshCheckpoints()"
                                    (newTrainingRequested)="openNewTrainingWizard($event)"
                                    (resumeTrainingRequested)="openResumeTrainingWizard($event)"
                                    (datasetMetadataRequested)="viewDatasetMetadata($event)"
                                    (datasetDeleteRequested)="deleteDataset($event)"
                                    (checkpointDetailsRequested)="viewCheckpointDetails($event)"
                                    (checkpointDeleteRequested)="deleteCheckpoint($event)"
                                />
                            </div>
                        }
                        @case ('checkpoints') {
                            <div class="training-view-widget">
                                <adsmod-training-setup-row
                                    [processedDatasets]="store.processedDatasets()"
                                    [checkpoints]="store.checkpoints()"
                                    [isTraining]="store.trainingStatus().is_training"
                                    viewMode="checkpoints"
                                    [showSectionHeading]="false"
                                    (refreshDatasetsRequested)="refreshDatasets()"
                                    (refreshCheckpointsRequested)="refreshCheckpoints()"
                                    (newTrainingRequested)="openNewTrainingWizard($event)"
                                    (resumeTrainingRequested)="openResumeTrainingWizard($event)"
                                    (datasetMetadataRequested)="viewDatasetMetadata($event)"
                                    (datasetDeleteRequested)="deleteDataset($event)"
                                    (checkpointDetailsRequested)="viewCheckpointDetails($event)"
                                    (checkpointDeleteRequested)="deleteCheckpoint($event)"
                                />
                            </div>
                        }
                        @case ('dashboard') {
                        <div class="training-view-widget training-dashboard">
                            <div class="dashboard-header">
                                <div class="dashboard-title">
                                    <span class="dashboard-icon" aria-hidden="true">
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                            <path d="M3 17l6-6 4 4 8-8" />
                                            <path d="M15 7h6v6" />
                                        </svg>
                                    </span>
                                    <span>Training Dashboard</span>
                                </div>
                                <span class="training-status-badge" [class.active]="store.trainingStatus().is_training" [class.idle]="!store.trainingStatus().is_training">
                                    {{ store.trainingStatus().is_training ? 'Training in Progress' : 'Idle' }}
                                </span>
                            </div>

                            <div class="metrics-row">
                                <adsmod-metric-card label="EPOCH" [value]="epochMetric()" color="var(--slate-800)" />
                                <adsmod-metric-card label="TRAIN LOSS" [value]="trainLossMetric()" [color]="chartColors.loss" />
                                <adsmod-metric-card label="VAL LOSS" [value]="valLossMetric()" [color]="chartColors.valLoss" />
                                <adsmod-metric-card [label]="'TRAIN ' + metricLabel()" [value]="trainMetricValue()" [color]="chartColors.metric" />
                                <adsmod-metric-card [label]="'VAL ' + metricLabel()" [value]="valMetricValue()" [color]="chartColors.valMetric" />
                            </div>

                            <div class="dashboard-progress">
                                <div class="progress-label">
                                    <span>% Progress: {{ roundedProgress() }}%</span>
                                </div>
                                <div class="progress-bar-wrapper">
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill" [style.width.%]="store.trainingStatus().progress"></div>
                                    </div>
                                    <button class="stop-btn" type="button" [disabled]="!store.trainingStatus().is_training || store.actionLoading()" (click)="stopTraining()">
                                        Stop Training
                                    </button>
                                </div>
                            </div>

                            <div class="charts-container">
                                <adsmod-training-history-chart-panel
                                    title="LOSS"
                                    [hasHistory]="hasHistory()"
                                    [history]="history()"
                                    [primaryLine]="lossPrimaryLine"
                                    [secondaryLine]="lossSecondaryLine"
                                    placeholderHint="Loss metrics will appear once training starts."
                                />
                                <adsmod-training-history-chart-panel
                                    [title]="metricTitle()"
                                    [hasHistory]="hasHistory()"
                                    [history]="history()"
                                    [primaryLine]="metricPrimaryLine()"
                                    [secondaryLine]="metricSecondaryLine()"
                                    placeholderHint="Validation metrics will appear once training starts."
                                    [yAxisDomain]="metricAxisDomain()"
                                />
                            </div>

                            <div class="log-panel">
                                <div class="log-header">
                                    <span>Training Log</span>
                                    <button class="ghost-button" type="button" (click)="store.clearTrainingLog()">Clear</button>
                                </div>
                                <pre class="training-log-light">{{ trainingLog() }}</pre>
                            </div>
                        </div>
                        }
                    }
                }
            </section>

            @if (store.showNewTrainingWizard()) {
                <adsmod-new-training-wizard
                    [initialConfig]="store.config()"
                    [selectedDatasetLabel]="store.selectedDatasetLabel() || ''"
                    [isLoading]="store.actionLoading()"
                    (closed)="store.closeNewTrainingWizard()"
                    (confirmed)="confirmTraining($event)"
                />
            }

            @if (store.showResumeTrainingWizard()) {
                <adsmod-resume-training-wizard
                    [checkpoints]="store.checkpoints()"
                    [selectedCheckpointName]="store.resumeConfig().checkpoint_name"
                    [initialConfig]="store.resumeConfig()"
                    [isLoading]="store.actionLoading()"
                    (closed)="store.closeResumeTrainingWizard()"
                    (confirmed)="confirmResume($event)"
                />
            }

            <adsmod-info-modal
                [isOpen]="store.infoModalOpen()"
                [title]="store.infoModalTitle()"
                [data]="store.infoModalData()"
                (closed)="store.closeInfoModal()"
            />
        </main>
    `,
})
export class MachineLearningPageComponent {
    private readonly route = inject(ActivatedRoute);
    protected readonly store = inject(TrainingWorkspaceStore);
    private readonly actionRunner = inject(TrainingActionRunnerService);
    private readonly statusPolling = inject(TrainingStatusPollingService);
    private readonly viewNavigation = inject(TrainingViewNavigationService);
    protected readonly views = TRAINING_VIEWS;
    protected readonly chartColors = {
        loss: '#f59e0b',
        valLoss: '#2563eb',
        metric: '#16a34a',
        valMetric: '#9333ea',
    };
    protected readonly trainingAvailability = signal<'checking' | 'available' | 'unavailable'>('checking');
    protected readonly lossPrimaryLine = { dataKey: 'loss', color: this.chartColors.loss, name: 'Train Loss' } as const;
    protected readonly lossSecondaryLine = { dataKey: 'val_loss', color: this.chartColors.valLoss, name: 'Val Loss' } as const;
    private readonly routeView = toSignal(
        this.route.paramMap.pipe(map((params) => params.get('view'))),
        { initialValue: 'processing' }
    );
    protected readonly activeView = computed(() => {
        const viewId = this.routeView();
        return TRAINING_VIEWS.find((view) => view.id === (isTrainingViewId(viewId) ? viewId : 'processing')) || TRAINING_VIEWS[0];
    });
    protected readonly dashboardStatus = computed(() => {
        const status = this.store.trainingStatus();
        if (this.store.trainingStatusError()) {
            return this.store.trainingStatusError();
        }
        return status.is_training
            ? `Training epoch ${status.current_epoch} of ${status.total_epochs}.`
            : 'No training run is currently active.';
    });
    protected readonly metrics = computed<TrainingMetrics>(() => this.store.trainingStatus().metrics || {});
    protected readonly history = computed<TrainingHistoryPoint[]>(() => this.store.trainingStatus().history || []);
    protected readonly hasAccuracyMetric = computed(() =>
        this.history().some((entry) => typeof entry.accuracy === 'number' || typeof entry.val_accuracy === 'number')
        || typeof this.metrics().accuracy === 'number'
        || typeof this.metrics().val_accuracy === 'number'
    );
    protected readonly metricKey = computed(() => this.hasAccuracyMetric() ? 'accuracy' : 'masked_r2');
    protected readonly valMetricKey = computed(() => this.hasAccuracyMetric() ? 'val_accuracy' : 'val_masked_r2');
    protected readonly metricLabel = computed(() => this.hasAccuracyMetric() ? 'ACC' : 'R2');
    protected readonly metricTitle = computed(() => this.hasAccuracyMetric() ? 'ACCURACY' : 'R2 SCORE');
    protected readonly metricAxisDomain = computed<[number, number] | ['auto', 'auto']>(() => this.hasAccuracyMetric() ? [0, 1] : ['auto', 'auto']);
    protected readonly hasHistory = computed(() => this.history().length > 0);
    protected readonly epochMetric = computed(() => `${this.store.trainingStatus().current_epoch} / ${this.store.trainingStatus().total_epochs || this.store.config().epochs}`);
    protected readonly trainLossMetric = computed(() => this.formatLossValue(this.metrics().loss));
    protected readonly valLossMetric = computed(() => this.formatLossValue(this.metrics().val_loss));
    protected readonly trainMetricValue = computed(() => this.formatMetricValue(this.metrics()[this.metricKey()], this.hasAccuracyMetric()));
    protected readonly valMetricValue = computed(() => this.formatMetricValue(this.metrics()[this.valMetricKey()], this.hasAccuracyMetric()));
    protected readonly roundedProgress = computed(() => Math.round(this.store.trainingStatus().progress));
    protected readonly trainingLog = computed(() => (this.store.trainingStatus().log || []).join('\n'));
    protected readonly metricPrimaryLine = computed(() => ({
        dataKey: this.metricKey() as TrainingMetricKey,
        color: this.chartColors.metric,
        name: `Train ${this.metricLabel()}`,
    }));
    protected readonly metricSecondaryLine = computed(() => ({
        dataKey: this.valMetricKey() as TrainingMetricKey,
        color: this.chartColors.valMetric,
        name: `Val ${this.metricLabel()}`,
    }));

    constructor() {
        effect(() => {
            this.store.setActiveView(this.activeView().id);
        });
        void this.initializeTrainingAvailability();
    }

    private async initializeTrainingAvailability(): Promise<void> {
        const status = await getTrainingStatus();
        if (status.error) {
            this.trainingAvailability.set('unavailable');
            this.store.trainingStatusError.set(status.error);
            return;
        }
        this.trainingAvailability.set('available');
        this.handleTrainingStatus(status);
        await Promise.all([this.store.loadCheckpoints(), this.store.loadProcessedDatasets()]);
        if (status.is_training) {
            this.startTrainingStatusPolling(status.poll_interval);
        }
    }

    protected async refreshStatus(): Promise<void> {
        await this.statusPolling.checkStatus(
            (status) => this.handleTrainingStatus(status),
            (error) => this.handleTrainingStatusError(error),
            () => this.handleTrainingEnded()
        );
    }

    protected async refreshWorkspace(): Promise<void> {
        await this.store.refreshWorkspace();
    }

    protected async refreshDatasets(): Promise<void> {
        await this.store.loadProcessedDatasets();
    }

    protected async refreshCheckpoints(): Promise<void> {
        await this.store.loadCheckpoints();
    }

    protected openNewTrainingWizard(datasetLabel: string): void {
        this.store.showNewTrainingWizardFor(datasetLabel);
    }

    protected openResumeTrainingWizard(checkpointName: string): void {
        this.store.showResumeTrainingWizardFor(checkpointName);
    }

    protected async confirmTraining(config: TrainingConfig): Promise<void> {
        this.store.setConfig(config);
        await this.actionRunner.runTrainingAction({
            action: () => this.store.startTraining(),
            successStatus: 'started',
            actionLabel: 'start training',
            setLoading: (loading) => this.store.setActionLoading(loading),
            appendLog: (message) => this.store.appendTrainingLog(message),
            onSuccess: async (result) => {
                this.store.closeNewTrainingWizard();
                this.startTrainingStatusPolling(result.poll_interval);
                await this.refreshStatus();
                await this.viewNavigation.navigateTo('dashboard');
            },
        });
    }

    protected async confirmResume(config: ResumeTrainingConfig): Promise<void> {
        this.store.setResumeConfig(config);
        await this.actionRunner.runTrainingAction({
            action: () => this.store.resumeTraining(),
            successStatus: 'started',
            actionLabel: 'resume training',
            setLoading: (loading) => this.store.setActionLoading(loading),
            appendLog: (message) => this.store.appendTrainingLog(message),
            onSuccess: async (result) => {
                this.store.closeResumeTrainingWizard();
                this.startTrainingStatusPolling(result.poll_interval);
                await this.refreshStatus();
                await this.viewNavigation.navigateTo('dashboard');
            },
        });
    }

    protected async stopTraining(): Promise<void> {
        await this.actionRunner.runTrainingAction({
            action: () => this.store.stopTraining(),
            successStatus: 'stopped',
            actionLabel: 'stop training',
            setLoading: (loading) => this.store.setActionLoading(loading),
            appendLog: (message) => this.store.appendTrainingLog(message),
            onSuccess: async () => {
                await this.refreshStatus();
            },
        });
    }

    protected async deleteDataset(label: string): Promise<void> {
        if (!window.confirm(`Are you sure you want to delete dataset '${label}'?`)) {
            return;
        }
        const result = await this.store.deleteProcessedDataset(label);
        if (result.success) {
            await this.store.loadProcessedDatasets();
        } else {
            this.store.openErrorModal('Delete Dataset Failed', result.message || 'Failed to delete dataset.');
        }
    }

    protected async viewDatasetMetadata(label: string): Promise<void> {
        this.store.setActionLoading(true);
        const info = await this.store.fetchDatasetMetadata(label);
        this.store.setActionLoading(false);
        if (info && info.available) {
            this.store.openInfoModal('Dataset Metadata', buildDatasetMetadataModalData(info));
        } else {
            this.store.openErrorModal('Dataset Metadata', `Could not fetch details for dataset '${label}'.`);
        }
    }

    protected async deleteCheckpoint(name: string): Promise<void> {
        if (!window.confirm(`Are you sure you want to delete checkpoint '${name}'?`)) {
            return;
        }
        const result = await this.store.deleteCheckpoint(name);
        if (result.success) {
            await this.store.loadCheckpoints();
        } else {
            this.store.openErrorModal('Delete Checkpoint Failed', result.error || 'Failed to delete checkpoint.');
        }
    }

    protected async viewCheckpointDetails(name: string): Promise<void> {
        const { details, error } = await this.store.fetchCheckpointDetails(name);
        if (error) {
            this.store.openErrorModal('Checkpoint Details', error);
        } else if (details) {
            this.store.openInfoModal('Checkpoint Details', buildCheckpointDetailsModalData(details));
        }
    }

    private formatLossValue(value: number | undefined): string {
        const safeValue = typeof value === 'number' ? value : 0;
        return safeValue.toFixed(4);
    }

    private startTrainingStatusPolling(intervalSeconds: number | undefined): void {
        this.statusPolling.startPolling(
            intervalSeconds,
            (status) => this.handleTrainingStatus(status),
            (error) => this.handleTrainingStatusError(error),
            () => this.handleTrainingEnded()
        );
    }

    private handleTrainingStatus(status: TrainingStatus): void {
        this.store.setTrainingStatus(status);
    }

    private handleTrainingStatusError(error: string | null): void {
        this.store.trainingStatusError.set(error);
    }

    private handleTrainingEnded(): void {
        void this.store.loadCheckpoints();
    }

    private formatMetricValue(value: number | undefined, asPercent: boolean): string {
        const safeValue = typeof value === 'number' ? value : 0;
        if (asPercent) {
            return `${(safeValue * 100).toFixed(2)}%`;
        }
        return safeValue.toFixed(4);
    }

}
