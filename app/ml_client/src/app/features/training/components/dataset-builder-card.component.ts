import { Component, computed, input, signal } from '@angular/core';
import { SplitSelectionCardComponent } from './split-selection-card.component';
import { DatasetProcessingWizardComponent } from './dataset-processing-wizard.component';
import type { DatasetBuildConfig } from '../../../models/dataset-build.model';
import type { DatasetFullInfo, DatasetSourceInfo } from '../../../models/training.model';
import { buildTrainingDataset, clearTrainingDataset, deleteDatasetSource, getTrainingDatasetInfo } from '../../../services/dataset-builder.service';
import { fetchDatasetSources } from '../../../services/training.service';

const buildDatasetKey = (dataset: DatasetSourceInfo): string => `${dataset.source}:${dataset.dataset_name}`;

@Component({
    selector: 'adsmod-dataset-builder-card',
    standalone: true,
    imports: [SplitSelectionCardComponent, DatasetProcessingWizardComponent],
    template: `
        <div class="section-container">
            <adsmod-split-selection-card
                title="Dataset Processing"
                subtitle="Compose training-ready data from your available sources."
                [showRefresh]="true"
                [hideHeader]="!showSectionHeading()"
                (refresh)="loadDatasetSources()"
            >
                <div card-left>
                    <div class="dataset-table dataset-table-flat">
                        <div class="dataset-table-header dataset-table-header-sticky">
                            <span>Name</span>
                            <span>Source</span>
                            <span>Rows</span>
                            <span class="dataset-actions-header">Actions</span>
                        </div>
                        <div class="dataset-table-body dataset-table-body-unbounded">
                            @if (datasetSources().length === 0) {
                                <div class="dataset-table-empty dataset-table-empty-lg">No datasets available yet.</div>
                            }
                            @for (dataset of datasetSources(); track datasetKey(dataset)) {
                                <div
                                    class="dataset-row dataset-row-flat"
                                    [class.selected]="isSelected(dataset)"
                                    (click)="toggleDataset(dataset)"
                                    role="button"
                                    tabindex="0"
                                    (keydown.enter)="toggleDataset(dataset)"
                                    (keydown.space)="toggleDatasetWithPreventDefault($event, dataset)"
                                >
                                    <span class="dataset-name-cell">{{ dataset.display_name }}</span>
                                    <span class="dataset-source">{{ dataset.source }}</span>
                                    <span class="dataset-count">{{ dataset.row_count }}</span>
                                    <span class="dataset-actions-cell">
                                        <button
                                            class="icon-action-button"
                                            type="button"
                                            [disabled]="dataset.source !== 'uploaded'"
                                            [title]="dataset.source === 'uploaded' ? 'Delete Dataset' : 'NIST datasets cannot be removed'"
                                            (click)="deleteSourceDataset($event, dataset)"
                                        >
                                            🗑️
                                        </button>
                                    </span>
                                </div>
                            }
                        </div>
                    </div>
                </div>

                <div card-right>
                    <div class="split-selection-card-header-row">
                        <div class="split-selection-card-icon-wrap">
                            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                                <path d="M12 12v9" />
                                <path d="m8 17 4 4 4-4" />
                            </svg>
                        </div>
                        <h4 class="split-selection-card-title">Build Training Dataset</h4>
                    </div>

                    <p class="split-selection-card-description">
                        Merge uploaded and NIST adsorption data into a machine-learning-ready dataset.
                        @if (datasetInfo()?.available) {
                            <span class="split-selection-card-ready-note">
                                Ready: {{ datasetInfo()?.train_samples }}T / {{ datasetInfo()?.validation_samples }}V
                            </span>
                        }
                    </p>

                    <div>
                        @if (selectedCount() > 0) {
                            <div class="split-selection-card-selection">
                                <span class="split-selection-card-selection-label">Selection</span>
                                <div class="split-selection-card-selection-value">{{ selectedCount() }} datasets selected</div>
                            </div>
                        } @else {
                            <div class="split-selection-card-selection-placeholder"></div>
                        }
                    </div>

                    <div class="split-selection-card-actions">
                        <button
                            class="primary split-selection-card-action-button"
                            type="button"
                            [disabled]="selectedCount() === 0 || isBuilding()"
                            (click)="isWizardOpen.set(true)"
                        >
                            {{ isBuilding() ? 'Building...' : 'Configure Dataset Build' }}
                        </button>
                        <button
                            class="secondary split-selection-card-action-button"
                            type="button"
                            [disabled]="!datasetInfo()?.available || isBuilding()"
                            title="Clear current training dataset"
                            (click)="clearDataset()"
                        >
                            Clear Dataset
                        </button>
                    </div>
                </div>
            </adsmod-split-selection-card>

            @if (statusMessage()) {
                <div class="dataset-status" [class.info]="statusTone() === 'info'" [class.success]="statusTone() === 'success'" [class.error]="statusTone() === 'error'">
                    {{ statusMessage() }}
                    @if (isBuilding() && jobProgress() !== null) {
                        <span class="dataset-progress">{{ roundedProgress() }}%</span>
                    }
                </div>
            }

            @if (isWizardOpen()) {
                <adsmod-dataset-processing-wizard
                    [selectedDatasets]="selectedDatasets()"
                    (closed)="isWizardOpen.set(false)"
                    (buildStarted)="handleBuildStart($event)"
                />
            }
        </div>
    `,
})
export class DatasetBuilderCardComponent {
    readonly showSectionHeading = input(true);
    readonly datasetBuilt = input<(() => void) | undefined>(undefined);
    readonly workspaceChanged = input<(() => void) | undefined>(undefined);
    protected readonly datasetSources = signal<DatasetSourceInfo[]>([]);
    protected readonly selectedKeys = signal<Set<string>>(new Set());
    protected readonly isWizardOpen = signal(false);
    protected readonly isBuilding = signal(false);
    protected readonly statusMessage = signal<string | null>(null);
    protected readonly statusTone = signal<'info' | 'success' | 'error'>('info');
    protected readonly datasetInfo = signal<DatasetFullInfo | null>(null);
    protected readonly jobProgress = signal<number | null>(null);
    protected readonly selectedDatasets = computed(() => this.datasetSources().filter((dataset) => this.selectedKeys().has(buildDatasetKey(dataset))));
    protected readonly selectedCount = computed(() => this.selectedKeys().size);
    protected readonly roundedProgress = computed(() => Math.round(this.jobProgress() ?? 0));

    constructor() {
        void this.loadDatasetInfo();
        void this.loadDatasetSources();
    }

    protected datasetKey(dataset: DatasetSourceInfo): string {
        return buildDatasetKey(dataset);
    }

    protected isSelected(dataset: DatasetSourceInfo): boolean {
        return this.selectedKeys().has(buildDatasetKey(dataset));
    }

    protected toggleDataset(dataset: DatasetSourceInfo): void {
        const key = buildDatasetKey(dataset);
        this.selectedKeys.update((previous) => {
            const next = new Set(previous);
            if (next.has(key)) {
                next.delete(key);
            } else {
                next.add(key);
            }
            return next;
        });
    }

    protected toggleDatasetWithPreventDefault(event: Event, dataset: DatasetSourceInfo): void {
        event.preventDefault();
        this.toggleDataset(dataset);
    }

    protected async loadDatasetSources(): Promise<void> {
        this.selectedKeys.set(new Set());
        this.statusMessage.set(null);
        const result = await fetchDatasetSources();
        if (result.error) {
            this.datasetSources.set([]);
            this.statusTone.set('error');
            this.statusMessage.set(`ERROR: ${result.error}`);
            return;
        }

        this.datasetSources.set(result.datasets);
    }

    protected async clearDataset(): Promise<void> {
        const result = await clearTrainingDataset();
        if (result.success) {
            this.statusTone.set('info');
            this.statusMessage.set('Dataset cleared');
            await this.loadDatasetInfo();
            this.workspaceChanged()?.();
        } else {
            this.statusTone.set('error');
            this.statusMessage.set(`ERROR: ${result.message}`);
        }
    }

    protected async handleBuildStart(config: DatasetBuildConfig): Promise<void> {
        this.isBuilding.set(true);
        this.jobProgress.set(0);
        this.statusTone.set('info');
        this.statusMessage.set('Building dataset...');
        const result = await buildTrainingDataset(config, (status) => this.jobProgress.set(status.progress));
        if (result.success) {
            this.statusTone.set('success');
            this.statusMessage.set(`OK: ${result.message} (${result.train_samples} train, ${result.validation_samples} val)`);
            await this.loadDatasetInfo();
            this.datasetBuilt()?.();
            this.workspaceChanged()?.();
        } else {
            this.statusTone.set('error');
            this.statusMessage.set(`ERROR: ${result.message}`);
        }

        this.isBuilding.set(false);
        this.jobProgress.set(null);
    }

    protected async deleteSourceDataset(event: Event, dataset: DatasetSourceInfo): Promise<void> {
        event.stopPropagation();
        if (dataset.source !== 'uploaded') {
            return;
        }
        if (!window.confirm(`Are you sure you want to delete dataset '${dataset.display_name}'?`)) {
            return;
        }

        const result = await deleteDatasetSource(dataset.source, dataset.dataset_name);
        if (result.success) {
            await this.loadDatasetSources();
            this.workspaceChanged()?.();
        } else {
            this.statusTone.set('error');
            this.statusMessage.set(`ERROR: Failed to delete dataset: ${result.message}`);
        }
    }

    private async loadDatasetInfo(): Promise<void> {
        this.datasetInfo.set(await getTrainingDatasetInfo());
    }
}
