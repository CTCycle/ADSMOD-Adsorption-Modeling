import { Component, input, output, signal } from '@angular/core';
import type { CheckpointInfo, ProcessedDatasetInfo } from '../../../models/training.model';

type TrainingSetupViewMode = 'all' | 'datasets' | 'checkpoints';

@Component({
    selector: 'adsmod-training-setup-row',
    standalone: true,
    template: `
        <div class="training-setup-container">
            @if (showDatasets()) {
                <div class="section-container">
                    @if (showSectionHeading()) {
                        <h3 class="split-selection-title">Training Datasets</h3>
                    }

                    <div class="split-selection-card">
                        <div class="split-selection-card-left">
                            <div class="split-selection-card-toolbar">
                                @if (onRefreshDatasets()) {
                                    <button type="button" class="split-selection-refresh-button" (click)="onRefreshDatasets()!()">Refresh</button>
                                }
                            </div>

                            <div class="split-selection-card-content">
                                <table class="split-table">
                                    <thead class="split-table-head">
                                        <tr class="split-table-header-row">
                                            <th class="split-table-header-cell col-name">Name</th>
                                            <th class="split-table-header-cell col-train">Train</th>
                                            <th class="split-table-header-cell col-val">Val</th>
                                            <th class="split-table-header-cell col-actions">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        @if (processedDatasets().length === 0) {
                                            <tr><td colspan="4" class="split-table-empty-cell">No datasets available.</td></tr>
                                        } @else {
                                            @for (dataset of processedDatasets(); track dataset.dataset_label) {
                                                <tr
                                                    class="split-table-row"
                                                    [class.selected]="selectedDatasetLabel() === dataset.dataset_label"
                                                    (click)="toggleDataset(dataset.dataset_label)"
                                                    role="button"
                                                    tabindex="0"
                                                    (keydown.enter)="toggleDataset(dataset.dataset_label)"
                                                    (keydown.space)="toggleDatasetWithPreventDefault($event, dataset.dataset_label)"
                                                >
                                                    <td class="split-table-cell split-table-cell-strong split-table-cell-ellipsis">{{ dataset.dataset_label }}</td>
                                                    <td class="split-table-cell">{{ dataset.train_samples }}</td>
                                                    <td class="split-table-cell">{{ dataset.validation_samples }}</td>
                                                    <td class="split-table-cell split-table-cell-right">
                                                        <div class="split-table-actions-wrap">
                                                            <button class="icon-action-button" type="button" title="View Metadata" (click)="viewDataset($event, dataset.dataset_label)">
                                                                <span aria-hidden="true">i</span>
                                                            </button>
                                                            <button class="icon-action-button" type="button" title="Delete Dataset" (click)="deleteDataset($event, dataset.dataset_label)">
                                                                <span aria-hidden="true">x</span>
                                                            </button>
                                                        </div>
                                                    </td>
                                                </tr>
                                            }
                                        }
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div class="split-selection-card-right">
                            <div class="split-selection-card-header-row">
                                <div class="split-selection-card-icon-wrap">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                                        <circle cx="12" cy="12" r="3" />
                                        <circle cx="6" cy="6" r="2" />
                                        <circle cx="6" cy="18" r="2" />
                                        <circle cx="18" cy="6" r="2" />
                                        <circle cx="18" cy="18" r="2" />
                                        <path d="M8 7.5L10.5 10" />
                                        <path d="M8 16.5L10.5 14" />
                                        <path d="M13.5 10L16 7.5" />
                                        <path d="M13.5 14L16 16.5" />
                                    </svg>
                                </div>
                                <h4 class="split-selection-card-title">Start Training Run</h4>
                            </div>

                            <p class="split-selection-card-description split-selection-card-description-wide">Select a processed adsorption dataset and open the training setup.</p>

                            @if (selectedDatasetLabel()) {
                                <div class="split-selection-card-selection">
                                    <span class="split-selection-card-selection-label">Selected</span>
                                    <div class="split-selection-card-selection-value">{{ selectedDatasetLabel() }}</div>
                                </div>
                            } @else {
                                <div class="split-selection-card-selection-placeholder"></div>
                            }

                            <div class="split-selection-card-actions split-selection-card-actions-single">
                                <button class="primary split-selection-card-action-button" type="button" [disabled]="!selectedDatasetLabel() || isTraining()" (click)="newTrainingRequested.emit(selectedDatasetLabel()!)">
                                    Open Training Setup
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            }

            @if (showCheckpoints()) {
                <div class="section-container">
                    @if (showSectionHeading()) {
                        <h3 class="split-selection-title">Checkpoints</h3>
                    }

                    <div class="split-selection-card">
                        <div class="split-selection-card-left">
                            <div class="split-selection-card-toolbar">
                                @if (onRefreshCheckpoints()) {
                                    <button type="button" class="split-selection-refresh-button" (click)="onRefreshCheckpoints()!()">Refresh</button>
                                }
                            </div>

                            <div class="split-selection-card-content">
                                <table class="split-table">
                                    <thead class="split-table-head">
                                        <tr class="split-table-header-row">
                                            <th class="split-table-header-cell col-name">Name</th>
                                            <th class="split-table-header-cell col-epochs">Epochs</th>
                                            <th class="split-table-header-cell col-loss">Loss</th>
                                            <th class="split-table-header-cell col-actions">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        @if (checkpoints().length === 0) {
                                            <tr><td colspan="4" class="split-table-empty-cell">No checkpoints available.</td></tr>
                                        } @else {
                                            @for (checkpoint of checkpoints(); track checkpoint.name) {
                                                <tr
                                                    class="split-table-row"
                                                    [class.selected]="selectedCheckpointName() === checkpoint.name"
                                                    (click)="toggleCheckpoint(checkpoint.name)"
                                                    role="button"
                                                    tabindex="0"
                                                    (keydown.enter)="toggleCheckpoint(checkpoint.name)"
                                                    (keydown.space)="toggleCheckpointWithPreventDefault($event, checkpoint.name)"
                                                >
                                                    <td class="split-table-cell split-table-cell-strong split-table-cell-ellipsis">{{ checkpoint.name }}</td>
                                                    <td class="split-table-cell">{{ checkpoint.epochs_trained ?? '-' }}</td>
                                                    <td class="split-table-cell">{{ checkpoint.final_loss?.toFixed(4) ?? '-' }}</td>
                                                    <td class="split-table-cell split-table-cell-right">
                                                        <div class="split-table-actions-wrap">
                                                            <button class="icon-action-button" type="button" title="View Details" (click)="viewCheckpoint($event, checkpoint.name)">
                                                                <span aria-hidden="true">i</span>
                                                            </button>
                                                            <button class="icon-action-button" type="button" title="Delete Checkpoint" (click)="deleteCheckpointEntry($event, checkpoint.name)">
                                                                <span aria-hidden="true">x</span>
                                                            </button>
                                                        </div>
                                                    </td>
                                                </tr>
                                            }
                                        }
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div class="split-selection-card-right">
                            <div class="split-selection-card-header-row">
                                <div class="split-selection-card-icon-wrap">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                                        <rect x="4" y="4" width="16" height="16" rx="2" />
                                        <path d="M8 9h8" />
                                        <path d="M8 13h8" />
                                        <path d="M8 17h5" />
                                    </svg>
                                </div>
                                <h4 class="split-selection-card-title">Resume Training Run</h4>
                            </div>

                            <p class="split-selection-card-description">Continue model training from a saved checkpoint.</p>

                            @if (selectedCheckpointName()) {
                                <div class="split-selection-card-selection">
                                    <span class="split-selection-card-selection-label">Selected</span>
                                    <div class="split-selection-card-selection-value">{{ selectedCheckpointName() }}</div>
                                </div>
                            } @else {
                                <div class="split-selection-card-selection-placeholder"></div>
                            }

                            <div class="split-selection-card-actions split-selection-card-actions-single">
                                <button class="secondary split-selection-card-action-button" type="button" [disabled]="!selectedCheckpointName() || isTraining()" (click)="resumeTrainingRequested.emit(selectedCheckpointName()!)">
                                    Resume Training
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            }
        </div>
    `,
})
export class TrainingSetupRowComponent {
    readonly processedDatasets = input.required<ProcessedDatasetInfo[]>();
    readonly checkpoints = input.required<CheckpointInfo[]>();
    readonly isTraining = input(false);
    readonly onRefreshDatasets = input<(() => void) | undefined>(undefined);
    readonly onRefreshCheckpoints = input<(() => void) | undefined>(undefined);
    readonly viewMode = input<TrainingSetupViewMode>('all');
    readonly showSectionHeading = input(true);
    readonly newTrainingRequested = output<string>();
    readonly resumeTrainingRequested = output<string>();
    readonly datasetMetadataRequested = output<string>();
    readonly datasetDeleteRequested = output<string>();
    readonly checkpointDetailsRequested = output<string>();
    readonly checkpointDeleteRequested = output<string>();
    protected readonly selectedDatasetLabel = signal<string | null>(null);
    protected readonly selectedCheckpointName = signal<string | null>(null);

    protected showDatasets(): boolean {
        return this.viewMode() === 'all' || this.viewMode() === 'datasets';
    }

    protected showCheckpoints(): boolean {
        return this.viewMode() === 'all' || this.viewMode() === 'checkpoints';
    }

    protected toggleDataset(label: string): void {
        this.selectedDatasetLabel.update((current) => current === label ? null : label);
    }

    protected toggleCheckpoint(name: string): void {
        this.selectedCheckpointName.update((current) => current === name ? null : name);
    }

    protected toggleDatasetWithPreventDefault(event: Event, label: string): void {
        event.preventDefault();
        this.toggleDataset(label);
    }

    protected toggleCheckpointWithPreventDefault(event: Event, name: string): void {
        event.preventDefault();
        this.toggleCheckpoint(name);
    }

    protected viewDataset(event: Event, label: string): void {
        event.stopPropagation();
        this.datasetMetadataRequested.emit(label);
    }

    protected deleteDataset(event: Event, label: string): void {
        event.stopPropagation();
        this.datasetDeleteRequested.emit(label);
    }

    protected viewCheckpoint(event: Event, name: string): void {
        event.stopPropagation();
        this.checkpointDetailsRequested.emit(name);
    }

    protected deleteCheckpointEntry(event: Event, name: string): void {
        event.stopPropagation();
        this.checkpointDeleteRequested.emit(name);
    }
}
