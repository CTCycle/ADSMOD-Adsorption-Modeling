// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, inject, signal, viewChild } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { DatasetSummary } from '../../models/dataset.model';
import { fetchDatasetConfiguration } from '../../services/dataset.service';
import { fetchIdentityReviews } from '../../services/materials.service';
import { DatasetImportWizardComponent } from '../source/dataset-import-wizard.component';
import { DatasetDeleteDialogComponent } from './dataset-delete-dialog.component';
import { DatasetListComponent } from './dataset-list.component';
import { EntityDetailDrawerComponent } from './entity-detail-drawer.component';
import { EntityPanelComponent, type EntityKind, type EntityRow } from './entity-panel.component';

const DEFAULT_FILE_ACCEPT = '.csv,.xls,.xlsx';
const HIGHLIGHT_MS = 4000;

@Component({
    selector: 'adsmod-datasets-page',
    standalone: true,
    imports: [
        DatasetDeleteDialogComponent,
        DatasetImportWizardComponent,
        DatasetListComponent,
        EntityDetailDrawerComponent,
        EntityPanelComponent,
        RouterLink,
    ],
    template: `
        <div class="data-page datasets-page">
            <input #sourceFileInput class="source-file-input" type="file" aria-label="Choose a dataset file" [accept]="acceptedFileTypes()" (change)="fileChanged($event)" />
            @if (store.managementStatus()) {
                <div class="dataset-status error" role="alert">
                    <span>{{ store.managementStatus() }}</span>
                    <button class="button secondary" type="button" (click)="store.refreshDatasets()">Retry</button>
                </div>
            }
            @if (notice()) {
                <div class="dataset-status success" role="status">{{ notice() }}</div>
            }
            @if (openReviews()) {
                <div class="identity-strip" role="status">
                    <span>{{ openReviews() }} possible duplicate {{ openReviews() === 1 ? 'identity needs' : 'identities need' }} review.</span>
                    <a routerLink="/datasets/identity-review">Review</a>
                </div>
            }
            <div class="datasets-layout">
                <adsmod-dataset-list
                    class="console-card datasets-primary"
                    [datasets]="store.datasets()"
                    [state]="store.datasetListState()"
                    [highlightedId]="highlightedId()"
                    (stateChanged)="store.datasetListState.set($event)"
                    (addRequested)="sourceFileInput.click()"
                    (deleteRequested)="pendingDelete.set($event)"
                />
                <adsmod-entity-panel
                    class="console-card datasets-secondary"
                    [refreshKey]="entityRefresh()"
                    [selectedId]="selectedEntity()?.id ?? null"
                    [initialKind]="initialKind"
                    [initialSource]="initialSource"
                    (selected)="selectedEntity.set($event)"
                    (kindChanged)="selectedEntity.set(null)"
                />
            </div>
            @if (pendingFile(); as file) {
                <adsmod-dataset-import-wizard [file]="file" (closed)="wizardClosed()" (saved)="wizardSaved($event)" (explore)="explore($event)" />
            }
            @if (selectedEntity(); as entity) {
                <adsmod-entity-detail-drawer
                    [kind]="entity.kind"
                    [entityId]="entity.id"
                    (closed)="selectedEntity.set(null)"
                    (deleted)="entityDeleted(entity)"
                />
            }
            <adsmod-dataset-delete-dialog [dataset]="pendingDelete()" (closed)="pendingDelete.set(null)" (deleted)="datasetDeleted()" />
        </div>
    `,
    styles: [
        `
            .identity-strip { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm); padding: 0.55rem var(--spacing-md); border-left: 4px solid #ca8a04; background: var(--warning-bg); font-size: 0.875rem; }
            .identity-strip a { color: var(--primary-600); font-weight: 600; }
            .datasets-layout {
                display: grid;
                grid-template-columns: minmax(0, 2fr) minmax(20rem, 1fr);
                gap: var(--spacing-lg);
                align-items: stretch;
                min-height: calc(100vh - var(--console-header-height, 5rem) - 7rem);
            }
            .datasets-primary,
            .datasets-secondary {
                display: flex;
                flex-direction: column;
                padding: var(--spacing-md) var(--spacing-lg);
            }
            @media (max-width: 1200px) {
                .datasets-layout { grid-template-columns: minmax(0, 1fr); min-height: 0; }
            }
            @media (max-width: 640px) {
                .datasets-primary,
                .datasets-secondary { padding: var(--spacing-sm) var(--spacing-md); }
            }
        `,
    ],
})
export class DatasetsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);
    private readonly router = inject(Router);
    private readonly query = inject(ActivatedRoute).snapshot.queryParamMap;
    /** Deep-link filters, e.g. from the Sources catalog tiles. */
    protected readonly initialKind: EntityKind = this.query.get('kind') === 'species' ? 'species' : 'material';
    protected readonly initialSource: string | null = this.query.get('source');
    private readonly list = viewChild.required(DatasetListComponent);
    protected readonly pendingFile = signal<File | null>(null);
    protected readonly acceptedFileTypes = signal(DEFAULT_FILE_ACCEPT);
    protected readonly highlightedId = signal<number | null>(null);
    protected readonly pendingDelete = signal<DatasetSummary | null>(null);
    protected readonly selectedEntity = signal<EntityRow | null>(null);
    protected readonly entityRefresh = signal(0);
    protected readonly notice = signal('');
    protected readonly openReviews = signal(0);
    private noticeTimer: ReturnType<typeof setTimeout> | null = null;

    constructor() {
        const origin = this.query.get('origin');
        if (origin === 'nist' || origin === 'uploaded') {
            this.store.datasetListState.update((state) => ({ ...state, source: origin, page: 1 }));
        }
        void this.store.refreshDatasets();
        void this.loadDatasetConfiguration();
        void this.loadOpenReviews();
    }

    private async loadOpenReviews(): Promise<void> {
        const result = await fetchIdentityReviews({ page: 1, page_size: 1, status: 'open' });
        this.openReviews.set(result.data?.pagination.total ?? 0);
    }

    private async loadDatasetConfiguration(): Promise<void> {
        const result = await fetchDatasetConfiguration();
        if (result.data?.allowed_extensions?.length) {
            this.acceptedFileTypes.set(result.data.allowed_extensions.join(','));
        }
    }

    protected fileChanged(event: Event): void {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        if (file) {
            this.pendingFile.set(file);
        }
        // Allow choosing the same file again after the wizard closes.
        input.value = '';
    }

    protected async wizardSaved(datasetId: number): Promise<void> {
        await this.store.refreshDatasets();
        this.entityRefresh.update((value) => value + 1);
        void this.loadOpenReviews();
        this.list().revealDataset(datasetId);
        this.highlightedId.set(datasetId);
        setTimeout(() => {
            if (this.highlightedId() === datasetId) this.highlightedId.set(null);
        }, HIGHLIGHT_MS);
    }

    protected wizardClosed(): void {
        this.pendingFile.set(null);
        queueMicrotask(() => this.list().focusAddButton());
    }

    protected explore(datasetId: number): void {
        void this.router.navigate(['/datasets', datasetId]);
    }

    protected datasetDeleted(): void {
        this.entityRefresh.update((value) => value + 1);
        this.showNotice('Dataset deleted.');
    }

    protected entityDeleted(entity: EntityRow): void {
        this.selectedEntity.set(null);
        this.entityRefresh.update((value) => value + 1);
        this.showNotice(`${entity.name} deleted.`);
    }

    private showNotice(message: string): void {
        this.notice.set(message);
        if (this.noticeTimer) clearTimeout(this.noticeTimer);
        this.noticeTimer = setTimeout(() => this.notice.set(''), HIGHLIGHT_MS);
    }
}
