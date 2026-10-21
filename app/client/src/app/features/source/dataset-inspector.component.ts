// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { CommonModule } from '@angular/common';
import {
    Component,
    EventEmitter,
    Input,
    OnChanges,
    Output,
    SimpleChanges,
    signal,
} from '@angular/core';
import type {
    ColumnRole,
    DatasetImportInspection,
    DatasetSummary,
    ExperimentSummary,
    ImportMapping,
    ObservationPage,
} from '../../models/dataset.model';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';

interface ColumnMappingRow {
    name: string;
    role: ColumnRole;
    unit: string;
    grouped: boolean;
}

@Component({
    selector: 'adsmod-dataset-inspector',
    standalone: true,
    imports: [CommonModule, DataPaginationComponent],
    template: `
        <section
            class="console-card dataset-inspector"
            aria-labelledby="dataset-inspector-title"
        >
            <div class="card-title-row">
                <div>
                    <h2 id="dataset-inspector-title">{{ dataset.name }}</h2>
                    <p class="dataset-inspector-description" [title]="dataset.description || 'No description added yet.'">
                        {{ dataset.description || 'No description added yet.' }}
                    </p>
                    <p class="dataset-inspector-tags">{{ dataset.tags.join(', ') || 'No tags' }}</p>
                </div>
            </div>

            @if (importLoading) {
                <p class="dataset-inspector-status" role="status">
                    Loading saved import details…
                </p>
            }
            @if (importError) {
                <p class="dataset-status error" role="alert">{{ importError }}</p>
            }
            @if (inspection; as imported) {
                <div class="dataset-inspector-summary">
                    <div>
                        <span>Source file</span>
                        <strong>{{ imported.original_filename }}</strong>
                    </div>
                    <div>
                        <span>Structure</span>
                        <strong>{{ imported.source_structure }}</strong>
                    </div>
                    <div>
                        <span>Validation</span>
                        <strong>{{ imported.validation.status }}</strong>
                    </div>
                    <div>
                        <span>Parser</span>
                        <strong>{{ imported.parser_version }}</strong>
                    </div>
                </div>

                <section class="dataset-inspector-section">
                    <div class="dataset-inspector-heading">
                        <div>
                            <h3>Detected columns</h3>
                            <p>
                                The saved mapping is the column interpretation used for
                                validation and persistence.
                            </p>
                        </div>
                        <span class="dataset-inspector-count">
                            {{ columnMappings(imported.mapping).length }} columns
                        </span>
                    </div>
                    <div class="dataset-inspector-table-wrap">
                        <table class="dataset-inspector-table">
                            <caption class="sr-only">
                                Detected columns for {{ dataset.name }}
                            </caption>
                            <thead>
                                <tr>
                                    <th scope="col">Source column</th>
                                    <th scope="col">Detected meaning</th>
                                    <th scope="col">Unit</th>
                                    <th scope="col">Grouping</th>
                                </tr>
                            </thead>
                            <tbody>
                                @for (
                                    column of columnMappings(imported.mapping);
                                    track column.name
                                ) {
                                    <tr>
                                        <th scope="row">{{ column.name }}</th>
                                        <td>{{ roleLabel(column.role) }}</td>
                                        <td>{{ column.unit || '—' }}</td>
                                        <td>{{ column.grouped ? 'Yes' : 'No' }}</td>
                                    </tr>
                                }
                            </tbody>
                        </table>
                    </div>
                </section>

                @if (imported.warnings.length) {
                    <div class="dataset-inspector-warnings">
                        <strong>Import warnings</strong>
                        @for (warning of imported.warnings; track warning.code + warning.source_row) {
                            <p>{{ warning.message }}</p>
                        }
                    </div>
                }
            }

            <section class="dataset-inspector-section">
                <div class="dataset-inspector-heading">
                    <div>
                        <h3>Experiments</h3>
                        <p>Select an experiment to inspect its persisted observation rows.</p>
                    </div>
                    <span class="dataset-inspector-count">{{ experiments.length }}</span>
                </div>
                @if (experimentsLoading) {
                    <p class="dataset-inspector-status" role="status">
                        Loading experiments…
                    </p>
                } @else if (!experiments.length) {
                    <p class="dataset-inspector-empty">No experiments were persisted.</p>
                } @else {
                    <div class="data-table-frame data-table-frame--compact dataset-experiments-frame">
                        <table class="data-table dataset-experiments-table">
                            <caption class="sr-only">Dataset experiments</caption>
                            <thead>
                                <tr>
                                    <th class="col-experiment" scope="col">Experiment</th>
                                    <th class="col-material" scope="col">Material</th>
                                    <th class="col-adsorbate" scope="col">Adsorbate</th>
                                    <th class="col-temperature num" scope="col">Temperature</th>
                                    <th class="col-points num" scope="col">Observations</th>
                                    <th class="col-actions" scope="col"></th>
                                </tr>
                            </thead>
                            <tbody>
                                @for (experiment of visibleExperiments; track experiment.id) {
                                    <tr [class.selected]="experiment.id === selectedExperimentId">
                                        <td class="col-experiment">
                                            <strong class="experiment-name" [title]="experiment.name">{{ experiment.name }}</strong>
                                            <span class="experiment-key" [title]="experiment.external_key">{{ experiment.external_key }}</span>
                                        </td>
                                        <td class="col-material truncate-cell" [title]="experiment.adsorbent">{{ experiment.adsorbent }}</td>
                                        <td class="col-adsorbate truncate-cell" [title]="adsorbateLabel(experiment)">
                                            {{ adsorbateLabel(experiment) }}
                                        </td>
                                        <td class="col-temperature num">{{ formatTemperature(experiment.temperature_k) }}</td>
                                        <td class="col-points num">{{ experiment.observation_count }}</td>
                                        <td class="col-actions">
                                            <div class="table-actions">
                                                <button
                                                    class="table-action"
                                                    type="button"
                                                    [attr.aria-pressed]="experiment.id === selectedExperimentId"
                                                    (click)="experimentSelected.emit(experiment.id)"
                                                >
                                                    Inspect
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                }
                            </tbody>
                        </table>
                    </div>
                    <adsmod-data-pagination
                        class="dataset-inspector-pagination"
                        [page]="experimentPage()"
                        [pageSize]="experimentPageSize()"
                        [total]="experiments.length"
                        itemLabel="experiment"
                        (pageChange)="setExperimentPage($event)"
                        (pageSizeChange)="setExperimentPageSize($event)"
                    />
                }
            </section>

            @if (selectedExperimentId !== null) {
                <section class="dataset-inspector-section" aria-labelledby="observation-title">
                    <div class="dataset-inspector-heading">
                        <div>
                            <h3 id="observation-title">Persisted observations</h3>
                            <p>Source values and canonical values retained by the dataset.</p>
                        </div>
                        @if (observations; as page) {
                            <span class="dataset-inspector-count">{{ page.total }} rows</span>
                        }
                    </div>
                    @if (observationsLoading) {
                        <p class="dataset-inspector-status" role="status">
                            Loading observations…
                        </p>
                    }
                    @if (observationsError) {
                        <p class="dataset-status error" role="alert">{{ observationsError }}</p>
                    }
                    @if (observations; as page) {
                        @if (page.rows.length) {
                            <div class="data-table-frame data-table-frame--compact dataset-observations-frame">
                                <table class="data-table dataset-observations-table">
                                    <caption class="sr-only">Persisted observations</caption>
                                    <thead>
                                        <tr>
                                            <th class="col-row num" scope="col">Source row</th>
                                            <th class="col-value" scope="col">Pressure</th>
                                            <th class="col-value" scope="col">Canonical pressure</th>
                                            <th class="col-value" scope="col">Uptake</th>
                                            <th class="col-value" scope="col">Canonical uptake</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        @for (row of page.rows; track row['id']) {
                                            <tr>
                                                <td class="col-row num">{{ displayValue(row['source_row']) }}</td>
                                                <td class="col-value">{{ displayValue(row['pressure_original']) }} {{ displayValue(row['pressure_original_unit']) }}</td>
                                                <td class="col-value">{{ displayValue(row['pressure_canonical']) }} {{ displayValue(row['pressure_canonical_unit']) }}</td>
                                                <td class="col-value">{{ displayValue(row['uptake_original']) }} {{ displayValue(row['uptake_original_unit']) }}</td>
                                                <td class="col-value">{{ displayValue(row['uptake_mol_kg']) }} mol/kg</td>
                                            </tr>
                                        }
                                    </tbody>
                                </table>
                            </div>
                        } @else if (!observationsLoading) {
                            <p class="dataset-inspector-empty">No observation rows were persisted for this experiment.</p>
                        }
                        @if (page.total) {
                            <adsmod-data-pagination
                                class="dataset-inspector-pagination"
                                [page]="observationsPage"
                                [pageSize]="observationsPageSize"
                                [total]="page.total"
                                itemLabel="observation"
                                [disabled]="observationsLoading"
                                (pageChange)="observationsPageChange.emit($event)"
                                (pageSizeChange)="observationsPageSizeChange.emit($event)"
                            />
                        }
                    }
                </section>
            }
        </section>
    `,
    styles: [
        `
            .dataset-inspector {
                display: grid;
                gap: 1.15rem;
            }
            .dataset-inspector .card-title-row,
            .dataset-inspector-heading {
                align-items: flex-start;
            }
            .dataset-inspector .card-title-row h2,
            .dataset-inspector .card-title-row p,
            .dataset-inspector-heading h3,
            .dataset-inspector-heading p {
                margin: 0;
            }
            .dataset-inspector .card-title-row p:not(.eyebrow),
            .dataset-inspector-heading p {
                margin-top: 0.25rem;
                color: var(--console-text-muted, #657386);
                font-size: 0.78rem;
            }
            .dataset-inspector-description,
            .dataset-inspector-tags {
                max-width: 100%;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-inspector-tags {
                margin-top: 0.15rem !important;
                font-size: 0.72rem !important;
            }
            .dataset-inspector-summary {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.7rem;
            }
            .dataset-inspector-summary > div {
                display: grid;
                gap: 0.2rem;
                min-width: 0;
                padding: 0.75rem;
                border: 1px solid #e1e7ef;
                border-radius: 10px;
                background: #f8fafc;
            }
            .dataset-inspector-summary span,
            .dataset-inspector-heading p {
                color: var(--console-text-muted, #657386);
                font-size: 0.72rem;
            }
            .dataset-inspector-summary strong {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .dataset-inspector-section {
                display: grid;
                gap: 0.65rem;
            }
            .dataset-inspector-heading {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
            }
            .dataset-inspector-heading h3 {
                color: var(--console-text, #1c2a45);
                font-size: 0.98rem;
            }
            .dataset-inspector-count {
                flex: 0 0 auto;
                color: var(--console-text-muted, #657386);
                font-size: 0.76rem;
                white-space: nowrap;
            }
            .dataset-inspector-table-wrap {
                max-width: 100%;
                overflow: auto;
                border: 1px solid #dfe5ed;
                border-radius: 10px;
            }
            .dataset-inspector-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.76rem;
            }
            .dataset-inspector-table th,
            .dataset-inspector-table td {
                padding: 0.6rem 0.7rem;
                border-bottom: 1px solid #e7ebf1;
                text-align: left;
                white-space: nowrap;
            }
            .dataset-inspector-table thead th {
                background: #f3f6fa;
            }
            .dataset-inspector-table tbody tr:last-child th,
            .dataset-inspector-table tbody tr:last-child td {
                border-bottom: 0;
            }
            .dataset-experiments-frame {
                --data-table-min-width: 820px;
                max-height: min(32vh, 20rem);
            }
            .dataset-observations-frame {
                --data-table-min-width: 680px;
                max-height: min(34vh, 22rem);
            }
            .dataset-experiments-table .col-experiment {
                width: auto;
            }
            .dataset-experiments-table .col-material,
            .dataset-experiments-table .col-adsorbate {
                width: 22%;
            }
            .dataset-experiments-table .col-temperature {
                width: 110px;
            }
            .dataset-experiments-table .col-points {
                width: 112px;
            }
            .dataset-experiments-table .col-actions {
                width: 88px;
            }
            .dataset-observations-table .col-row {
                width: 92px;
            }
            .experiment-name,
            .experiment-key {
                display: block;
                max-width: 100%;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .experiment-name {
                color: var(--console-text, #1c2a45);
                font-size: 0.8rem;
            }
            .experiment-key {
                color: var(--console-text-muted, #657386);
                font-size: 0.7rem;
            }
            .dataset-inspector-pagination {
                display: block;
            }
            .dataset-inspector-status,
            .dataset-inspector-empty {
                margin: 0;
                color: var(--console-text-muted, #657386);
                font-size: 0.8rem;
            }
            .dataset-inspector-warnings {
                padding: 0.7rem 0.85rem;
                border-left: 3px solid #ca8a04;
                background: #fffbeb;
                font-size: 0.78rem;
            }
            .dataset-inspector-warnings p {
                margin: 0.25rem 0 0;
            }
            @media (max-width: 800px) {
                .dataset-inspector-summary {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        `,
    ],
})
export class DatasetInspectorComponent implements OnChanges {
    @Input({ required: true }) dataset!: DatasetSummary;
    @Input() inspection: DatasetImportInspection | null = null;
    @Input() importLoading = false;
    @Input() importError = '';
    @Input() experiments: ExperimentSummary[] = [];
    @Input() experimentsLoading = false;
    @Input() selectedExperimentId: number | null = null;
    @Input() observations: ObservationPage | null = null;
    @Input() observationsLoading = false;
    @Input() observationsError = '';
    @Input() observationsPage = 1;
    @Input() observationsPageSize = 10;
    @Output() readonly experimentSelected = new EventEmitter<number>();
    @Output() readonly observationsPageChange = new EventEmitter<number>();
    @Output() readonly observationsPageSizeChange = new EventEmitter<number>();
    readonly experimentPage = signal(1);
    readonly experimentPageSize = signal(10);
    private datasetId: number | null = null;

    protected get visibleExperiments(): ExperimentSummary[] {
        const start = (this.experimentPage() - 1) * this.experimentPageSize();
        return this.experiments.slice(start, start + this.experimentPageSize());
    }

    ngOnChanges(changes: SimpleChanges): void {
        if (this.dataset && this.dataset.id !== this.datasetId) {
            this.datasetId = this.dataset.id;
            this.experimentPage.set(1);
        }
        if (changes['experiments']) {
            this.clampExperimentPage();
        }
    }

    protected setExperimentPage(page: number): void {
        const max = Math.max(
            1,
            Math.ceil(this.experiments.length / this.experimentPageSize()),
        );
        this.experimentPage.set(Math.min(Math.max(1, page), max));
    }

    protected setExperimentPageSize(size: number): void {
        if (size === this.experimentPageSize()) return;
        this.experimentPageSize.set(size);
        this.experimentPage.set(1);
    }

    protected adsorbateLabel(experiment: ExperimentSummary): string {
        return experiment.adsorbates.join(', ') || '—';
    }

    protected formatTemperature(value: number): string {
        return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)} K`;
    }

    protected columnMappings(mapping: ImportMapping): ColumnMappingRow[] {
        return Object.entries(mapping.column_roles).map(([name, role]) => ({
            name,
            role,
            unit: mapping.unit_overrides[role] || '',
            grouped: mapping.grouping_columns.includes(name),
        }));
    }

    protected roleLabel(role: ColumnRole): string {
        return role
            .split('_')
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(' ');
    }

    protected displayValue(value: unknown): string {
        if (value === null || value === undefined || value === '') {
            return '—';
        }
        return String(value);
    }

    private clampExperimentPage(): void {
        const max = Math.max(
            1,
            Math.ceil(this.experiments.length / this.experimentPageSize()),
        );
        if (this.experimentPage() > max) this.experimentPage.set(max);
    }
}
