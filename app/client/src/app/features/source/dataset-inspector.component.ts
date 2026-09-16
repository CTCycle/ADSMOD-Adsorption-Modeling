import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import type {
    ColumnRole,
    DatasetImportInspection,
    DatasetSummary,
    ExperimentSummary,
    ImportMapping,
    ObservationPage,
} from '../../models/dataset.model';

interface ColumnMappingRow {
    name: string;
    role: ColumnRole;
    unit: string;
    grouped: boolean;
}

@Component({
    selector: 'adsmod-dataset-inspector',
    standalone: true,
    imports: [CommonModule],
    template: `
        <section
            class="console-card dataset-inspector"
            aria-labelledby="dataset-inspector-title"
        >
            <div class="card-title-row">
                <div>
                    <p class="eyebrow">Dataset inspection</p>
                    <h2 id="dataset-inspector-title">{{ dataset.name }}</h2>
                    <p>
                        Review the saved import mapping and canonical observations.
                    </p>
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
                    <div class="dataset-experiment-list" role="list" aria-label="Dataset experiments">
                        @for (experiment of experiments; track experiment.id) {
                            <div role="listitem">
                                <button
                                    class="dataset-experiment"
                                    [class.selected]="experiment.id === selectedExperimentId"
                                    type="button"
                                    [attr.aria-pressed]="experiment.id === selectedExperimentId"
                                    (click)="experimentSelected.emit(experiment.id)"
                                >
                                    <span>
                                        <strong>{{ experiment.name }}</strong>
                                        <small>{{ experiment.external_key }}</small>
                                    </span>
                                    <span>
                                        {{ experiment.adsorbates.join(', ') }} on
                                        {{ experiment.adsorbent }} ·
                                        {{ experiment.temperature_k }} K ·
                                        {{ experiment.observation_count }} points
                                    </span>
                                </button>
                            </div>
                        }
                    </div>
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
                        <div class="dataset-inspector-table-wrap">
                            <table class="dataset-inspector-table">
                                <caption class="sr-only">Persisted observations</caption>
                                <thead>
                                    <tr>
                                        <th scope="col">Source row</th>
                                        <th scope="col">Pressure</th>
                                        <th scope="col">Canonical pressure</th>
                                        <th scope="col">Uptake</th>
                                        <th scope="col">Canonical uptake</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @for (row of page.rows; track row['id']) {
                                        <tr>
                                            <td>{{ displayValue(row['source_row']) }}</td>
                                            <td>{{ displayValue(row['pressure_original']) }} {{ displayValue(row['pressure_original_unit']) }}</td>
                                            <td>{{ displayValue(row['pressure_canonical']) }} {{ displayValue(row['pressure_canonical_unit']) }}</td>
                                            <td>{{ displayValue(row['uptake_original']) }} {{ displayValue(row['uptake_original_unit']) }}</td>
                                            <td>{{ displayValue(row['uptake_mol_kg']) }} mol/kg</td>
                                        </tr>
                                    }
                                </tbody>
                            </table>
                        </div>
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
            .dataset-experiment small {
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
            .dataset-experiment-list {
                display: grid;
                gap: 0.55rem;
            }
            .dataset-experiment {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                width: 100%;
                padding: 0.7rem 0.8rem;
                border: 1px solid #dfe5ed;
                border-radius: 10px;
                color: var(--console-text, #1c2a45);
                background: #fff;
                text-align: left;
                cursor: pointer;
            }
            .dataset-experiment:hover,
            .dataset-experiment.selected {
                border-color: #9bbcf0;
                box-shadow: 0 8px 20px -18px rgba(7, 91, 213, 0.75);
            }
            .dataset-experiment > span:first-child {
                display: grid;
                gap: 0.15rem;
                min-width: 0;
            }
            .dataset-experiment > span:last-child {
                flex: 0 0 auto;
                color: var(--console-text-muted, #657386);
                font-size: 0.74rem;
                text-align: right;
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
                .dataset-experiment {
                    align-items: flex-start;
                    flex-direction: column;
                }
                .dataset-experiment > span:last-child {
                    text-align: left;
                }
            }
        `,
    ],
})
export class DatasetInspectorComponent {
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
    @Output() readonly experimentSelected = new EventEmitter<number>();

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
}
