import { CommonModule } from '@angular/common';
import {
    Component,
    EventEmitter,
    Input,
    OnInit,
    Output,
    signal,
} from '@angular/core';
import type {
    ColumnRole,
    DatasetImportResponse,
    ImportMapping,
    ImportPreview,
    ImportValidation,
    ImportableStructure,
    PressureBasis,
} from '../../models/dataset.model';
import {
    commitDataset,
    previewDataset,
    validateDataset,
} from '../../services/dataset.service';

const COLUMN_ROLES: readonly { value: ColumnRole; label: string }[] = [
    { value: 'ignore', label: 'Ignore' },
    { value: 'experiment_id', label: 'Experiment ID' },
    { value: 'experiment_name', label: 'Experiment name' },
    { value: 'pressure', label: 'Pressure series' },
    { value: 'uptake', label: 'Adsorbed amount / uptake' },
    { value: 'adsorbate', label: 'Adsorbate species' },
    { value: 'adsorbate_smiles', label: 'Adsorbate SMILES' },
    { value: 'adsorbent', label: 'Adsorbent material' },
    { value: 'temperature', label: 'Temperature' },
    { value: 'pressure_unit', label: 'Pressure unit' },
    { value: 'uptake_unit', label: 'Uptake unit' },
    { value: 'temperature_unit', label: 'Temperature unit' },
    { value: 'uptake_stddev', label: 'Uptake uncertainty' },
    { value: 'saturation_pressure', label: 'Saturation pressure p₀' },
    { value: 'metadata', label: 'Additional metadata' },
];

@Component({
    selector: 'adsmod-dataset-import-wizard',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="dataset-modal-backdrop">
            <section
                class="import-wizard"
                role="dialog"
                aria-modal="true"
                aria-labelledby="import-wizard-title"
            >
                <header class="import-wizard-header">
                    <div>
                        <p class="eyebrow">Dataset import</p>
                        <h2 id="import-wizard-title">
                            Understand {{ file.name }}
                        </h2>
                        <p>
                            Map scientific meaning and units before anything is
                            saved.
                        </p>
                    </div>
                    <button
                        class="button quiet"
                        type="button"
                        aria-label="Close import wizard"
                        (click)="closed.emit()"
                    >
                        ×
                    </button>
                </header>

                <ol class="import-steps" aria-label="Import progress">
                    @for (label of stepLabels; track label; let index = $index) {
                        <li [class.active]="step() === index + 1">
                            <span>{{ index + 1 }}</span>{{ label }}
                        </li>
                    }
                </ol>

                <div class="import-wizard-body">
                    @if (busy()) {
                        <div class="import-loading" role="status">
                            Checking the dataset…
                        </div>
                    }
                    @if (error()) {
                        <p class="dataset-status error" role="alert">
                            {{ error() }}
                        </p>
                    }

                    @if (preview(); as detected) {
                        @if (step() === 1) {
                            <section class="import-section">
                                <div class="import-summary-grid">
                                    <div>
                                        <span>Detected structure</span>
                                        <strong>{{
                                            detected.detected_structure
                                        }}</strong>
                                    </div>
                                    <div>
                                        <span>Rows</span>
                                        <strong>{{ detected.row_count }}</strong>
                                    </div>
                                    <div>
                                        <span>Columns</span>
                                        <strong>{{
                                            detected.column_count
                                        }}</strong>
                                    </div>
                                    <div>
                                        <span>Confidence</span>
                                        <strong>{{
                                            detected.structure_confidence
                                                | percent: '1.0-0'
                                        }}</strong>
                                    </div>
                                </div>
                                <div class="import-guidance">
                                    @for (
                                        guidance of detected.guidance;
                                        track guidance
                                    ) {
                                        <p>{{ guidance }}</p>
                                    }
                                </div>
                                <div class="import-table-wrap">
                                    <table class="import-preview-table">
                                        <thead>
                                            <tr>
                                                @for (
                                                    column of detected.columns;
                                                    track column.name
                                                ) {
                                                    <th>{{ column.name }}</th>
                                                }
                                            </tr>
                                        </thead>
                                        <tbody>
                                            @for (
                                                row of detected.preview_rows;
                                                track $index
                                            ) {
                                                <tr>
                                                    @for (
                                                        column of detected.columns;
                                                        track column.name
                                                    ) {
                                                        <td>
                                                            {{
                                                                displayValue(
                                                                    row[
                                                                        column
                                                                            .name
                                                                    ]
                                                                )
                                                            }}
                                                        </td>
                                                    }
                                                </tr>
                                            }
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        }

                        @if (step() === 2) {
                            <section class="import-section">
                                <div class="import-settings-grid">
                                    <label>
                                        <span>Dataset name</span>
                                        <input
                                            class="select-input"
                                            [value]="mapping().dataset_name"
                                            (input)="setDatasetName($event)"
                                        />
                                    </label>
                                    <label>
                                        <span>Source structure</span>
                                        <select
                                            class="select-input"
                                            [value]="mapping().structure"
                                            (change)="setStructure($event)"
                                        >
                                            <option value="atomic">
                                                Atomic — one observation per row
                                            </option>
                                            <option value="aggregated">
                                                Aggregated — series inside rows
                                            </option>
                                            <option value="mixed">
                                                Mixed, explicitly mapped
                                            </option>
                                        </select>
                                    </label>
                                    <label>
                                        <span>Pressure meaning</span>
                                        <select
                                            class="select-input"
                                            [value]="mapping().pressure_basis"
                                            (change)="setPressureBasis($event)"
                                        >
                                            <option value="absolute">
                                                Absolute
                                            </option>
                                            <option value="partial">
                                                Partial
                                            </option>
                                            <option value="relative">
                                                Relative p/p₀
                                            </option>
                                        </select>
                                    </label>
                                    <label>
                                        <span>Duplicate pressure points</span>
                                        <select
                                            class="select-input"
                                            [value]="mapping().duplicate_policy"
                                            (change)="setDuplicatePolicy($event)"
                                        >
                                            <option value="reject">
                                                Reject until reviewed
                                            </option>
                                            <option value="average">
                                                Average within experiment
                                            </option>
                                            <option value="keep">Keep all</option>
                                        </select>
                                    </label>
                                    <label>
                                        <span>Decimal separator</span>
                                        <select
                                            class="select-input"
                                            [value]="mapping().decimal_separator"
                                            (change)="setDecimalSeparator($event)"
                                        >
                                            <option value="auto">Detect automatically</option>
                                            <option value=".">Decimal point (.)</option>
                                            <option value=",">Comma decimal (,)</option>
                                        </select>
                                    </label>
                                    <label>
                                        <span>Pressure unit override</span>
                                        <input
                                            class="select-input"
                                            placeholder="e.g. bar or p/p0"
                                            [value]="
                                                mapping().unit_overrides[
                                                    'pressure'
                                                ] || ''
                                            "
                                            (input)="
                                                setUnitOverride(
                                                    'pressure',
                                                    $event
                                                )
                                            "
                                        />
                                    </label>
                                    <label>
                                        <span>Uptake unit override</span>
                                        <input
                                            class="select-input"
                                            placeholder="e.g. mmol/g"
                                            [value]="
                                                mapping().unit_overrides[
                                                    'uptake'
                                                ] || ''
                                            "
                                            (input)="
                                                setUnitOverride('uptake', $event)
                                            "
                                        />
                                    </label>
                                    <label>
                                        <span>Temperature unit override</span>
                                        <input
                                            class="select-input"
                                            placeholder="e.g. K or °C"
                                            [value]="
                                                mapping().unit_overrides[
                                                    'temperature'
                                                ] || ''
                                            "
                                            (input)="
                                                setUnitOverride(
                                                    'temperature',
                                                    $event
                                                )
                                            "
                                        />
                                    </label>
                                    <label>
                                        <span>Series delimiter</span>
                                        <input
                                            class="select-input"
                                            maxlength="4"
                                            placeholder="Only for delimited series"
                                            [value]="
                                                mapping().series_delimiter || ''
                                            "
                                            (input)="setSeriesDelimiter($event)"
                                        />
                                    </label>
                                </div>

                                <label class="import-checkbox"><input type="checkbox" [checked]="mapping().whole_file_grouping" (change)="setWholeFileGrouping($event)" /> Treat the entire file as one experiment (use only for a single-isotherm file without an identifier).</label>
                                <h3>Column mapping</h3>
                                <div class="import-table-wrap">
                                    <table class="import-mapping-table">
                                        <thead>
                                            <tr>
                                                <th>Source column</th>
                                                <th>Representative values</th>
                                                <th>Canonical field</th>
                                                <th>Experiment grouping</th>
                                                <th>Detection</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            @for (
                                                column of detected.columns;
                                                track column.name
                                            ) {
                                                <tr>
                                                    <th>{{ column.name }}</th>
                                                    <td>
                                                        {{
                                                            column.sample_values
                                                                .slice(0, 3)
                                                                .join(', ')
                                                        }}
                                                    </td>
                                                    <td>
                                                        <select
                                                            class="select-input"
                                                            [value]="
                                                                mapping()
                                                                    .column_roles[
                                                                    column.name
                                                                ]
                                                            "
                                                            (change)="
                                                                setColumnRole(
                                                                    column.name,
                                                                    $event
                                                                )
                                                            "
                                                        >
                                                            @for (
                                                                role of roles;
                                                                track role.value
                                                            ) {
                                                                <option
                                                                    [value]="
                                                                        role.value
                                                                    "
                                                                    [selected]="
                                                                        mapping()
                                                                            .column_roles[
                                                                            column.name
                                                                        ] ===
                                                                        role.value
                                                                    "
                                                                >
                                                                    {{
                                                                        role.label
                                                                    }}
                                                                </option>
                                                            }
                                                        </select>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="checkbox"
                                                            [checked]="
                                                                mapping().grouping_columns.includes(
                                                                    column.name
                                                                )
                                                            "
                                                            (change)="
                                                                setGrouping(
                                                                    column.name,
                                                                    $event
                                                                )
                                                            "
                                                            [attr.aria-label]="
                                                                'Group experiments by ' +
                                                                column.name
                                                            "
                                                        />
                                                    </td>
                                                    <td>
                                                        <span
                                                            class="confidence-badge"
                                                            >{{
                                                                column.confidence
                                                                    | percent
                                                            }}</span
                                                        >
                                                        @if (
                                                            column.detected_unit
                                                        ) {
                                                            <span
                                                                >{{
                                                                    column.detected_unit
                                                                }}</span
                                                            >
                                                        }
                                                        @if (column.array_like) {
                                                            <span
                                                                >series-like</span
                                                            >
                                                        }
                                                    </td>
                                                </tr>
                                            }
                                        </tbody>
                                    </table>
                                </div>

                                <details class="import-constants">
                                    <summary>
                                        Dataset-wide constants and wide pairs
                                    </summary>
                                    <div class="import-settings-grid">
                                        @for (
                                            constant of constantFields;
                                            track constant.key
                                        ) {
                                            <label>
                                                <span>{{ constant.label }}</span>
                                                <input
                                                    class="select-input"
                                                    [value]="
                                                        mapping().constants[
                                                            constant.key
                                                        ] || ''
                                                    "
                                                    (input)="
                                                        setConstant(
                                                            constant.key,
                                                            $event
                                                        )
                                                    "
                                                />
                                            </label>
                                        }
                                    </div>
                                    <div class="wide-pair-row">
                                        <select
                                            #widePressure
                                            class="select-input"
                                            aria-label="Wide pressure column"
                                        >
                                            @for (
                                                column of detected.columns;
                                                track column.name
                                            ) {
                                                <option [value]="column.name">
                                                    {{ column.name }}
                                                </option>
                                            }
                                        </select>
                                        <select
                                            #wideUptake
                                            class="select-input"
                                            aria-label="Wide uptake column"
                                        >
                                            @for (
                                                column of detected.columns;
                                                track column.name
                                            ) {
                                                <option [value]="column.name">
                                                    {{ column.name }}
                                                </option>
                                            }
                                        </select>
                                        <button
                                            class="button secondary"
                                            type="button"
                                            (click)="
                                                addWidePair(
                                                    widePressure.value,
                                                    wideUptake.value
                                                )
                                            "
                                        >
                                            Add pressure–uptake pair
                                        </button>
                                    </div>
                                    @for (
                                        pair of mapping().wide_pairs;
                                        track pair.pressure_column +
                                            pair.uptake_column
                                    ) {
                                        <p>
                                            {{ pair.pressure_column }} ↔
                                            {{ pair.uptake_column }}
                                            <button
                                                class="button quiet"
                                                type="button"
                                                (click)="removeWidePair($index)"
                                            >
                                                Remove
                                            </button>
                                        </p>
                                    }
                                </details>
                            </section>
                        }

                        @if (step() === 3 && validation(); as checked) {
                            <section class="import-section">
                                <div class="import-summary-grid">
                                    <div>
                                        <span>Validation</span>
                                        <strong>{{ checked.status }}</strong>
                                    </div>
                                    <div>
                                        <span>Experiments</span>
                                        <strong>{{
                                            checked.experiment_count
                                        }}</strong>
                                    </div>
                                    <div>
                                        <span>Observations</span>
                                        <strong>{{
                                            checked.observation_count
                                        }}</strong>
                                    </div>
                                </div>
                                @if (checked.issues.length) {
                                    <div class="import-issues">
                                        @for (
                                            issue of checked.issues;
                                            track issue.code +
                                                issue.source_row +
                                                issue.experiment
                                        ) {
                                            <article
                                                [class]="
                                                    'import-issue ' +
                                                    issue.severity
                                                "
                                            >
                                                @if (
                                                    issue.severity ===
                                                    'confirmation'
                                                ) {
                                                    <input
                                                        type="checkbox"
                                                        [checked]="
                                                            mapping().confirmed_issue_codes.includes(
                                                                issue.code
                                                            )
                                                        "
                                                        (change)="
                                                            confirmIssue(
                                                                issue.code,
                                                                $event
                                                            )
                                                        "
                                                    />
                                                }
                                                <div>
                                                    <strong>{{
                                                        issue.severity
                                                    }}</strong>
                                                    <p>{{ issue.message }}</p>
                                                    @if (issue.remediation) {
                                                        <small>{{
                                                            issue.remediation
                                                        }}</small>
                                                    }
                                                </div>
                                            </article>
                                        }
                                    </div>
                                }
                                <div class="experiment-preview-list">
                                    @for (
                                        experiment of checked.experiments;
                                        track experiment.external_key
                                    ) {
                                        <article>
                                            <header>
                                                <div>
                                                    <strong>{{
                                                        experiment.name
                                                    }}</strong>
                                                    <span
                                                        >{{
                                                            experiment.adsorbate
                                                        }}
                                                        on
                                                        {{
                                                            experiment.adsorbent
                                                        }}</span
                                                    >
                                                </div>
                                                <span
                                                    >{{
                                                        experiment.temperature_k
                                                    }}
                                                    K ·
                                                    {{
                                                        experiment.observation_count
                                                    }}
                                                    points</span
                                                >
                                            </header>
                                            <div class="series-pills">
                                                @for (
                                                    observation of experiment.observations;
                                                    track observation.sequence_index
                                                ) {
                                                    <span
                                                        >{{
                                                            observation.pressure_canonical
                                                                | number: '1.0-4'
                                                        }}
                                                        {{
                                                            observation.pressure_canonical_unit
                                                        }}
                                                        →
                                                        {{
                                                            observation.uptake_mol_kg
                                                                | number: '1.0-4'
                                                        }}
                                                        mol/kg</span
                                                    >
                                                }
                                            </div>
                                        </article>
                                    }
                                </div>
                            </section>
                        }

                        @if (step() === 4 && importedPreview(); as saved) {
                            <section class="import-complete">
                                <div aria-hidden="true">✓</div>
                                <h3>{{ saved.dataset.name }} is ready</h3>
                                <p>
                                    Saved
                                    {{ saved.dataset.experiment_count }}
                                    experiments and
                                    {{ saved.dataset.observation_count }}
                                    canonical observations with source values and
                                    units retained.
                                </p>
                            </section>
                        }
                    }
                </div>

                <footer class="import-wizard-footer">
                    @if (step() > 1 && step() < 4) {
                        <button
                            class="button secondary"
                            type="button"
                            [disabled]="busy()"
                            (click)="back()"
                        >
                            Back
                        </button>
                    } @else {
                        <span></span>
                    }
                    @if (step() === 1) {
                        <button
                            class="button primary"
                            type="button"
                            [disabled]="busy() || !preview()"
                            (click)="step.set(2)"
                        >
                            Review mapping
                        </button>
                    }
                    @if (step() === 2) {
                        <button
                            class="button primary"
                            type="button"
                            [disabled]="busy()"
                            (click)="validate()"
                        >
                            Validate and preview series
                        </button>
                    }
                    @if (step() === 3) {
                        <button
                            class="button primary"
                            type="button"
                            [disabled]="busy() || validation()?.status !== 'valid'"
                            (click)="commit()"
                        >
                            Save validated dataset
                        </button>
                    }
                    @if (step() === 4) {
                        <button
                            class="button primary"
                            type="button"
                            (click)="finish()"
                        >
                            Done
                        </button>
                    }
                </footer>
            </section>
        </div>
    `,
    styles: [
        `
            .import-wizard {
                width: min(1180px, 96vw);
                max-height: 94vh;
                display: grid;
                grid-template-rows: auto auto minmax(0, 1fr) auto;
                overflow: hidden;
                border: 1px solid #d7dee8;
                border-radius: 16px;
                background: #fff;
                box-shadow: 0 28px 70px rgb(3 24 47 / 0.28);
            }
            .import-wizard-header,
            .import-wizard-footer {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: 1rem 1.25rem;
                border-bottom: 1px solid #e3e8ef;
            }
            .import-wizard-header h2,
            .import-wizard-header p {
                margin: 0;
            }
            .import-wizard-header p:not(.eyebrow) {
                margin-top: 0.25rem;
                color: #657386;
            }
            .import-wizard-footer {
                border-top: 1px solid #e3e8ef;
                border-bottom: 0;
            }
            .import-steps {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 0.5rem;
                margin: 0;
                padding: 0.75rem 1.25rem;
                background: #f7f9fc;
                list-style: none;
            }
            .import-steps li {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #718096;
                font-size: 0.8rem;
                font-weight: 650;
            }
            .import-steps span {
                display: grid;
                width: 1.6rem;
                height: 1.6rem;
                place-items: center;
                border-radius: 50%;
                background: #e4eaf2;
            }
            .import-steps li.active {
                color: #075bd5;
            }
            .import-steps li.active span {
                color: #fff;
                background: #075bd5;
            }
            .import-wizard-body {
                min-height: 18rem;
                overflow: auto;
                padding: 1.25rem;
            }
            .import-loading {
                padding: 2rem;
                text-align: center;
            }
            .import-summary-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.75rem;
                margin-bottom: 1rem;
            }
            .import-summary-grid > div {
                display: grid;
                gap: 0.2rem;
                padding: 0.8rem;
                border: 1px solid #e1e7ef;
                border-radius: 10px;
                background: #f8fafc;
            }
            .import-summary-grid span {
                color: #657386;
                font-size: 0.75rem;
            }
            .import-guidance {
                padding: 0.7rem 1rem;
                border-left: 3px solid #3b82f6;
                background: #eff6ff;
            }
            .import-guidance p {
                margin: 0.3rem 0;
            }
            .import-table-wrap {
                max-width: 100%;
                margin-top: 1rem;
                overflow: auto;
                border: 1px solid #dfe5ed;
                border-radius: 10px;
            }
            .import-preview-table,
            .import-mapping-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.78rem;
            }
            th,
            td {
                max-width: 22rem;
                padding: 0.65rem 0.75rem;
                overflow: hidden;
                border-bottom: 1px solid #e7ebf1;
                text-align: left;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            thead th {
                position: sticky;
                top: 0;
                z-index: 1;
                background: #f3f6fa;
            }
            .import-settings-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.75rem;
            }
            .import-settings-grid label {
                display: grid;
                gap: 0.35rem;
                color: #44546a;
                font-size: 0.78rem;
                font-weight: 650;
            }
            .confidence-badge {
                margin-right: 0.5rem;
                padding: 0.15rem 0.4rem;
                border-radius: 999px;
                color: #075bd5;
                background: #e8f1ff;
            }
            .import-constants {
                margin-top: 1rem;
                padding: 0.8rem;
                border: 1px solid #dfe5ed;
                border-radius: 10px;
            }
            .import-constants summary {
                cursor: pointer;
                font-weight: 700;
            }
            .import-constants .import-settings-grid {
                margin-top: 0.8rem;
            }
            .wide-pair-row {
                display: grid;
                grid-template-columns: 1fr 1fr auto;
                gap: 0.5rem;
                margin-top: 0.8rem;
            }
            .import-issues {
                display: grid;
                gap: 0.5rem;
            }
            .import-issue {
                display: flex;
                gap: 0.65rem;
                padding: 0.7rem;
                border-left: 4px solid #ca8a04;
                background: #fffbeb;
            }
            .import-issue.error {
                border-color: #dc2626;
                background: #fff1f2;
            }
            .import-issue p {
                margin: 0.15rem 0;
            }
            .experiment-preview-list {
                display: grid;
                gap: 0.7rem;
                margin-top: 1rem;
            }
            .experiment-preview-list article {
                padding: 0.8rem;
                border: 1px solid #dfe5ed;
                border-radius: 10px;
            }
            .experiment-preview-list header {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
            }
            .experiment-preview-list header div {
                display: grid;
            }
            .series-pills {
                display: flex;
                gap: 0.35rem;
                margin-top: 0.6rem;
                overflow: auto;
            }
            .series-pills span {
                flex: 0 0 auto;
                padding: 0.25rem 0.45rem;
                border-radius: 6px;
                background: #eef2f7;
                font-size: 0.72rem;
            }
            .import-complete {
                display: grid;
                min-height: 18rem;
                place-content: center;
                text-align: center;
            }
            .import-complete > div {
                display: grid;
                width: 3rem;
                height: 3rem;
                place-items: center;
                margin: auto;
                border-radius: 50%;
                color: #fff;
                background: #168454;
                font-size: 1.5rem;
            }
            @media (max-width: 800px) {
                .import-summary-grid,
                .import-settings-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .import-steps li {
                    font-size: 0;
                }
                .import-steps li span {
                    font-size: 0.8rem;
                }
            }
        `,
    ],
})
export class DatasetImportWizardComponent implements OnInit {
    @Input({ required: true }) file!: File;
    @Output() readonly cancelled = new EventEmitter<void>();
    @Output() readonly closed = new EventEmitter<void>();
    @Output() readonly saved = new EventEmitter<number>();

    protected readonly stepLabels = [
        'Preview',
        'Map columns',
        'Validate series',
        'Save',
    ];
    protected readonly roles = COLUMN_ROLES;
    protected readonly constantFields = [
        { key: 'adsorbate', label: 'Constant adsorbate' },
        { key: 'adsorbate_smiles', label: 'Constant adsorbate SMILES' },
        { key: 'adsorbent', label: 'Constant adsorbent' },
        { key: 'temperature', label: 'Constant temperature' },
        { key: 'pressure_unit', label: 'Constant pressure unit' },
        { key: 'uptake_unit', label: 'Constant uptake unit' },
        { key: 'temperature_unit', label: 'Constant temperature unit' },
    ];
    protected readonly step = signal(1);
    protected readonly busy = signal(false);
    protected readonly error = signal('');
    protected readonly preview = signal<ImportPreview | null>(null);
    protected readonly validation = signal<ImportValidation | null>(null);
    protected readonly savedImport = signal<DatasetImportResponse | null>(null);
    protected readonly mapping = signal<ImportMapping>({
        dataset_name: '',
        structure: 'atomic',
        column_roles: {},
        grouping_columns: [],
        constants: {},
        unit_overrides: {},
        pressure_basis: 'absolute',
        decimal_separator: 'auto',
        series_delimiter: null,
        wide_pairs: [],
        duplicate_policy: 'reject',
        confirmed_issue_codes: [],
    });

    protected importedPreview(): DatasetImportResponse | null {
        return this.savedImport();
    }

    async ngOnInit(): Promise<void> {
        this.busy.set(true);
        const result = await previewDataset(this.file);
        this.busy.set(false);
        if (result.error || !result.data) {
            this.error.set(result.error || 'Could not parse this file.');
            return;
        }
        const detected = result.data;
        this.preview.set(detected);
        const structure: ImportableStructure =
            detected.detected_structure === 'ambiguous'
                ? 'atomic'
                : detected.detected_structure;
        const grouping =
            detected.proposed_grouping_columns.length > 0
                ? detected.proposed_grouping_columns
                : detected.columns
                      .filter(
                          (column) =>
                              column.proposed_role === 'experiment_id',
                      )
                      .map((column) => column.name);
        const unitOverrides: Record<string, string> = {};
        for (const column of detected.columns) {
            if (
                column.detected_unit &&
                ['pressure', 'uptake', 'temperature'].includes(
                    column.proposed_role,
                )
            ) {
                unitOverrides[column.proposed_role] = column.detected_unit;
            }
        }
        this.mapping.set({
            ...this.mapping(),
            dataset_name: this.file.name.replace(/\.[^.]+$/, ''),
            structure,
            pressure_basis: detected.proposed_pressure_basis || 'absolute',
            grouping_columns: grouping,
            column_roles: Object.fromEntries(
                detected.columns.map((column) => [
                    column.name,
                    column.proposed_role,
                ]),
            ),
            unit_overrides: unitOverrides,
        });
    }

    protected displayValue(value: unknown): string {
        if (value === null || value === undefined) {
            return '—';
        }
        if (typeof value === 'object') {
            return JSON.stringify(value);
        }
        return String(value);
    }

    protected inputValue(event: Event): string {
        return (event.target as HTMLInputElement).value;
    }

    protected setDatasetName(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            dataset_name: this.inputValue(event),
        }));
    }

    protected setStructure(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            structure: (event.target as HTMLSelectElement)
                .value as ImportableStructure,
        }));
    }

    protected setPressureBasis(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            pressure_basis: (event.target as HTMLSelectElement)
                .value as PressureBasis,
        }));
    }

    protected setDuplicatePolicy(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            duplicate_policy: (event.target as HTMLSelectElement)
                .value as ImportMapping['duplicate_policy'],
        }));
    }

    protected setDecimalSeparator(event: Event): void {
        const value = (event.target as HTMLSelectElement).value;
        if (value === 'auto' || value === '.' || value === ',') {
            this.mapping.update((current) => ({
                ...current,
                decimal_separator: value,
            }));
        }
    }

    protected setSeriesDelimiter(event: Event): void {
        const value = this.inputValue(event);
        this.mapping.update((current) => ({
            ...current,
            series_delimiter: value || null,
        }));
    }

    protected setUnitOverride(quantity: string, event: Event): void {
        const value = this.inputValue(event).trim();
        this.mapping.update((current) => {
            const unit_overrides = { ...current.unit_overrides };
            if (value) {
                unit_overrides[quantity] = value;
            } else {
                delete unit_overrides[quantity];
            }
            return { ...current, unit_overrides };
        });
    }

    protected setConstant(role: string, event: Event): void {
        const value = this.inputValue(event).trim();
        this.mapping.update((current) => {
            const constants = { ...current.constants };
            if (value) {
                constants[role] = value;
            } else {
                delete constants[role];
            }
            return { ...current, constants };
        });
    }

    protected setColumnRole(column: string, event: Event): void {
        const role = (event.target as HTMLSelectElement).value as ColumnRole;
        this.mapping.update((current) => ({
            ...current,
            column_roles: { ...current.column_roles, [column]: role },
        }));
    }

    protected setGrouping(column: string, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.mapping.update((current) => ({
            ...current,
            grouping_columns: checked
                ? [...new Set([...current.grouping_columns, column])]
                : current.grouping_columns.filter((item) => item !== column),
        }));
    }

    protected setWholeFileGrouping(event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.mapping.update((current) => ({ ...current, whole_file_grouping: checked, grouping_columns: checked ? [] : current.grouping_columns }));
    }

    protected addWidePair(pressure: string, uptake: string): void {
        if (!pressure || !uptake || pressure === uptake) {
            return;
        }
        this.mapping.update((current) => ({
            ...current,
            wide_pairs: [
                ...current.wide_pairs,
                { pressure_column: pressure, uptake_column: uptake },
            ],
        }));
    }

    protected removeWidePair(index: number): void {
        this.mapping.update((current) => ({
            ...current,
            wide_pairs: current.wide_pairs.filter(
                (_, itemIndex) => itemIndex !== index,
            ),
        }));
    }

    protected confirmIssue(code: string, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.mapping.update((current) => ({
            ...current,
            confirmed_issue_codes: checked
                ? [...new Set([...current.confirmed_issue_codes, code])]
                : current.confirmed_issue_codes.filter(
                      (candidate) => candidate !== code,
                  ),
        }));
        void this.validate();
    }

    protected back(): void {
        this.error.set('');
        this.step.update((current) => Math.max(1, current - 1));
    }

    protected async validate(): Promise<void> {
        this.busy.set(true);
        this.error.set('');
        const result = await validateDataset(this.file, this.mapping());
        this.busy.set(false);
        if (result.error || !result.data) {
            this.error.set(result.error || 'Validation failed.');
            return;
        }
        this.validation.set(result.data);
        this.step.set(3);
    }

    protected async commit(): Promise<void> {
        this.busy.set(true);
        this.error.set('');
        const result = await commitDataset(this.file, this.mapping());
        this.busy.set(false);
        if (result.error || !result.data) {
            this.error.set(result.error || 'The dataset could not be saved.');
            return;
        }
        this.savedImport.set(result.data);
        this.step.set(4);
    }

    protected finish(): void {
        const dataset = this.savedImport()?.dataset;
        if (dataset) {
            this.saved.emit(dataset.id);
            this.closed.emit();
        }
    }
}
