// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { CommonModule } from '@angular/common';
import {
    AfterViewInit,
    Component,
    ElementRef,
    EventEmitter,
    Input,
    OnInit,
    Output,
    ViewChild,
    computed,
    signal,
} from '@angular/core';
import type {
    DatasetImportResponse,
    ImportIssue,
    ImportMapping,
    ImportPreview,
    ImportValidation,
    ImportableStructure,
    NormalizedExperimentPreview,
    UnitCatalog,
    UnitOption,
    UnitQuantity,
} from '../../models/dataset.model';
import {
    commitDataset,
    fetchDatasetConfiguration,
    previewDataset,
    updateMetadata,
    validateDataset,
} from '../../services/dataset.service';
import {
    ExperimentExplorerComponent,
    type ExplorerExperiment,
    type ExplorerRow,
    type ExplorerStatus,
} from '../../shared/components/experiment-explorer/experiment-explorer.component';
import { findUnit, unitKey } from '../../shared/utils/unit-key';
import { ImportMappingStepComponent } from './import-mapping-step.component';

const QUANTITIES: readonly UnitQuantity[] = ['pressure', 'uptake', 'temperature'];

const NUMBER_FORMAT = new Intl.NumberFormat('en-US', {
    maximumSignificantDigits: 6,
});

function formatNumber(value: number): string {
    return NUMBER_FORMAT.format(value);
}

function formatPressureUnit(unit: string): string {
    return unit === '1' ? 'p/p₀' : unit;
}

@Component({
    selector: 'adsmod-dataset-import-wizard',
    standalone: true,
    imports: [
        CommonModule,
        ExperimentExplorerComponent,
        ImportMappingStepComponent,
    ],
    template: `
        <div class="dataset-modal-backdrop">
            <section
                class="import-wizard"
                role="dialog"
                aria-modal="true"
                aria-labelledby="import-wizard-title"
                (keydown)="handleKeydown($event)"
            >
                <header class="import-wizard-header">
                    <div>
                        <h2 id="import-wizard-title">
                            Understand {{ file.name }}
                        </h2>
                        <p>
                            Map scientific meaning and units before anything is
                            saved.
                        </p>
                    </div>
                    <button
                        #closeButton
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
                            <adsmod-import-mapping-step
                                [preview]="detected"
                                [unitCatalog]="unitCatalog()"
                                [unitIssues]="unitIssues()"
                                [(mapping)]="mapping"
                                [(description)]="description"
                                (catalogChanged)="onCatalogChanged($event)"
                            />
                        }

                        @if (step() === 3 && validation(); as checked) {
                            <section class="import-section">
                                <div
                                    class="validation-summary"
                                    [class]="'validation-summary ' + checked.status"
                                    role="status"
                                >
                                    <div class="summary-status">
                                        <span>Validation</span>
                                        <strong>{{ statusLabel(checked.status) }}</strong>
                                    </div>
                                    <div>
                                        <span>Experiments</span>
                                        <strong>{{ checked.experiment_count }}</strong>
                                    </div>
                                    <div>
                                        <span>Observations</span>
                                        <strong>{{ checked.observation_count }}</strong>
                                    </div>
                                    <div>
                                        <span>Errors</span>
                                        <strong [class.bad]="issueCount(checked, 'error') > 0">{{
                                            issueCount(checked, 'error')
                                        }}</strong>
                                    </div>
                                    <div>
                                        <span>Warnings</span>
                                        <strong>{{
                                            issueCount(checked, 'warning') +
                                                issueCount(checked, 'confirmation')
                                        }}</strong>
                                    </div>
                                </div>
                                @if (checked.status === 'invalid') {
                                    <p class="dataset-status error" role="alert">
                                        Blocking errors must be fixed before this dataset
                                        can be saved. Use Back to adjust the mapping or
                                        units, or correct the source file.
                                    </p>
                                }
                                @if (generalIssues().length) {
                                    <div class="import-issues">
                                        @for (
                                            issue of generalIssues();
                                            track issue.code + issue.source_row + issue.experiment
                                        ) {
                                            <article [class]="'import-issue ' + issue.severity">
                                                @if (issue.severity === 'confirmation') {
                                                    <input
                                                        type="checkbox"
                                                        [attr.aria-label]="'Confirm: ' + issue.message"
                                                        [checked]="mapping().confirmed_issue_codes.includes(issue.code)"
                                                        (change)="confirmIssue(issue.code, $event)"
                                                    />
                                                }
                                                <div>
                                                    <strong>{{
                                                        issue.severity === 'confirmation'
                                                            ? 'needs review'
                                                            : issue.severity
                                                    }}</strong>
                                                    <p>{{ issue.message }}</p>
                                                    @if (issue.remediation) {
                                                        <small>{{ issue.remediation }}</small>
                                                    }
                                                </div>
                                            </article>
                                        }
                                    </div>
                                }
                                <adsmod-experiment-explorer
                                    [experiments]="explorerExperiments()"
                                />
                                @if (checked.experiment_count > checked.experiments.length) {
                                    <p class="explorer-footnote">
                                        Showing the first {{ checked.experiments.length }} of
                                        {{ checked.experiment_count }} experiments. All of them
                                        are validated and saved.
                                    </p>
                                }
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
                            [disabled]="busy() || blockedUnits().length > 0"
                            [attr.title]="
                                blockedUnits().length
                                    ? 'A selected unit has no conversion definition.'
                                    : null
                            "
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
                width: min(1240px, 96vw);
                max-height: 94vh;
                display: grid;
                grid-template-rows: auto auto minmax(0, 1fr) auto;
                overflow: hidden;
                border: 1px solid var(--slate-300);
                border-radius: 16px;
                background: var(--surface);
                box-shadow: 0 28px 70px rgb(3 24 47 / 0.28);
            }
            .import-wizard-header,
            .import-wizard-footer {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: 1rem 1.25rem;
                border-bottom: 1px solid var(--slate-200);
            }
            .import-wizard-header h2,
            .import-wizard-header p {
                margin: 0;
            }
            .import-wizard-header p:not(.eyebrow) {
                margin-top: 0.25rem;
                color: var(--slate-500);
            }
            .import-wizard-footer {
                border-top: 1px solid var(--slate-200);
                border-bottom: 0;
            }
            .import-steps {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 0.5rem;
                margin: 0;
                padding: 0.75rem 1.25rem;
                background: var(--slate-50);
                list-style: none;
            }
            .import-steps li {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--slate-500);
                font-size: 0.8rem;
                font-weight: 650;
            }
            .import-steps span {
                display: grid;
                width: 1.6rem;
                height: 1.6rem;
                place-items: center;
                border-radius: 50%;
                background: var(--slate-200);
            }
            .import-steps li.active {
                color: var(--link);
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
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                background: var(--slate-50);
            }
            .import-summary-grid span {
                color: var(--slate-500);
                font-size: 0.75rem;
            }
            .import-guidance {
                padding: 0.7rem 1rem;
                border-left: 3px solid #3b82f6;
                background: var(--slate-50);
            }
            .import-guidance p {
                margin: 0.3rem 0;
            }
            .import-table-wrap {
                min-width: 0;
                max-width: 100%;
                margin-top: 1rem;
                overflow: auto;
                overscroll-behavior-x: contain;
                border: 1px solid var(--slate-200);
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
                border-bottom: 1px solid var(--slate-200);
                text-align: left;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            thead th {
                position: sticky;
                top: 0;
                z-index: 1;
                background: var(--slate-50);
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
                background: var(--warning-bg);
            }
            .import-issue.error {
                border-color: #dc2626;
                background: var(--danger-bg);
            }
            .import-issue p {
                margin: 0.15rem 0;
            }
            .validation-summary {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
                gap: 0.6rem;
                margin-bottom: 0.9rem;
                padding: 0.7rem;
                border: 1px solid var(--slate-200);
                border-left: 4px solid #16a34a;
                border-radius: 10px;
                background: var(--slate-50);
            }
            .validation-summary.invalid {
                border-left-color: #dc2626;
            }
            .validation-summary.confirmation_required {
                border-left-color: #ca8a04;
            }
            .validation-summary > div {
                display: grid;
                gap: 0.15rem;
            }
            .validation-summary span {
                color: var(--slate-500);
                font-size: 0.72rem;
            }
            .validation-summary strong.bad {
                color: #b91c1c;
            }
            .explorer-footnote {
                margin: 0.6rem 0 0;
                color: var(--slate-500);
                font-size: 0.8rem;
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
                .import-summary-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .import-steps li {
                    font-size: 0;
                }
                .import-steps li span {
                    font-size: 0.8rem;
                }
            }
            @media (max-width: 520px) {
                .import-wizard {
                    width: calc(100vw - 1rem);
                    max-height: calc(100dvh - 1rem);
                    border-radius: 12px;
                }
                .import-wizard-header,
                .import-wizard-footer {
                    gap: 0.65rem;
                    padding: 0.75rem;
                }
                .import-wizard-header h2 {
                    font-size: 1.05rem;
                    line-height: 1.3;
                }
                .import-wizard-header p:not(.eyebrow) {
                    font-size: 0.8rem;
                    line-height: 1.4;
                }
                .import-steps {
                    gap: 0.25rem;
                    padding: 0.6rem 0.75rem;
                }
                .import-steps li {
                    justify-content: center;
                }
                .import-wizard-body {
                    min-height: 0;
                    padding: 0.75rem;
                }
                .import-summary-grid {
                    grid-template-columns: minmax(0, 1fr);
                    gap: 0.6rem;
                }
                .import-summary-grid > div {
                    padding: 0.65rem;
                }
            }
        `,
    ],
})
export class DatasetImportWizardComponent implements OnInit, AfterViewInit {
    @Input({ required: true }) file!: File;
    @Output() readonly cancelled = new EventEmitter<void>();
    @Output() readonly closed = new EventEmitter<void>();
    @Output() readonly saved = new EventEmitter<number>();
    @ViewChild('closeButton') private closeButton?: ElementRef<HTMLButtonElement>;

    protected readonly stepLabels = [
        'Preview',
        'Map columns',
        'Validate series',
        'Save',
    ];
    protected readonly step = signal(1);
    protected readonly busy = signal(false);
    protected readonly error = signal('');
    protected readonly preview = signal<ImportPreview | null>(null);
    protected readonly validation = signal<ImportValidation | null>(null);
    protected readonly savedImport = signal<DatasetImportResponse | null>(null);
    protected readonly unitCatalog = signal<UnitCatalog | null>(null);
    protected readonly unitIssues = signal<ImportIssue[]>([]);
    protected readonly description = signal('');
    protected readonly mapping = signal<ImportMapping>({
        dataset_name: '',
        structure: 'atomic',
        column_roles: {},
        grouping_columns: [],
        constants: {},
        unit_overrides: {},
        unit_definitions: [],
        pressure_basis: 'absolute',
        decimal_separator: 'auto',
        series_delimiter: null,
        wide_pairs: [],
        duplicate_policy: 'reject',
        confirmed_issue_codes: [],
    });

    /** Selected units that are saved as labels without conversion definitions. */
    protected readonly blockedUnits = computed(() => {
        const current = this.mapping();
        const catalog = this.unitCatalog();
        const blocked: string[] = [];
        for (const quantity of QUANTITIES) {
            const unit = current.unit_overrides[quantity];
            if (!unit) {
                continue;
            }
            const known = findUnit(catalog?.[quantity] ?? [], unit);
            const session = current.unit_definitions.find(
                (item) =>
                    item.quantity === quantity &&
                    unitKey(item.symbol) === unitKey(unit),
            );
            if (known ? !known.has_conversion : session?.factor_to_canonical === null) {
                blocked.push(quantity);
            }
        }
        return blocked;
    });

    protected readonly generalIssues = computed(() =>
        (this.validation()?.issues ?? []).filter(
            (issue) => !issue.experiment || issue.severity === 'confirmation',
        ),
    );

    protected readonly explorerExperiments = computed<ExplorerExperiment[]>(() => {
        const checked = this.validation();
        if (!checked) {
            return [];
        }
        const byExperiment = new Map<string, ImportIssue[]>();
        for (const issue of checked.issues) {
            if (issue.experiment) {
                byExperiment.set(issue.experiment, [
                    ...(byExperiment.get(issue.experiment) ?? []),
                    issue,
                ]);
            }
        }
        const seen = new Set<string>();
        const items = checked.experiments.map((experiment) => {
            seen.add(experiment.external_key);
            return this.toExplorer(
                experiment,
                byExperiment.get(experiment.external_key) ?? [],
            );
        });
        for (const [key, issues] of byExperiment) {
            if (!seen.has(key)) {
                items.push(this.issueOnlyExperiment(key, issues));
            }
        }
        return items;
    });

    protected importedPreview(): DatasetImportResponse | null {
        return this.savedImport();
    }

    protected statusLabel(status: ImportValidation['status']): string {
        return status === 'valid'
            ? 'Valid'
            : status === 'invalid'
              ? 'Blocking errors'
              : 'Needs review';
    }

    protected issueCount(
        checked: ImportValidation,
        severity: ImportIssue['severity'],
    ): number {
        return checked.issues.filter((issue) => issue.severity === severity)
            .length;
    }

    private statusOf(issues: readonly ImportIssue[]): ExplorerStatus {
        if (issues.some((issue) => issue.severity === 'error')) {
            return 'error';
        }
        return issues.length > 0 ? 'warning' : 'valid';
    }

    private toExplorer(
        experiment: NormalizedExperimentPreview,
        issues: ImportIssue[],
    ): ExplorerExperiment {
        const observations = experiment.observations;
        const complete = observations.length === experiment.observation_count;
        const pressures = observations.map((item) => item.pressure_canonical);
        const unit = formatPressureUnit(
            observations[0]?.pressure_canonical_unit ?? '',
        );
        const rows: ExplorerRow[] = observations.map((item) => {
            const rowIssues = issues.filter(
                (issue) =>
                    issue.source_row !== null &&
                    issue.source_row === item.source_row,
            );
            const pressureSource = `${formatNumber(item.pressure_original)} ${formatPressureUnit(item.pressure_original_unit)}`;
            const uptakeSource = `${formatNumber(item.uptake_original)} ${item.uptake_original_unit}`;
            const pressure = `${formatNumber(item.pressure_canonical)} ${unit}`;
            const uptake = `${formatNumber(item.uptake_mol_kg)} mol/kg`;
            return {
                point: item.sequence_index + 1,
                pressure,
                pressureSource: pressureSource === pressure ? null : pressureSource,
                uptake,
                uptakeSource: uptakeSource === uptake ? null : uptakeSource,
                status: this.statusOf(rowIssues),
                note: rowIssues[0]?.message ?? null,
            };
        });
        const temperature =
            experiment.temperature_original_unit === 'K'
                ? `${formatNumber(experiment.temperature_k)} K`
                : `${formatNumber(experiment.temperature_original)} ${experiment.temperature_original_unit} (${formatNumber(experiment.temperature_k)} K)`;
        return {
            key: experiment.external_key,
            name: experiment.name,
            subtitle: `${experiment.adsorbate} on ${experiment.adsorbent}`,
            temperature,
            pointCount: experiment.observation_count,
            pressureRange:
                complete && pressures.length > 0
                    ? `${formatNumber(Math.min(...pressures))}–${formatNumber(Math.max(...pressures))} ${unit}`
                    : null,
            status: this.statusOf(issues),
            errors: issues.filter((issue) => issue.severity === 'error').length,
            warnings: issues.filter((issue) => issue.severity !== 'error').length,
            issues,
            rows,
            totalRows: experiment.observation_count,
            paged: false,
            page: 1,
            pageSize: rows.length,
            loading: false,
            error: null,
        };
    }

    private issueOnlyExperiment(
        key: string,
        issues: ImportIssue[],
    ): ExplorerExperiment {
        return {
            key,
            name: key,
            subtitle: 'No valid observations could be read',
            temperature: null,
            pointCount: 0,
            pressureRange: null,
            status: 'error',
            errors: issues.filter((issue) => issue.severity === 'error').length,
            warnings: issues.filter((issue) => issue.severity !== 'error').length,
            issues,
            rows: [],
            totalRows: 0,
            paged: false,
            page: 1,
            pageSize: 0,
            loading: false,
            error: null,
        };
    }

    ngAfterViewInit(): void {
        window.setTimeout(() => this.closeButton?.nativeElement.focus());
    }

    protected handleKeydown(event: KeyboardEvent): void {
        if (event.key === 'Escape') {
            event.preventDefault();
            this.closed.emit();
            return;
        }

        if (event.key !== 'Tab') {
            return;
        }

        const dialog = event.currentTarget instanceof HTMLElement
            ? event.currentTarget
            : null;
        if (!dialog) {
            return;
        }

        const focusableElements = Array.from(
            dialog.querySelectorAll<HTMLElement>(
                'a[href], button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'
            )
        );
        if (focusableElements.length === 0) {
            event.preventDefault();
            return;
        }

        const firstElement = focusableElements[0];
        const activeElement = document.activeElement;
        const currentIndex = focusableElements.indexOf(activeElement as HTMLElement);
        const nextIndex = !dialog.contains(activeElement)
            ? event.shiftKey ? focusableElements.length - 1 : 0
            : event.shiftKey
                ? (currentIndex <= 0 ? focusableElements.length - 1 : currentIndex - 1)
                : (currentIndex === focusableElements.length - 1 ? 0 : currentIndex + 1);
        event.preventDefault();
        (focusableElements[nextIndex] || firstElement).focus();
    }

    async ngOnInit(): Promise<void> {
        this.busy.set(true);
        const [result, catalog] = await Promise.all([
            previewDataset(this.file),
            fetchDatasetConfiguration(),
        ]);
        this.busy.set(false);
        if (catalog.data) {
            this.unitCatalog.set(catalog.data);
        }
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

    protected onCatalogChanged(option: UnitOption): void {
        this.unitCatalog.update((catalog) =>
            catalog
                ? {
                      ...catalog,
                      [option.quantity]: [
                          ...catalog[option.quantity].filter(
                              (item) => item.id !== option.id,
                          ),
                          option,
                      ],
                  }
                : catalog,
        );
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
        const unitIssues = result.data.issues.filter(
            (issue) => issue.code === 'unsupported_unit',
        );
        this.unitIssues.set(unitIssues);
        this.validation.set(result.data);
        this.step.set(unitIssues.length > 0 ? 2 : 3);
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
        const description = this.description().trim();
        if (description) {
            const stored = await updateMetadata(result.data.dataset.id, {
                tags: result.data.dataset.tags,
                description,
            });
            if (stored.error) {
                this.error.set(
                    `The dataset was saved, but its description could not be stored: ${stored.error}`,
                );
            }
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
