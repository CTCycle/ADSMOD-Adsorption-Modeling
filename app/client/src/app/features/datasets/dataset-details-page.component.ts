// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import { Component, HostListener, computed, effect, inject, input, signal, untracked } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import type { EChartsCoreOption } from 'echarts/core';
import type { HasUnsavedChanges } from '../../core/guards/unsaved-changes.guard';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type {
    ColumnRole,
    DatasetDetail,
    DatasetEditIssue,
    DatasetImportInspection,
    DatasetSummary,
    ExperimentSummary,
    ObservationPage,
    UnitCatalog,
} from '../../models/dataset.model';
import {
    fetchDataset,
    fetchDatasetConfiguration,
    fetchDatasetImport,
    fetchExperimentDetail,
    fetchObservations,
    updateDataset,
} from '../../services/dataset.service';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';
import { ChartComponent } from '../dashboards/components/chart.component';
import { DatasetDeleteDialogComponent } from './dataset-delete-dialog.component';
import { DatasetEditSession } from './dataset-edit-session';
import { DATASET_SOURCES } from '../../shared/utils/dataset-source';
import { ExperimentMetaComponent } from './experiment-meta.component';
import { ExperimentNavigatorComponent } from './experiment-navigator.component';
import { MeasurementWorksheetComponent, pressureUnitLabel } from './measurement-worksheet.component';

type PaneView = 'table' | 'plot';

interface ColumnMappingRow {
    name: string;
    role: ColumnRole;
    unit: string;
    grouped: boolean;
}

const PLOT_POINT_LIMIT = 500;
const NOTICE_MS = 4000;
const NUMBER_FORMAT = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 6 });

@Component({
    selector: 'adsmod-dataset-details-page',
    standalone: true,
    imports: [
        ChartComponent,
        ConfirmDialogComponent,
        DatasetDeleteDialogComponent,
        DatePipe,
        ExperimentMetaComponent,
        ExperimentNavigatorComponent,
        MeasurementWorksheetComponent,
        RouterLink,
    ],
    template: `
        <div class="data-page dataset-details-page">
            <a class="back-link" routerLink="/datasets">← Datasets</a>

            @if (notFound()) {
                <section class="console-card details-empty" role="alert">
                    <h2>Dataset not found</h2>
                    <p>This dataset does not exist or was deleted.</p>
                    <a class="button primary" routerLink="/datasets">Back to Datasets</a>
                </section>
            } @else if (dataset(); as current) {
                <header class="console-card details-header" [class.editing]="session.active()">
                    <div class="details-title">
                        @if (session.active()) {
                            <label class="field" [class.invalid]="session.issueFor('dataset', null, 'name')">
                                <span>Dataset name</span>
                                <input type="text" [value]="session.datasetValue('name', current)" (change)="session.setDatasetValue('name', $any($event.target).value, current)" />
                                @if (session.issueFor('dataset', null, 'name'); as problem) { <small role="alert">{{ problem.message }}</small> }
                            </label>
                            <label class="field">
                                <span>Description</span>
                                <textarea rows="2" maxlength="2000" [value]="session.datasetValue('description', current)" (change)="session.setDatasetValue('description', $any($event.target).value, current)"></textarea>
                            </label>
                            <label class="field">
                                <span>Tags (comma separated)</span>
                                <input type="text" [value]="tagsText(session.datasetValue('tags', current))" (change)="setTags($any($event.target).value, current)" />
                            </label>
                        } @else {
                            <h2>{{ current.name }}</h2>
                            <p class="description">{{ current.description || 'No description.' }}</p>
                            @if (current.tags.length) {
                                <p class="tags">@for (tag of current.tags; track tag) { <span>{{ tag }}</span> }</p>
                            }
                        }
                        <dl class="dataset-facts">
                            <div><dt>Origin</dt><dd>{{ originText(current) }}</dd></div>
                            <div><dt>Experiments</dt><dd>{{ current.experiment_count }}</dd></div>
                            <div><dt>Measurements</dt><dd>{{ current.observation_count }}</dd></div>
                            <div><dt>Materials</dt><dd>{{ current.material_count }}</dd></div>
                            <div><dt>Species</dt><dd>{{ current.species_count }}</dd></div>
                            <div><dt>Created</dt><dd>{{ current.created_at | date: 'medium' }}</dd></div>
                            <div><dt>Last update</dt><dd>{{ current.last_update | date: 'medium' }}</dd></div>
                            @if (current.edited_experiment_count) {
                                <div><dt>Edited experiments</dt><dd>{{ current.edited_experiment_count }}</dd></div>
                            }
                        </dl>
                    </div>
                    <div class="details-actions">
                        @if (session.active()) {
                            <button class="button secondary" type="button" [disabled]="session.saving()" (click)="requestCancel()">Cancel</button>
                            <button class="button primary" type="button" [disabled]="session.saving() || !session.dirty()" (click)="save()">
                                {{ session.saving() ? 'Saving…' : 'Save changes' }}
                            </button>
                        } @else {
                            <button class="button primary" type="button" (click)="startEditing()">Edit dataset</button>
                            <button class="button quiet danger" type="button" (click)="pendingDelete.set(current)">Delete</button>
                        }
                    </div>
                </header>

                @if (session.active()) {
                    <div class="edit-banner" role="status">
                        <strong>Editing</strong>
                        <span>{{ session.changeCount() ? session.changeCount() + ' unsaved change' + (session.changeCount() === 1 ? '' : 's') : 'No changes yet' }}. Values are saved only when you choose Save changes.</span>
                    </div>
                }
                @if (saveError()) {
                    <div class="dataset-status error" role="alert">
                        <span>{{ saveError() }}</span>
                    </div>
                }
                @if (session.issues().length) {
                    <section class="issue-summary" aria-labelledby="issue-summary-title">
                        <h3 id="issue-summary-title">{{ session.issues().length }} value{{ session.issues().length === 1 ? ' needs' : 's need' }} attention</h3>
                        <ul>
                            @for (issue of session.issues().slice(0, 12); track $index) {
                                <li>
                                    <span>{{ issueLabel(issue) }}: {{ issue.message }}</span>
                                    @if (issue.experiment_id !== null) {
                                        <button class="button quiet" type="button" (click)="goToIssue(issue)">Go to</button>
                                    }
                                </li>
                            }
                        </ul>
                        @if (session.issues().length > 12) {
                            <p>…and {{ session.issues().length - 12 }} more.</p>
                        }
                    </section>
                }
                @if (notice()) {
                    <div class="dataset-status success" role="status">{{ notice() }}</div>
                }

                <details class="import-details" (toggle)="importToggled($any($event.target).open)">
                    <summary>Provenance and import details</summary>
                    <div class="import-body">
                        @if (detail(); as info) {
                            @if (info.latest_import; as imported) {
                                <p>Imported from <strong>{{ imported.original_filename }}</strong> ({{ imported.source_structure }} layout, parser {{ imported.parser_version }}) on {{ imported.created_at | date: 'medium' }}.</p>
                            } @else {
                                <p>{{ provenanceText(info) }}</p>
                            }
                        }
                        @if (inspection(); as imported) {
                            <div class="data-table-frame data-table-frame--compact">
                                <table class="data-table mapping-table">
                                    <caption class="sr-only">Saved column mapping</caption>
                                    <thead><tr><th scope="col">Source column</th><th scope="col">Meaning</th><th scope="col">Unit</th><th scope="col">Grouping</th></tr></thead>
                                    <tbody>
                                        @for (column of columnMappings(imported); track column.name) {
                                            <tr><th scope="row">{{ column.name }}</th><td>{{ roleLabel(column.role) }}</td><td>{{ column.unit || '—' }}</td><td>{{ column.grouped ? 'Yes' : 'No' }}</td></tr>
                                        }
                                    </tbody>
                                </table>
                            </div>
                            @for (warning of imported.warnings; track $index) {
                                <p class="muted">⚠ {{ warning.message }}</p>
                            }
                        }
                    </div>
                </details>

                <div class="details-body">
                    <adsmod-experiment-navigator
                        class="console-card details-nav"
                        [datasetId]="current.id"
                        [selectedId]="selected()?.id ?? null"
                        [refreshKey]="refreshKey()"
                        [pendingIds]="session.editedExperimentIds()"
                        (selected)="select($event)"
                        (loaded)="experimentsLoaded($event)"
                    />
                    <section class="console-card experiment-pane" aria-live="polite">
                        @if (selected(); as experiment) {
                            <div class="pane-head">
                                <h3>{{ experiment.name }}</h3>
                                <div class="pane-toggle" role="group" aria-label="Measurement view">
                                    <button type="button" [class.active]="pane() === 'table'" [attr.aria-pressed]="pane() === 'table'" (click)="pane.set('table')">Table</button>
                                    <button type="button" [class.active]="pane() === 'plot'" [attr.aria-pressed]="pane() === 'plot'" [disabled]="session.active()" (click)="pane.set('plot')">Plot</button>
                                </div>
                            </div>
                            <adsmod-experiment-meta [experiment]="experiment" [session]="session" [editing]="session.active()" [units]="units()" />
                            @if (pane() === 'table' || session.active()) {
                                <adsmod-measurement-worksheet
                                    [datasetId]="current.id"
                                    [experiment]="experiment"
                                    [session]="session"
                                    [editing]="session.active()"
                                    [refreshKey]="refreshKey()"
                                    [focusSequence]="focusSequence()"
                                />
                            } @else if (plotError()) {
                                <p class="dataset-status error" role="alert">{{ plotError() }}</p>
                            } @else if (plotOption(); as option) {
                                <div class="plot-frame"><adsmod-chart [option]="option" ariaLabel="Isotherm: uptake against pressure" /></div>
                                @if ((plotRows()?.total ?? 0) > plotLimit) {
                                    <p class="muted">Showing the first {{ plotLimit }} measurements.</p>
                                }
                            } @else {
                                <p class="muted" role="status">Loading measurements…</p>
                            }
                        } @else {
                            <p class="muted">Select an experiment to inspect its measurements.</p>
                        }
                    </section>
                </div>
                <adsmod-dataset-delete-dialog [dataset]="pendingDelete()" (closed)="pendingDelete.set(null)" (deleted)="deleted()" />
                <adsmod-confirm-dialog
                    [open]="cancelConfirmOpen()"
                    title="Discard unsaved changes?"
                    [message]="session.changeCount() + ' change(s) will be lost.'"
                    confirmLabel="Discard changes"
                    (confirmed)="cancelEditing()"
                    (closed)="cancelConfirmOpen.set(false)"
                />
            } @else {
                <p class="muted" role="status">Loading dataset…</p>
            }
        </div>
    `,
    styles: [
        `
            :host { display: block; }
            .dataset-details-page { display: grid; gap: var(--spacing-md); }
            .back-link { justify-self: start; font-size: 0.85rem; font-weight: 600; }
            .details-header { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--spacing-md); padding: var(--spacing-md) var(--spacing-lg); }
            .details-header.editing { box-shadow: inset 4px 0 0 var(--primary-600); }
            .details-title { display: grid; gap: 0.35rem; min-width: 0; }
            .details-title h2 { margin: 0; font-size: 1.25rem; overflow-wrap: anywhere; }
            .description { margin: 0; color: var(--text-secondary); font-size: 0.85rem; }
            .tags { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0; }
            .tags span { padding: 1px 8px; border-radius: 999px; background: var(--slate-100); font-size: 0.7rem; }
            .dataset-facts { display: flex; flex-wrap: wrap; gap: 0.4rem 1.5rem; margin: 0.35rem 0 0; }
            .dataset-facts dt { color: var(--text-tertiary); font-size: 0.62rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }
            .dataset-facts dd { margin: 0; font-size: 0.82rem; font-variant-numeric: tabular-nums; }
            .details-actions { display: flex; align-items: flex-start; gap: 0.4rem; }
            .field { display: grid; gap: 0.25rem; color: var(--text-secondary); font-size: 0.72rem; font-weight: 600; }
            .field input, .field textarea { width: 100%; }
            .field textarea { min-height: 3.5rem; }
            .field.invalid input { border-color: var(--danger); }
            .field small { color: var(--danger-text); font-weight: 500; }
            .edit-banner { display: flex; flex-wrap: wrap; gap: 0.5rem; padding: 0.55rem 0.9rem; border-radius: var(--radius-sm); background: color-mix(in srgb, var(--primary-600) 10%, var(--surface)); color: var(--text-primary); font-size: 0.82rem; }
            .issue-summary { padding: 0.6rem 0.9rem; border: 1px solid var(--danger-border); border-radius: var(--radius-sm); background: var(--danger-bg); color: var(--danger-text); }
            .issue-summary h3 { margin: 0 0 0.35rem; font-size: 0.85rem; }
            .issue-summary ul { display: grid; gap: 0.15rem; margin: 0; padding: 0; list-style: none; font-size: 0.78rem; }
            .issue-summary li { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
            .issue-summary p { margin: 0.3rem 0 0; font-size: 0.75rem; }
            .import-details { padding: 0 var(--spacing-xs); }
            .import-details summary { cursor: pointer; color: var(--text-secondary); font-size: 0.8rem; font-weight: 600; }
            .import-body { display: grid; gap: 0.5rem; padding-top: 0.5rem; font-size: 0.8rem; }
            .import-body p { margin: 0; }
            .mapping-table { --data-table-min-width: 480px; }
            .details-body { display: grid; grid-template-columns: minmax(15rem, 18rem) minmax(0, 1fr); gap: var(--spacing-md); align-items: start; }
            .details-nav { position: sticky; top: var(--spacing-md); padding: var(--spacing-sm); }
            .experiment-pane { display: grid; gap: var(--spacing-sm); min-width: 0; padding: var(--spacing-md); }
            .pane-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
            .pane-head h3 { margin: 0; overflow: hidden; font-size: 1rem; text-overflow: ellipsis; white-space: nowrap; }
            .pane-toggle { display: inline-flex; border: 1px solid var(--border-strong); border-radius: var(--radius-sm); overflow: hidden; }
            .pane-toggle button { min-height: 2rem; padding: 0 0.8rem; border: 0; background: var(--surface); color: var(--text-secondary); font: inherit; font-size: 0.78rem; font-weight: 600; cursor: pointer; }
            .pane-toggle button.active { background: var(--slate-100); color: var(--text-primary); }
            .pane-toggle button:disabled { cursor: not-allowed; opacity: 0.5; }
            .plot-frame { height: 26rem; }
            .muted { margin: 0; color: var(--text-secondary); font-size: 0.82rem; }
            .details-empty { padding: var(--spacing-lg); }
            @media (max-width: 960px) {
                .details-body { grid-template-columns: minmax(0, 1fr); }
                .details-nav { position: static; }
            }
            @media (max-width: 640px) {
                .details-header { grid-template-columns: 1fr; padding: var(--spacing-sm) var(--spacing-md); }
                .experiment-pane { padding: var(--spacing-sm); }
            }
        `,
    ],
})
export class DatasetDetailsPageComponent implements HasUnsavedChanges {
    /** Route parameter. */
    readonly id = input<string>();
    /** Optional `?experiment=` query parameter selecting an experiment. */
    readonly experiment = input<string>();

    protected readonly store = inject(CoreWorkspaceStore);
    private readonly router = inject(Router);
    protected readonly session = new DatasetEditSession();
    protected readonly plotLimit = PLOT_POINT_LIMIT;

    protected readonly notFound = signal(false);
    protected readonly detail = signal<DatasetDetail | null>(null);
    protected readonly inspection = signal<DatasetImportInspection | null>(null);
    protected readonly units = signal<UnitCatalog | null>(null);
    protected readonly selected = signal<ExperimentSummary | null>(null);
    protected readonly pane = signal<PaneView>('table');
    protected readonly refreshKey = signal(0);
    protected readonly focusSequence = signal<number | null>(null);
    protected readonly pendingDelete = signal<DatasetSummary | null>(null);
    protected readonly cancelConfirmOpen = signal(false);
    protected readonly saveError = signal('');
    protected readonly notice = signal('');
    protected readonly plotRows = signal<ObservationPage | null>(null);
    protected readonly plotError = signal('');

    protected readonly datasetId = computed(() => {
        const parsed = Number(this.id());
        return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
    });
    protected readonly dataset = computed(() => this.store.datasets().find((item) => item.id === this.datasetId()));
    protected readonly plotOption = computed<EChartsCoreOption | null>(() => {
        const page = this.plotRows();
        if (!page) return null;
        const unit = pressureUnitLabel(page.rows[0]?.pressure_canonical_unit ?? null);
        return {
            animation: false,
            textStyle: { fontFamily: 'inherit' },
            grid: { left: 70, right: 24, top: 24, bottom: 52 },
            tooltip: {
                trigger: 'item',
                formatter: (params: { value: [number, number] }) =>
                    `p = ${NUMBER_FORMAT.format(params.value[0])} ${unit}<br>q = ${NUMBER_FORMAT.format(params.value[1])} mol/kg`,
            },
            xAxis: { type: 'value', name: `Pressure (${unit})`, nameLocation: 'middle', nameGap: 32, scale: true },
            yAxis: { type: 'value', name: 'Uptake (mol/kg)', nameLocation: 'middle', nameGap: 50, scale: true },
            series: [
                {
                    type: 'line',
                    symbol: 'circle',
                    symbolSize: 7,
                    lineStyle: { width: 1.5, opacity: 0.7 },
                    data: page.rows.map((row) => [row.pressure_canonical, row.uptake_mol_kg]),
                },
            ],
        };
    });

    private loadRevision = 0;
    private noticeTimer: ReturnType<typeof setTimeout> | null = null;

    constructor() {
        effect(() => {
            const id = this.datasetId();
            untracked(() => void this.load(id));
        });
        effect(() => {
            const experiment = this.selected();
            const pane = this.pane();
            if (experiment && pane === 'plot') untracked(() => void this.loadPlot(experiment.id));
        });
        void this.loadUnits();
    }

    hasUnsavedChanges(): boolean {
        return this.session.active() && this.session.dirty();
    }

    @HostListener('window:beforeunload', ['$event'])
    protected warnBeforeUnload(event: BeforeUnloadEvent): void {
        if (this.hasUnsavedChanges()) {
            event.preventDefault();
            event.returnValue = '';
        }
    }

    private async load(id: number | null): Promise<void> {
        const revision = ++this.loadRevision;
        this.notFound.set(false);
        this.detail.set(null);
        this.inspection.set(null);
        this.selected.set(null);
        this.session.cancel();
        if (id === null) {
            this.notFound.set(true);
            return;
        }
        const [detail] = await Promise.all([fetchDataset(id), this.store.datasets().some((item) => item.id === id) ? null : this.store.refreshDatasets()]);
        if (revision !== this.loadRevision) return;
        if (!detail.data) {
            this.notFound.set(true);
            return;
        }
        this.detail.set(detail.data);
    }

    private async loadUnits(): Promise<void> {
        const result = await fetchDatasetConfiguration();
        if (result.data) this.units.set(result.data);
    }

    /** Keep the selection current, honouring a `?experiment=` deep link on first load. */
    protected experimentsLoaded(items: ExperimentSummary[]): void {
        const current = this.selected();
        if (current) {
            const fresh = items.find((item) => item.id === current.id);
            if (fresh) this.selected.set(fresh);
            return;
        }
        const requested = Number(this.experiment());
        const linked = items.find((item) => item.id === requested);
        if (linked) this.select(linked);
        else if (Number.isInteger(requested) && requested > 0) void this.selectById(requested, items[0]);
        else if (items.length) this.select(items[0]);
    }

    protected select(experiment: ExperimentSummary): void {
        this.focusSequence.set(null);
        this.selected.set(experiment);
        void this.router.navigate([], { queryParams: { experiment: experiment.id }, queryParamsHandling: 'merge', replaceUrl: true });
    }

    private async selectById(experimentId: number, fallback?: ExperimentSummary): Promise<void> {
        const datasetId = this.datasetId();
        if (datasetId === null) return;
        if (this.selected()?.id === experimentId) return;
        const result = await fetchExperimentDetail(datasetId, experimentId);
        if (result.data) this.select(result.data.experiment);
        else if (fallback) this.select(fallback);
    }

    protected startEditing(): void {
        this.saveError.set('');
        this.pane.set('table');
        this.session.start();
    }

    protected requestCancel(): void {
        if (this.session.dirty()) this.cancelConfirmOpen.set(true);
        else this.cancelEditing();
    }

    protected cancelEditing(): void {
        this.cancelConfirmOpen.set(false);
        this.saveError.set('');
        this.session.cancel();
        this.refreshKey.update((value) => value + 1);
    }

    protected async save(): Promise<void> {
        const datasetId = this.datasetId();
        if (datasetId === null) return;
        this.saveError.set('');
        const local = this.session.validateLocally();
        if (local.length) {
            this.session.issues.set(local);
            return;
        }
        this.session.saving.set(true);
        const result = await updateDataset(datasetId, this.session.toRequest());
        this.session.saving.set(false);
        if (!result.data) {
            this.session.issues.set(result.issues);
            this.saveError.set(result.error ?? 'The changes could not be saved.');
            return;
        }
        this.session.cancel();
        await this.store.refreshDatasets();
        const detail = await fetchDataset(datasetId);
        if (detail.data) this.detail.set(detail.data);
        this.refreshKey.update((value) => value + 1);
        this.showNotice('Changes saved.');
    }

    protected async goToIssue(issue: DatasetEditIssue): Promise<void> {
        if (issue.experiment_id === null) return;
        await this.selectById(issue.experiment_id);
        const location = issue.observation_id === null ? null : this.session.observationLocation(issue.observation_id);
        this.focusSequence.set(location?.sequenceIndex ?? null);
    }

    protected issueLabel(issue: DatasetEditIssue): string {
        if (issue.scope === 'dataset') return 'Dataset';
        const experiment = issue.experiment_id === null ? '' : (this.session.experimentLabel(issue.experiment_id) ?? (issue.experiment_id === this.selected()?.id ? this.selected()?.name : `Experiment #${issue.experiment_id}`));
        if (issue.scope === 'experiment') return experiment ?? 'Experiment';
        const location = issue.observation_id === null ? null : this.session.observationLocation(issue.observation_id);
        return `${experiment}, measurement ${location ? location.sequenceIndex + 1 : '#' + issue.observation_id}`;
    }

    protected deleted(): void {
        this.session.cancel();
        void this.router.navigate(['/datasets']);
    }

    protected setTags(raw: string, current: DatasetSummary): void {
        const tags = [...new Set(raw.split(',').map((tag) => tag.trim()).filter(Boolean))];
        this.session.setDatasetValue('tags', tags, current);
    }

    protected tagsText(tags: string[]): string {
        return tags.join(', ');
    }

    protected originText(dataset: DatasetSummary): string {
        const imported = this.detail()?.latest_import;
        return imported ? `${DATASET_SOURCES[dataset.source].label} · ${imported.original_filename}` : DATASET_SOURCES[dataset.source].label;
    }

    protected provenanceText(info: DatasetDetail): string {
        const repository = info.provenance['repository'];
        return typeof repository === 'string'
            ? `Acquired from ${repository}. Each experiment keeps its source identifier; re-downloading asks before overwriting local edits.`
            : 'No import manifest is stored for this dataset.';
    }

    protected importToggled(open: boolean): void {
        const datasetId = this.datasetId();
        if (!open || this.inspection() || datasetId === null || !this.detail()?.latest_import) return;
        void fetchDatasetImport(datasetId).then((result) => this.inspection.set(result.data));
    }

    protected columnMappings(imported: DatasetImportInspection): ColumnMappingRow[] {
        const mapping = imported.mapping;
        const first = imported.validation.experiments[0];
        const sourceUnits: Partial<Record<ColumnRole, string>> = {
            pressure: first?.observations[0]?.pressure_original_unit,
            uptake: first?.observations[0]?.uptake_original_unit,
            temperature: first?.temperature_original_unit,
        };
        return Object.entries(mapping.column_roles).map(([name, role]) => ({
            name,
            role,
            unit: mapping.unit_overrides[role] || (sourceUnits[role] ? pressureUnitLabel(sourceUnits[role] ?? null) : ''),
            grouped: mapping.grouping_columns.includes(name),
        }));
    }

    protected roleLabel(role: ColumnRole): string {
        return role
            .split('_')
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(' ');
    }

    private async loadPlot(experimentId: number): Promise<void> {
        const datasetId = this.datasetId();
        if (datasetId === null) return;
        this.plotRows.set(null);
        this.plotError.set('');
        const result = await fetchObservations(datasetId, experimentId, 0, PLOT_POINT_LIMIT, 'pressure');
        if (this.selected()?.id !== experimentId) return;
        if (result.data) this.plotRows.set(result.data);
        else this.plotError.set(result.error || 'Failed to load measurements.');
    }

    private showNotice(message: string): void {
        this.notice.set(message);
        if (this.noticeTimer) clearTimeout(this.noticeTimer);
        this.noticeTimer = setTimeout(() => this.notice.set(''), NOTICE_MS);
    }
}
