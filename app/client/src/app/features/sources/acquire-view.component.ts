// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { AfterViewInit, Component, DestroyRef, ElementRef, OnInit, computed, inject, signal, viewChild } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import type {
    ChemicalLookupItem,
    CODSearchResult,
    NISTCategory,
    NISTCollectionView,
    PublicDataOverviewResponse,
    PublicDataSyncRun,
} from '../../models/public-data.model';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';
import { JobsMonitorService } from '../../services/jobs-monitor.service';
import {
    enrichChemicals as startChemicalEnrichment,
    enrichConformers as startConformerEnrichment,
    fetchBulkStructures as startBulkStructures,
    fetchNistCollections,
    fetchNistEditedExperiments,
    fetchNistIsotherms,
    fetchPublicDataSyncRun,
    fetchPublicOverview,
    importCOD,
    lookupChemicals,
    resolvePubChem,
    searchCOD,
    startPublicDataSync,
    type ApiResult,
} from '../../services/public-data.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import { EntityPickerComponent, type PickedEntity } from '../../shared/components/entity-picker/entity-picker.component';
import { SegmentedControlComponent, type SegmentedOption } from '../../shared/components/segmented-control/segmented-control.component';
import { formatDate, isActiveStatus, NIST_COLLECTIONS } from '../../shared/utils/job-presentation';
import { AcquireActivityComponent } from './acquire-activity.component';
import { NistTargetedSearchComponent } from './nist-targeted-search.component';

type AcquireMode = 'single' | 'bulk';
type AcquireCard = 'nist' | 'pubchem' | 'cod';

const MODE_OPTIONS: readonly SegmentedOption<AcquireMode>[] = [
    { value: 'single', label: 'Single' },
    { value: 'bulk', label: 'Bulk' },
];
const AVAILABILITY_POLL_MS = 800;
const MAX_COD_SELECTION = 50;

const readFraction = (category: NISTCategory): number => {
    try {
        const value = Number.parseFloat(window.localStorage.getItem(`adsmod.nist.fraction.${category}`) ?? '');
        return Number.isFinite(value) && value > 0 && value <= 1 ? value : 0.1;
    } catch {
        return 0.1;
    }
};

const readMode = (card: AcquireCard, fallback: AcquireMode): AcquireMode => {
    try {
        const value = window.localStorage.getItem(`adsmod.acquire.mode.${card}`);
        return value === 'single' || value === 'bulk' ? value : fallback;
    } catch {
        return fallback;
    }
};

const persistMode = (card: AcquireCard, mode: AcquireMode): void => {
    try {
        window.localStorage.setItem(`adsmod.acquire.mode.${card}`, mode);
    } catch {
        // Storage is a convenience only.
    }
};

/** Shares are entered as 1-100 percent and stored as the 0-1 fraction the API takes. */
const percentToFraction = (value: number | string | null): number | null => {
    if (value === null || value === '') return null;
    const percent = Number(value);
    return Number.isFinite(percent) && percent >= 1 && percent <= 100 ? percent / 100 : null;
};

const persistFraction = (category: NISTCategory, value: number): void => {
    try {
        window.localStorage.setItem(`adsmod.nist.fraction.${category}`, String(value));
    } catch {
        // Storage is a convenience only.
    }
};

@Component({
    selector: 'adsmod-public-data-acquire',
    standalone: true,
    imports: [FormsModule, AcquireActivityComponent, EntityPickerComponent, DataPaginationComponent, ConfirmDialogComponent, NistTargetedSearchComponent, SegmentedControlComponent],
    template: `
        @if (error()) {
            <div class="public-alert error" role="alert"><strong>Request failed.</strong><span>{{ error() }}</span><button type="button" class="button secondary" (click)="error.set(null)">Dismiss</button></div>
        }

        <section class="public-section source-panel" id="acquire-nist">
            <div class="section-heading">
                <div>
                    <h4>Adsorption collections</h4>
                    <p>{{ nistMode() === 'bulk' ? 'Fetch NIST records in bounded runs; fetch again to continue.' : 'Search the NIST index and fetch only the isotherms you pick.' }}</p>
                </div>
                <div class="section-actions">
                    @if (nistMode() === 'bulk') {
                        <button class="table-action availability-action" type="button" [disabled]="isSubmitting('nist:index:all')" [attr.aria-busy]="isSubmitting('nist:index:all')" (click)="checkAllAvailability()">{{ isSubmitting('nist:index:all') ? 'Checking…' : (availabilityKnown() ? 'Refresh availability' : 'Check availability') }}</button>
                    }
                    <adsmod-segmented-control label="NIST acquisition mode" [options]="modeOptions" [value]="nistMode()" (valueChange)="setMode('nist', $event)" />
                </div>
            </div>
            @if (nistMode() === 'single') {
                <adsmod-nist-targeted-search #targeted [submitting]="isSubmitting('nist:targeted')" (fetchSelected)="fetchTargeted($event)" (failed)="error.set($event)" />
            } @else {
            <div class="data-table-frame collection-table-frame">
                <table class="data-table collection-table">
                    <caption class="sr-only">NIST adsorption collections</caption>
                    <thead>
                        <tr>
                            <th class="col-collection" scope="col">Collection</th>
                            <th class="col-count" scope="col">Stored locally / available</th>
                            <th class="col-fraction" scope="col">Fetch share</th>
                            <th class="col-actions" scope="col"></th>
                        </tr>
                    </thead>
                    <tbody>
                        @for (collection of collectionRows; track collection.category) {
                            @let state = collectionState(collection.category);
                            @let busy = isBusy('nist', collection.category);
                            @let shareKey = 'nist:' + collection.category;
                            <tr class="collection-row">
                                <td>
                                    <strong class="collection-name">{{ collection.label }}</strong>
                                </td>
                                <td>
                                    <div class="collection-count">
                                        @if (state.available_count === null) {
                                            <strong>{{ state.local_count.toLocaleString() }} stored</strong>
                                            <span>Availability not checked</span>
                                        } @else {
                                            <strong>{{ state.local_count.toLocaleString() }} / {{ state.available_count.toLocaleString() }} records</strong>
                                            @if (state.available_count > 0) {
                                                <div class="coverage-meter">
                                                    <progress max="100" [value]="coveragePercent(state)" [attr.aria-label]="collection.label + ' local coverage'"></progress>
                                                    <span>{{ coveragePercent(state) }}%</span>
                                                </div>
                                            }
                                            <span>Checked {{ formatDate(state.indexed_at) }}</span>
                                        }
                                    </div>
                                </td>
                                <td>
                                    <label class="collection-fraction share-field">
                                        <input
                                            type="number"
                                            min="1"
                                            max="100"
                                            step="1"
                                            [class.invalid]="isShareInvalid(shareKey)"
                                            [attr.aria-invalid]="isShareInvalid(shareKey)"
                                            [ngModel]="toPercent(fractions()[collection.category])"
                                            (ngModelChange)="setFraction(collection.category, $event)"
                                            [attr.aria-label]="'Percentage of the ' + collection.label + ' index to fetch'"
                                        />
                                        <span aria-hidden="true">%</span>
                                    </label>
                                    @if (isShareInvalid(shareKey)) {
                                        <span class="share-preview invalid">Enter 1 to 100</span>
                                    } @else if (state.available_count) {
                                        <span class="share-preview">First {{ shareCount(fractions()[collection.category], state.available_count).toLocaleString() }} of {{ state.available_count.toLocaleString() }}</span>
                                    }
                                </td>
                                <td>
                                    <div class="table-actions">
                                        <button class="button primary" type="button" [disabled]="busy || isShareInvalid(shareKey) || isSubmitting('nist:fetch:' + collection.category)" (click)="fetchCollection(collection.category)">{{ busy ? 'Running…' : 'Fetch' }}</button>
                                    </div>
                                </td>
                            </tr>
                        }
                    </tbody>
                </table>
            </div>
            }
            <adsmod-acquire-activity source="nist" [sessionRunIds]="sessionRunIds()" />
        </section>

        <div class="acquire-grid">
            <section class="public-section source-panel" id="acquire-pubchem">
                <div class="section-heading">
                    <div>
                        <h4>Chemical identity and properties</h4>
                        <p>{{ pubchemMode() === 'single' ? 'Look up one compound, then import it.' : 'Enrich stored species in bulk.' }}</p>
                    </div>
                    <adsmod-segmented-control label="Chemical acquisition mode" [options]="modeOptions" [value]="pubchemMode()" (valueChange)="setMode('pubchem', $event)" />
                </div>
                <p class="acq-stats">
                    @if (overview(); as stats) {
                        <span><strong>{{ stats.species_enriched.toLocaleString() }} / {{ stats.species_total.toLocaleString() }}</strong> species enriched from PubChem</span>
                        @if (stats.species_total > 0) {
                            <progress max="100" [value]="percent(stats.species_enriched, stats.species_total)" aria-label="Enriched species coverage"></progress>
                            <span>{{ percent(stats.species_enriched, stats.species_total) }}%</span>
                        }
                    } @else {
                        <span>Coverage unavailable</span>
                    }
                </p>
                @if (pubchemMode() === 'single') {
                    @if (pubchemTarget(); as target) {
                        <p class="target-chip">Attaching the result to <strong>{{ target.name }}</strong><button class="text-button" type="button" (click)="pubchemTarget.set(null)">Clear</button></p>
                    }
                    <div class="query-row">
                        <label class="query-field"><span>Compound</span><input id="acquire-pubchem-query" [(ngModel)]="pubchemQuery" placeholder="CID, InChIKey, or name" (keyup.enter)="lookupCompound()" /></label>
                        <button class="button primary" type="button" [disabled]="!pubchemQuery.trim() || pubchemLooking()" (click)="lookupCompound()">{{ pubchemLooking() ? 'Searching…' : 'Search' }}</button>
                    </div>
                    @if (pubchemMessage(); as note) { <p class="run-note result-note" role="status">{{ note }}</p> }
                    @if (pubchemResults().length) {
                        <div class="data-table-frame data-table-frame--compact chemical-results-frame">
                            <table class="data-table chemical-results-table">
                                <caption class="sr-only">PubChem matches</caption>
                                <thead><tr><th scope="col">Compound</th><th scope="col">Formula</th><th scope="col">CID</th><th class="col-actions" scope="col"></th></tr></thead>
                                <tbody>
                                    @for (item of pubchemResults(); track item.cid) {
                                        <tr>
                                            <td class="truncate-cell" [title]="item.name + (item.inchi_key ? ' · ' + item.inchi_key : '')">{{ item.name }}</td>
                                            <td class="truncate-cell" [title]="item.formula || ''">{{ item.formula || '—' }}</td>
                                            <td><a [href]="item.source_url" target="_blank" rel="noopener noreferrer">{{ item.cid }}</a></td>
                                            <td><button class="table-action" type="button" [disabled]="isSubmitting('pubchem:resolve:' + item.cid)" (click)="importCompound(item.cid)">{{ item.local_adsorbate_id ? 'Stored · refresh' : 'Import' }}</button></td>
                                        </tr>
                                    }
                                </tbody>
                            </table>
                        </div>
                        @if (pubchemResults().length > 1) { <p class="run-note">Several compounds match; import the one you need.</p> }
                    }
                } @else {
                    <div class="bulk-controls">
                        <label class="bulk-share share-field"><span>Share</span><input type="number" min="1" max="100" step="1" [class.invalid]="isShareInvalid('bulk:pubchem')" [attr.aria-invalid]="isShareInvalid('bulk:pubchem')" [ngModel]="toPercent(bulkFractions().pubchem)" (ngModelChange)="setBulkFraction('pubchem', $event)" aria-label="Percentage of unenriched species to enrich" /><span aria-hidden="true">%</span></label>
                        <button class="button primary compact" type="button" [disabled]="isBusy('pubchem', 'pubchem_enrich') || isSubmitting('pubchem:enrich') || isShareInvalid('bulk:pubchem') || !!enrichBlocked()" (click)="enrichChemicals()">{{ isSubmitting('pubchem:enrich') ? 'Starting…' : 'Enrich species' }}</button>
                        <button class="button quiet compact bulk-secondary" type="button" [disabled]="isBusy('pubchem', 'pubchem_enrich') || isSubmitting('pubchem:conformers') || isShareInvalid('bulk:pubchem') || conformersBlocked()" (click)="fetchConformers()">{{ isSubmitting('pubchem:conformers') ? 'Starting…' : 'Fetch 3D structures' }}</button>
                    </div>
                    <p class="run-note bulk-hint" [class.invalid]="isShareInvalid('bulk:pubchem')">{{ isShareInvalid('bulk:pubchem') ? 'Enter a share from 1 to 100.' : enrichHint() }}</p>
                    <details class="how-it-works">
                        <summary>How this works</summary>
                        <p>Looks up stored species that have no PubChem identity yet{{ overview() ? ' (' + (overview()!.species_total - overview()!.species_enriched).toLocaleString() + ' remaining)' : '' }}. Species already enriched or confirmed unresolvable are skipped. “Fetch 3D structures” downloads the 3D conformer for enriched species that have one but no local copy yet; it can also be fetched one species at a time from the Datasets detail panel.</p>
                    </details>
                }
                <adsmod-acquire-activity source="pubchem" [sessionRunIds]="sessionRunIds()" />
            </section>

            <section class="public-section source-panel" id="acquire-cod">
                <div class="section-heading">
                    <div>
                        <h4>Crystal structures</h4>
                        <p>{{ codMode() === 'single' ? 'Search COD, inspect matches, and import selected structures.' : 'Look up structures for stored materials in bulk.' }}</p>
                    </div>
                    <adsmod-segmented-control label="Structure acquisition mode" [options]="modeOptions" [value]="codMode()" (valueChange)="setMode('cod', $event)" />
                </div>
                <p class="acq-stats">
                    @if (overview(); as stats) {
                        <span><strong>{{ stats.materials_with_structure.toLocaleString() }} / {{ stats.materials_total.toLocaleString() }}</strong> materials with a structure · {{ stats.structures.toLocaleString() }} from COD</span>
                        @if (stats.materials_total > 0) {
                            <progress max="100" [value]="percent(stats.materials_with_structure, stats.materials_total)" aria-label="Materials with a structure"></progress>
                            <span>{{ percent(stats.materials_with_structure, stats.materials_total) }}%</span>
                        }
                    } @else {
                        <span>Coverage unavailable</span>
                    }
                </p>
                @if (codMode() === 'bulk') {
                    <div class="bulk-controls">
                        <label class="bulk-share share-field"><span>Share</span><input type="number" min="1" max="100" step="1" [class.invalid]="isShareInvalid('bulk:cod')" [attr.aria-invalid]="isShareInvalid('bulk:cod')" [ngModel]="toPercent(bulkFractions().cod)" (ngModelChange)="setBulkFraction('cod', $event)" aria-label="Percentage of materials without a structure to look up" /><span aria-hidden="true">%</span></label>
                        <button class="button primary compact" type="button" [disabled]="isBusy('cod', 'cod_import') || isSubmitting('cod:bulk') || isShareInvalid('bulk:cod') || !!codBulkBlocked()" (click)="fetchBulkStructures()">{{ isSubmitting('cod:bulk') ? 'Starting…' : 'Fetch structures' }}</button>
                    </div>
                    <p class="run-note bulk-hint" [class.invalid]="isShareInvalid('bulk:cod')">{{ isShareInvalid('bulk:cod') ? 'Enter a share from 1 to 100.' : codBulkHint() }}</p>
                    <details class="how-it-works">
                        <summary>How this works</summary>
                        <p>Searches COD by name for stored materials without a structure. Only an exact name match is imported and linked; if several entries share the name, the one with coordinates and the lowest COD ID is used. Materials with no exact match are skipped and listed under Jobs → Details. Many materials have no COD entry.</p>
                    </details>
                } @else {
                @if (codLinkMaterial(); as material) {
                    <p class="target-chip">Imports will be linked to <strong>{{ material.name }}</strong><button class="text-button" type="button" (click)="codLinkMaterial.set(null)">Clear</button></p>
                }
                <div class="query-row">
                    <label class="query-field query-select"><span>Search by</span><select [(ngModel)]="codQueryType" aria-label="COD query type"><option value="text">Text</option><option value="formula">Formula</option><option value="id">COD ID</option></select></label>
                    <label class="query-field"><span>Query</span><input id="acquire-cod-query" [(ngModel)]="codQuery" placeholder="Text, formula, or COD ID" (keyup.enter)="searchStructures()" /></label>
                    <button class="button primary" type="button" [disabled]="codSearching() || !codQuery.trim()" (click)="searchStructures()">{{ codSearching() ? 'Searching…' : 'Search' }}</button>
                </div>
                @if (codResults().length) {
                    <div class="data-table-frame data-table-frame--compact cod-results-frame">
                        <table class="data-table cod-results-table">
                            <caption class="sr-only">COD search results</caption>
                            <thead><tr><th class="select-column" scope="col"><input type="checkbox" [checked]="allCodSelected()" title="Select all results, up to 50" aria-label="Select all results, up to 50" (change)="toggleAllCod($event)" /></th><th scope="col">COD ID</th><th scope="col">Name</th><th scope="col">Formula</th><th scope="col">Space group</th><th scope="col">Reference</th></tr></thead>
                            <tbody>
                                @for (item of visibleCodResults(); track item.cod_id) {
                                    <tr>
                                        <td class="select-column"><input type="checkbox" [checked]="selectedCodIds().includes(item.cod_id)" [attr.aria-label]="'Select COD ' + item.cod_id" (change)="toggleCod(item.cod_id, $event)" /></td>
                                        <td><a [href]="item.source_url" target="_blank" rel="noopener noreferrer">{{ item.cod_id }}</a></td>
                                        <td class="truncate-cell" [title]="item.name || ''">{{ item.name || '—' }}</td>
                                        <td class="truncate-cell" [title]="item.formula || ''">{{ item.formula || '—' }}</td>
                                        <td class="truncate-cell" [title]="item.space_group || ''">{{ item.space_group || '—' }}</td>
                                        <td class="truncate-cell" [title]="item.doi || ''">{{ item.year ?? '' }}{{ item.doi ? ' · ' + item.doi : '' }}</td>
                                    </tr>
                                }
                            </tbody>
                        </table>
                    </div>
                    <adsmod-data-pagination
                        [page]="codPage()"
                        [pageSize]="codPageSize()"
                        [total]="codResults().length"
                        itemLabel="result"
                        [disabled]="codSearching()"
                        (pageChange)="setCodPage($event)"
                        (pageSizeChange)="setCodPageSize($event)"
                    />
                    @if (codResults().length > 1) { <p class="run-note result-note">Several structures match; one material can have many entries. Select the ones to import.</p> }
                    <div class="cod-import-bar">
                        @if (!codLinkMaterial()) {
                            <adsmod-entity-picker kind="material" label="Link imports to material (optional)" (selected)="codLinkMaterial.set($event)" />
                        }
                        <span class="cod-submit">
                            @if (selectedCodIds().length) {
                                <span>{{ selectedCodIds().length }} selected</span>
                                <button class="text-button" type="button" (click)="selectedCodIds.set([])">Clear</button>
                            }
                            <button class="button primary" type="button" [disabled]="!selectedCodIds().length || isSubmitting('cod:import')" (click)="importStructures()">{{ isSubmitting('cod:import') ? 'Starting…' : 'Import selected (' + selectedCodIds().length + ')' }}</button>
                        </span>
                    </div>
                } @else if (codSearchCompleted()) {
                    <p class="empty-inline" role="status">No matching COD records. Not every material has a published structure; try a narrower query.</p>
                }
                }
                <adsmod-acquire-activity source="cod" [sessionRunIds]="sessionRunIds()" />
            </section>
        </div>
        <adsmod-confirm-dialog
            [open]="overwriteCount() !== null"
            title="Overwrite local edits?"
            [message]="(overwriteCount() ?? 0) + ' NIST experiment(s) were edited locally. Fetching NIST experiments again replaces those edits with the source data. Continue?'"
            confirmLabel="Overwrite and fetch"
            (confirmed)="confirmOverwrite()"
            (closed)="overwriteCount.set(null)"
        />
    `,
    styleUrls: ['./sources-page.component.css', './acquire-view.component.css'],
})
export class AcquireViewComponent implements OnInit, AfterViewInit {
    private readonly route = inject(ActivatedRoute);
    private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
    private readonly destroyRef = inject(DestroyRef);
    private readonly jobs = inject(JobsMonitorService);
    private readonly targeted = viewChild<NistTargetedSearchComponent>('targeted');
    private destroyed = false;

    protected readonly collectionRows = NIST_COLLECTIONS;
    protected readonly modeOptions = MODE_OPTIONS;
    protected readonly collections = signal<NISTCollectionView[]>([]);
    protected readonly fractions = signal<Record<NISTCategory, number>>({
        guest: readFraction('guest'),
        host: readFraction('host'),
        experiments: readFraction('experiments'),
    });
    protected readonly error = signal<string | null>(null);
    private readonly pendingSubmissions = signal<ReadonlySet<string>>(new Set());

    protected readonly nistMode = signal<AcquireMode>(readMode('nist', 'bulk'));
    protected readonly pubchemMode = signal<AcquireMode>(readMode('pubchem', 'single'));
    protected readonly codMode = signal<AcquireMode>(readMode('cod', 'single'));
    /** Share inputs currently holding a value outside 1-100, keyed `nist:<category>` or `bulk:<card>`. */
    private readonly invalidShares = signal<ReadonlySet<string>>(new Set());
    protected readonly sessionRunIds = signal<ReadonlySet<string>>(new Set());
    protected readonly overview = signal<PublicDataOverviewResponse | null>(null);
    protected readonly bulkFractions = signal({ pubchem: 1, cod: 1 });
    protected readonly pubchemLooking = signal(false);
    protected readonly pubchemResults = signal<ChemicalLookupItem[]>([]);
    protected readonly pubchemMessage = signal<string | null>(null);

    protected pubchemQuery = '';
    protected readonly pubchemTarget = signal<{ id: number; name: string } | null>(null);

    protected codQuery = '';
    protected codQueryType: 'text' | 'formula' | 'id' = 'text';
    protected readonly codSearching = signal(false);
    protected readonly codSearchCompleted = signal(false);
    protected readonly codResults = signal<CODSearchResult[]>([]);
    protected readonly selectedCodIds = signal<string[]>([]);
    protected readonly codLinkMaterial = signal<PickedEntity | null>(null);
    /** Locally edited NIST experiments a pending experiments fetch would overwrite. */
    protected readonly overwriteCount = signal<number | null>(null);
    protected readonly codPage = signal(1);
    protected readonly codPageSize = signal(10);

    protected readonly availabilityKnown = computed(() =>
        this.collectionRows.every((row) => this.collections().find((item) => item.category === row.category)?.available_count != null),
    );
    protected readonly enrichBlocked = computed(() => {
        const stats = this.overview();
        if (!stats) return null;
        if (stats.species_total === 0) return 'No stored species yet; fetch Adsorbate species above.';
        return stats.species_enriched >= stats.species_total ? 'All stored species are enriched.' : null;
    });
    protected readonly conformersBlocked = computed(() => this.overview()?.species_enriched === 0);
    protected readonly enrichHint = computed(() => {
        const stats = this.overview();
        if (!stats) return '';
        const remaining = stats.species_total - stats.species_enriched;
        return this.enrichBlocked() ?? this.shareHint(this.bulkFractions().pubchem, remaining, 'species without PubChem data');
    });
    protected readonly codBulkBlocked = computed(() => {
        const stats = this.overview();
        if (!stats) return null;
        if (stats.materials_total === 0) return 'No stored materials yet; fetch Adsorbent materials above.';
        return stats.materials_with_structure >= stats.materials_total ? 'All stored materials have a structure.' : null;
    });
    protected readonly codBulkHint = computed(() => {
        const stats = this.overview();
        if (!stats) return '';
        const remaining = stats.materials_total - stats.materials_with_structure;
        return this.codBulkBlocked() ?? this.shareHint(this.bulkFractions().cod, remaining, 'materials without a structure');
    });

    protected readonly visibleCodResults = computed(() => {
        const start = (this.codPage() - 1) * this.codPageSize();
        return this.codResults().slice(start, start + this.codPageSize());
    });
    protected readonly allCodSelected = computed(
        () => this.codResults().length > 0 && this.selectedCodIds().length === Math.min(this.codResults().length, MAX_COD_SELECTION),
    );

    constructor() {
        this.destroyRef.onDestroy(() => {
            this.destroyed = true;
        });
        this.jobs.finished.pipe(takeUntilDestroyed()).subscribe((ended) => {
            this.rememberRuns(ended.flatMap((job) => (job.run ? [job.run.id] : [])));
            if (ended.some((job) => job.source === 'nist')) void this.refreshCollections();
            void this.refreshOverview();
        });
    }

    ngOnInit(): void {
        const params = this.route.snapshot.queryParamMap;
        const query = params.get('query');
        if (query) this.pubchemQuery = query;
        const adsorbateId = Number(params.get('adsorbate_id'));
        if (Number.isInteger(adsorbateId) && adsorbateId > 0) {
            this.pubchemTarget.set({ id: adsorbateId, name: params.get('adsorbate_name') ?? query ?? `chemical ${adsorbateId}` });
        }
        const formula = params.get('formula');
        const materialName = params.get('material_name');
        if (formula) {
            this.codQuery = formula;
            this.codQueryType = 'formula';
        } else if (materialName) {
            this.codQuery = materialName;
        }
        const materialId = Number(params.get('material_id'));
        if (Number.isInteger(materialId) && materialId > 0) {
            this.codLinkMaterial.set({ id: materialId, name: params.get('material_name') ?? `material ${materialId}` });
        }
        void this.refreshCollections();
        void this.refreshOverview();
        void this.jobs.refresh();
        // A deep link says "Search" or "Resolve", so it runs the search it prefilled.
        if (query) {
            this.pubchemMode.set('single');
            void this.lookupCompound();
        }
        if (this.codQuery) {
            this.codMode.set('single');
            void this.searchStructures();
        }
    }

    ngAfterViewInit(): void {
        const panel = this.route.snapshot.queryParamMap.get('panel');
        if (panel === 'runs') {
            this.jobs.popoverOpen.set(true);
        } else if (panel) {
            this.host.nativeElement.querySelector(`#acquire-${panel}`)?.scrollIntoView({ block: 'start' });
        }
    }

    protected collectionState(category: NISTCategory): NISTCollectionView {
        return this.collections().find((item) => item.category === category) ?? { category, local_count: 0, available_count: null, indexed_at: null };
    }

    protected setMode(card: AcquireCard, mode: AcquireMode): void {
        ({ nist: this.nistMode, pubchem: this.pubchemMode, cod: this.codMode })[card].set(mode);
        persistMode(card, mode);
    }

    protected toPercent(fraction: number): number {
        return Math.round(fraction * 10000) / 100;
    }

    /** Matches the backend, which rounds a share of an index up to whole records. */
    protected shareCount(fraction: number, total: number): number {
        return Math.ceil(fraction * total);
    }

    /** Names the batch only when it is a partial share; a full share needs no count restated. */
    private shareHint(fraction: number, remaining: number, noun: string): string {
        const count = this.shareCount(fraction, remaining);
        return count >= remaining ? '' : `Next ${count.toLocaleString()} of ${remaining.toLocaleString()} ${noun}.`;
    }

    protected isShareInvalid(key: string): boolean {
        return this.invalidShares().has(key);
    }

    private markShareValidity(key: string, valid: boolean): void {
        const next = new Set(this.invalidShares());
        if (valid) next.delete(key);
        else next.add(key);
        this.invalidShares.set(next);
    }

    protected setFraction(category: NISTCategory, value: number | string | null): void {
        const parsed = percentToFraction(value);
        this.markShareValidity(`nist:${category}`, parsed !== null);
        if (parsed === null) return;
        this.fractions.update((current) => ({ ...current, [category]: parsed }));
        persistFraction(category, parsed);
    }

    protected coveragePercent(state: NISTCollectionView): number {
        if (!state.available_count) return 0;
        return Math.min(100, Math.round((state.local_count / state.available_count) * 100));
    }

    /** Availability probes are not acquisitions: poll the single run directly and refresh coverage. */
    protected async checkAvailability(category: NISTCategory): Promise<void> {
        const key = `nist:index:${category}`;
        if (this.isSubmitting(key)) return;
        this.error.set(null);
        this.markSubmitting(key, true);
        try {
            const started = await startPublicDataSync({ source_key: 'nist', operation: 'nist_index', category });
            if (!started.data) {
                this.error.set(started.error);
                return;
            }
            let run = started.data;
            while (!this.destroyed && isActiveStatus(run.status)) {
                await new Promise((resolve) => window.setTimeout(resolve, AVAILABILITY_POLL_MS));
                const polled = await fetchPublicDataSyncRun(run.id);
                if (!polled.data) {
                    this.error.set(polled.error);
                    return;
                }
                run = polled.data;
            }
            if (run.status !== 'completed') this.error.set(run.error_detail || `Availability check ${run.status.replaceAll('_', ' ')}.`);
            await this.refreshCollections();
        } finally {
            this.markSubmitting(key, false);
        }
    }

    protected async checkAllAvailability(): Promise<void> {
        if (this.isSubmitting('nist:index:all')) return;
        this.markSubmitting('nist:index:all', true);
        try {
            for (const { category } of this.collectionRows) {
                if (this.isBusy('nist', category)) continue;
                await this.checkAvailability(category);
                if (this.error() || this.destroyed) break;
            }
        } finally {
            this.markSubmitting('nist:index:all', false);
        }
    }

    protected async fetchCollection(category: NISTCategory): Promise<void> {
        if (category === 'experiments') {
            const edited = await fetchNistEditedExperiments();
            if (edited.error) {
                this.error.set(edited.error);
                return;
            }
            if (edited.data?.count) {
                this.overwriteCount.set(edited.data.count);
                return;
            }
        }
        await this.startCollectionFetch(category, false);
    }

    protected confirmOverwrite(): void {
        this.overwriteCount.set(null);
        void this.startCollectionFetch('experiments', true);
    }

    private startCollectionFetch(category: NISTCategory, overwriteLocalEdits: boolean): Promise<boolean> {
        return this.queue(
            `nist:fetch:${category}`,
            () =>
                startPublicDataSync({
                    source_key: 'nist',
                    operation: 'nist_category',
                    category,
                    fraction: this.fractions()[category],
                    ...(overwriteLocalEdits ? { overwrite_local_edits: true } : {}),
                }),
        );
    }

    protected async lookupCompound(): Promise<void> {
        const query = this.pubchemQuery.trim();
        if (!query) return;
        this.pubchemLooking.set(true);
        this.pubchemResults.set([]);
        this.pubchemMessage.set(null);
        this.error.set(null);
        try {
            const result = await lookupChemicals(query);
            if (!result.data) {
                this.pubchemMessage.set(result.error ?? 'PubChem lookup failed.');
                return;
            }
            this.pubchemResults.set(result.data.items);
            if (!result.data.items.length) this.pubchemMessage.set('PubChem returned no matching compound.');
        } finally {
            this.pubchemLooking.set(false);
        }
    }

    protected async importCompound(cid: string): Promise<void> {
        if (await this.queue(`pubchem:resolve:${cid}`, () => resolvePubChem(cid, this.pubchemTarget()?.id))) this.pubchemTarget.set(null);
    }

    protected enrichChemicals(): Promise<boolean> {
        return this.queue('pubchem:enrich', () => startChemicalEnrichment(this.bulkFractions().pubchem));
    }

    protected fetchConformers(): Promise<boolean> {
        return this.queue('pubchem:conformers', () => startConformerEnrichment(this.bulkFractions().pubchem));
    }

    protected fetchBulkStructures(): Promise<boolean> {
        return this.queue('cod:bulk', () => startBulkStructures(this.bulkFractions().cod));
    }

    protected async fetchTargeted(identifiers: string[]): Promise<void> {
        if (await this.queue('nist:targeted', () => fetchNistIsotherms(identifiers))) this.targeted()?.reset();
    }

    protected setBulkFraction(key: 'pubchem' | 'cod', value: number | string | null): void {
        const parsed = percentToFraction(value);
        this.markShareValidity(`bulk:${key}`, parsed !== null);
        if (parsed === null) return;
        this.bulkFractions.update((current) => ({ ...current, [key]: parsed }));
    }

    protected percent(part: number, total: number): number {
        return total > 0 ? Math.min(100, Math.round((part / total) * 100)) : 0;
    }

    protected async searchStructures(): Promise<void> {
        const query = this.codQuery.trim();
        if (!query) return;
        this.codSearching.set(true);
        this.codSearchCompleted.set(false);
        this.codResults.set([]);
        this.selectedCodIds.set([]);
        this.codPage.set(1);
        this.error.set(null);
        try {
            const params = this.codQueryType === 'id' ? { cod_id: query } : this.codQueryType === 'formula' ? { formula: query } : { q: query };
            const result = await searchCOD(params);
            this.codResults.set(result.data?.items ?? []);
            this.codSearchCompleted.set(result.data !== null);
            this.error.set(result.error);
        } finally {
            this.codSearching.set(false);
        }
    }

    protected setCodPage(page: number): void {
        const max = Math.max(1, Math.ceil(this.codResults().length / this.codPageSize()));
        this.codPage.set(Math.min(Math.max(1, page), max));
    }

    protected setCodPageSize(size: number): void {
        if (size === this.codPageSize()) return;
        this.codPageSize.set(size);
        this.codPage.set(1);
    }

    protected toggleCod(codId: string, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        const selected = this.selectedCodIds();
        if (!checked) {
            this.selectedCodIds.set(selected.filter((item) => item !== codId));
        } else if (selected.length < MAX_COD_SELECTION) {
            this.selectedCodIds.set([...selected, codId]);
        } else {
            (event.target as HTMLInputElement).checked = false;
            this.error.set(`Select at most ${MAX_COD_SELECTION} COD records per import.`);
        }
    }

    protected toggleAllCod(event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.selectedCodIds.set(checked ? this.codResults().slice(0, MAX_COD_SELECTION).map((item) => item.cod_id) : []);
    }

    protected async importStructures(): Promise<void> {
        const selected = this.selectedCodIds();
        if (!selected.length) return;
        if (await this.queue('cod:import', () => importCOD(selected, this.codLinkMaterial()?.id))) this.selectedCodIds.set([]);
    }

    protected isBusy(source: string, key: string): boolean {
        return this.jobs.activeRuns().some(
            (run) => run.source_key === source && (run.parameters['category'] === key || run.operation === key),
        );
    }

    protected isSubmitting(key: string): boolean {
        return this.pendingSubmissions().has(key);
    }

    protected readonly formatDate = formatDate;

    private async refreshCollections(): Promise<void> {
        const result = await fetchNistCollections();
        if (this.destroyed) return;
        if (result.data) this.collections.set(result.data.collections);
        else this.error.set(result.error);
    }

    private async refreshOverview(): Promise<void> {
        const result = await fetchPublicOverview();
        if (!this.destroyed && result.data) this.overview.set(result.data);
    }

    /** Resolves true once the backend accepted the run, so callers keep user input on failure. */
    private async queue(
        key: string,
        request: () => Promise<ApiResult<PublicDataSyncRun>>,
    ): Promise<boolean> {
        if (this.isSubmitting(key)) return false;
        this.error.set(null);
        this.markSubmitting(key, true);
        try {
            const result = await request();
            if (!result.data) {
                this.error.set(result.error);
                return false;
            }
            this.rememberRuns([result.data.id]);
            void this.jobs.refresh();
            if (!isActiveStatus(result.data.status)) {
                if (result.data.source_key === 'nist') void this.refreshCollections();
                void this.refreshOverview();
            }
            return true;
        } finally {
            this.markSubmitting(key, false);
        }
    }

    private rememberRuns(ids: string[]): void {
        if (ids.length) this.sessionRunIds.update((current) => new Set([...current, ...ids]));
    }

    private markSubmitting(key: string, active: boolean): void {
        const next = new Set(this.pendingSubmissions());
        if (active) next.add(key);
        else next.delete(key);
        this.pendingSubmissions.set(next);
    }
}
