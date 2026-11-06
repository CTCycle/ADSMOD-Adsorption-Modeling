// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { AfterViewInit, Component, DestroyRef, ElementRef, OnInit, computed, inject, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import type {
    ChemicalLookupItem,
    CODSearchResult,
    NISTCategory,
    NISTCollectionView,
    PublicDataOverviewResponse,
    PublicDataSyncFailure,
    PublicDataSyncRun,
} from '../../models/public-data.model';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';
import {
    cancelPublicDataSyncRun,
    enrichChemicals as startChemicalEnrichment,
    fetchBulkStructures as startBulkStructures,
    fetchNistCollections,
    fetchNistEditedExperiments,
    fetchNistIsotherms,
    fetchPublicDataSyncFailures,
    fetchPublicDataSyncRun,
    fetchPublicDataSyncRuns,
    fetchPublicOverview,
    importCOD,
    lookupChemicals,
    resumePublicDataSyncRun,
    resolvePubChem,
    searchCOD,
    startPublicDataSync,
    type ApiResult,
} from '../../services/public-data.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import { EntityPickerComponent, type PickedEntity } from '../../shared/components/entity-picker/entity-picker.component';
import { NistTargetedSearchComponent } from './nist-targeted-search.component';

interface ImportedLink {
    label: string;
    path: string[];
    queryParams: Record<string, string | number>;
}

const COLLECTIONS: { category: NISTCategory; label: string }[] = [
    { category: 'guest', label: 'Adsorbate species' },
    { category: 'host', label: 'Adsorbent materials' },
    { category: 'experiments', label: 'Adsorption isotherms' },
];
const ACTIVE_STATUSES = new Set(['queued', 'running']);
const POLL_INTERVAL_MS = 1500;
const AVAILABILITY_POLL_MS = 800;
const POLL_RETRY_MIN_MS = 3000;
const POLL_RETRY_MAX_MS = 15000;
const MAX_COD_SELECTION = 50;
const MAX_ACTIVE_ROWS = 25;

const readFraction = (category: NISTCategory): number => {
    try {
        const value = Number.parseFloat(window.localStorage.getItem(`adsmod.nist.fraction.${category}`) ?? '');
        return Number.isFinite(value) && value > 0 && value <= 1 ? value : 0.1;
    } catch {
        return 0.1;
    }
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
    imports: [FormsModule, RouterLink, EntityPickerComponent, DataPaginationComponent, ConfirmDialogComponent, NistTargetedSearchComponent],
    template: `
        <div class="acquisition-toolbar">
            <button class="button secondary" type="button" data-acquisitions-trigger aria-haspopup="dialog"
                aria-controls="acquire-runs" [attr.aria-expanded]="runsOpen()" (click)="openRuns()">
                Acquisitions <span aria-live="polite">({{ activeRuns().length }} active)</span>
            </button>
            @if (pollError()) { <span class="run-note" role="status">Live updates paused; retrying automatically.</span> }
        </div>
        @if (error()) {
            <div class="public-alert error" role="alert"><strong>Request failed.</strong><span>{{ error() }}</span><button type="button" class="button secondary" (click)="error.set(null)">Dismiss</button></div>
        }

        <section class="public-section source-panel" id="acquire-nist">
            <div class="section-heading">
                <div>
                    <h4>Adsorption collections</h4>
                    <p>{{ nistMode() === 'bulk' ? 'Fetch NIST records in bounded runs; fetch again to continue.' : 'Search the NIST index and fetch only the isotherms you pick.' }}</p>
                </div>
                <div class="mode-switch" role="group" aria-label="NIST acquisition mode">
                    <button type="button" [attr.aria-pressed]="nistMode() === 'bulk'" (click)="nistMode.set('bulk')">Bulk</button>
                    <button type="button" [attr.aria-pressed]="nistMode() === 'targeted'" (click)="nistMode.set('targeted')">Targeted</button>
                </div>
            </div>
            @if (nistMode() === 'targeted') {
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
                                    <label class="collection-fraction">
                                        <input
                                            type="number"
                                            min="0.01"
                                            max="1"
                                            step="0.05"
                                            [ngModel]="fractions()[collection.category]"
                                            (ngModelChange)="setFraction(collection.category, $event)"
                                            [attr.aria-label]="'Share of the ' + collection.label + ' index to fetch'"
                                        />
                                    </label>
                                </td>
                                <td>
                                    <div class="table-actions">
                                        <button class="table-action" type="button" [disabled]="busy || isSubmitting('nist:index:' + collection.category)" [attr.aria-busy]="isSubmitting('nist:index:' + collection.category)" (click)="checkAvailability(collection.category)">{{ isSubmitting('nist:index:' + collection.category) ? 'Checking…' : 'Check availability' }}</button>
                                        <button class="button primary" type="button" [disabled]="busy || isSubmitting('nist:fetch:' + collection.category)" (click)="fetchCollection(collection.category)">{{ busy ? 'Running…' : 'Fetch' }}</button>
                                    </div>
                                </td>
                            </tr>
                        }
                    </tbody>
                </table>
            </div>
            }
        </section>

        <div class="acquire-grid">
            <section class="public-section source-panel" id="acquire-pubchem">
                <div class="section-heading">
                    <div>
                        <h4>Chemical identity and properties</h4>
                        <p>{{ pubchemMode() === 'single' ? 'Look up one compound, then import it.' : 'Enrich stored species in bulk.' }}</p>
                    </div>
                    <div class="mode-switch" role="group" aria-label="Chemical acquisition mode">
                        <button type="button" [attr.aria-pressed]="pubchemMode() === 'single'" (click)="pubchemMode.set('single')">Single</button>
                        <button type="button" [attr.aria-pressed]="pubchemMode() === 'bulk'" (click)="pubchemMode.set('bulk')">Bulk</button>
                    </div>
                </div>
                <p class="acq-stats">
                    @if (overview(); as stats) {
                        <span><strong>{{ stats.species_enriched.toLocaleString() }} / {{ stats.species_total.toLocaleString() }}</strong> stored species enriched from PubChem</span>
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
                    <div class="query-row">
                        <label class="query-field query-select"><span>Share to enrich</span><input type="number" min="0.01" max="1" step="0.05" [ngModel]="bulkFractions().pubchem" (ngModelChange)="setBulkFraction('pubchem', $event)" aria-label="Share of unenriched species to enrich" /></label>
                        <button class="button primary" type="button" [disabled]="isBusy('pubchem', 'pubchem_enrich') || isSubmitting('pubchem:enrich')" (click)="enrichChemicals()">{{ isSubmitting('pubchem:enrich') ? 'Starting…' : 'Enrich stored species' }}</button>
                    </div>
                    <p class="run-note result-note">Looks up stored species that have no PubChem identity yet{{ overview() ? ' (' + (overview()!.species_total - overview()!.species_enriched).toLocaleString() + ' remaining)' : '' }}. Species already enriched or confirmed unresolvable are skipped.</p>
                }
            </section>

            <section class="public-section source-panel" id="acquire-cod">
                <div class="section-heading">
                    <div>
                        <h4>Crystal structures</h4>
                        <p>{{ codMode() === 'single' ? 'Search COD, inspect matches, and import selected structures.' : 'Look up structures for stored materials in bulk.' }}</p>
                    </div>
                    <div class="mode-switch" role="group" aria-label="Structure acquisition mode">
                        <button type="button" [attr.aria-pressed]="codMode() === 'single'" (click)="codMode.set('single')">Single</button>
                        <button type="button" [attr.aria-pressed]="codMode() === 'bulk'" (click)="codMode.set('bulk')">Bulk</button>
                    </div>
                </div>
                <p class="acq-stats">
                    @if (overview(); as stats) {
                        <span><strong>{{ stats.materials_with_structure.toLocaleString() }} / {{ stats.materials_total.toLocaleString() }}</strong> stored materials have a structure · {{ stats.structures.toLocaleString() }} structures from COD</span>
                        @if (stats.materials_total > 0) {
                            <progress max="100" [value]="percent(stats.materials_with_structure, stats.materials_total)" aria-label="Materials with a structure"></progress>
                            <span>{{ percent(stats.materials_with_structure, stats.materials_total) }}%</span>
                        }
                    } @else {
                        <span>Coverage unavailable</span>
                    }
                </p>
                @if (codMode() === 'bulk') {
                    <div class="query-row">
                        <label class="query-field query-select"><span>Share to fetch</span><input type="number" min="0.01" max="1" step="0.05" [ngModel]="bulkFractions().cod" (ngModelChange)="setBulkFraction('cod', $event)" aria-label="Share of materials without a structure to look up" /></label>
                        <button class="button primary" type="button" [disabled]="isBusy('cod', 'cod_import') || isSubmitting('cod:bulk')" (click)="fetchBulkStructures()">{{ isSubmitting('cod:bulk') ? 'Starting…' : 'Fetch structures' }}</button>
                    </div>
                    <p class="run-note result-note">Searches COD by name for stored materials without a structure. Only an exact, unique name match is imported and linked; materials with no match or several candidates are skipped and listed under Acquisitions → Details. Many materials have no COD entry.</p>
                } @else {
                <div class="query-row">
                    <label class="query-field query-select"><span>Search by</span><select [(ngModel)]="codQueryType" aria-label="COD query type"><option value="text">Text</option><option value="formula">Formula</option><option value="id">COD ID</option></select></label>
                    <label class="query-field"><span>Query</span><input id="acquire-cod-query" [(ngModel)]="codQuery" placeholder="Text, formula, or COD ID" (keyup.enter)="searchStructures()" /></label>
                    <button class="button primary" type="button" [disabled]="codSearching() || !codQuery.trim()" (click)="searchStructures()">{{ codSearching() ? 'Searching…' : 'Search' }}</button>
                </div>
                @if (codResults().length) {
                    <div class="data-table-frame data-table-frame--compact cod-results-frame">
                        <table class="data-table cod-results-table">
                            <caption class="sr-only">COD search results</caption>
                            <thead><tr><th class="select-column" scope="col"><input type="checkbox" [checked]="allCodSelected()" aria-label="Select all results" (change)="toggleAllCod($event)" /></th><th scope="col">COD ID</th><th scope="col">Name</th><th scope="col">Formula</th><th scope="col">Space group</th><th scope="col">Reference</th></tr></thead>
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
                        @if (codLinkMaterial(); as material) {
                            <p class="target-chip">Link imports to <strong>{{ material.name }}</strong><button class="text-button" type="button" (click)="codLinkMaterial.set(null)">Change</button></p>
                        } @else {
                            <adsmod-entity-picker kind="material" label="Link imports to material (optional)" (selected)="codLinkMaterial.set($event)" />
                        }
                        <button class="button primary" type="button" [disabled]="!selectedCodIds().length || isSubmitting('cod:import')" (click)="importStructures()">{{ isSubmitting('cod:import') ? 'Starting…' : 'Import selected (' + selectedCodIds().length + ')' }}</button>
                    </div>
                } @else if (codSearchCompleted()) {
                    <p class="empty-inline" role="status">No matching COD records. Not every material has a published structure; try a narrower query.</p>
                }
                }
            </section>
        </div>
        <dialog #runsDialog class="acquisitions-dialog" id="acquire-runs" aria-labelledby="acquisitions-title"
            (cancel)="$event.preventDefault(); closeRuns()" (click)="dismissBackdrop($event)" (keydown)="trapRunFocus($event)">
            <div class="acquisitions-dialog-heading">
                <div><h2 id="acquisitions-title">Acquisitions</h2><p>Progress and results of fetch runs.</p></div>
                <div class="table-actions">
                    <button class="button secondary" type="button" (click)="refreshAll()">Refresh</button>
                    <button class="button secondary" type="button" autofocus (click)="closeRuns()" aria-label="Close acquisitions">Close</button>
                </div>
            </div>
            <div class="acquisitions-dialog-body">
                @if (runError()) {
                    <div class="public-alert error" role="alert"><span>{{ runError() }}</span><button class="button secondary" type="button" (click)="runError.set(null)">Dismiss</button></div>
                }
                @if (pollError()) {
                    <p class="run-poll-note" role="status">Live updates paused; retrying automatically. {{ pollError() }}</p>
                }
                @if (activeRuns().length) {
                    <div class="active-runs">
                        <div class="run-list-head">
                            <strong>Active</strong>
                            <span>{{ activeRuns().length }} {{ activeRuns().length === 1 ? 'run' : 'runs' }}</span>
                        </div>
                        <div class="active-run-list">
                            @for (run of visibleActiveRuns(); track run.id) {
                                <article class="active-run-row">
                                    <div class="active-run-main">
                                        <strong [title]="operationLabel(run)">{{ operationLabel(run) }}</strong>
                                        <span>{{ run.source_key }} · {{ formatDate(run.created_at) }}</span>
                                    </div>
                                    <span class="run-status" [attr.data-status]="run.status">{{ statusLabel(run) }}</span>
                                    <div class="active-run-progress">
                                        <progress max="100" [value]="run.progress" [attr.aria-label]="operationLabel(run) + ' progress'"></progress>
                                        <span>{{ formatPercent(run.progress) }}</span>
                                    </div>
                                    <span class="active-run-counts">{{ run.imported_count }} saved · {{ run.skipped_count }} skipped · {{ run.failed_count }} failed</span>
                                    <div class="table-actions">
                                        <button class="table-action" type="button" (click)="showDetails(run.id)">Details</button>
                                        @if (run.cancel_requested) {
                                            <span class="run-pending">Cancelling…</span>
                                        } @else {
                                            <button class="table-action danger" type="button" (click)="cancelRun(run.id)">Cancel</button>
                                        }
                                    </div>
                                </article>
                            }
                        </div>
                        @if (activeRuns().length > visibleActiveRuns().length) {
                            <p class="run-note">Showing the first {{ visibleActiveRuns().length }} of {{ activeRuns().length }} active runs.</p>
                        }
                    </div>
                }
                <div class="run-list-head history-head">
                    <strong>Recent acquisitions</strong>
                    <span>{{ historyTotal() }} {{ historyTotal() === 1 ? 'run' : 'runs' }}</span>
                </div>
                <div class="data-table-frame data-table-frame--history acquisition-history-frame">
                    <table class="data-table acquisition-history-table">
                        <caption class="sr-only">Recent acquisitions</caption>
                        <thead>
                            <tr>
                                <th class="col-source" scope="col">Source</th>
                                <th class="col-operation" scope="col">Operation</th>
                                <th class="col-status" scope="col">Status</th>
                                <th class="col-progress" scope="col">Progress</th>
                                <th class="col-count num" scope="col">Saved</th>
                                <th class="col-count num" scope="col">Skipped</th>
                                <th class="col-count num" scope="col">Failed</th>
                                <th class="col-date" scope="col">Created</th>
                                <th class="col-date" scope="col">Finished</th>
                                <th class="col-actions" scope="col"></th>
                            </tr>
                        </thead>
                        <tbody>
                            @for (run of historyRuns(); track run.id) {
                                <tr [class.selected]="detailsRunId() === run.id">
                                    <td><span class="source-key">{{ run.source_key }}</span></td>
                                    <td class="truncate-cell" [title]="operationLabel(run)">{{ operationLabel(run) }}</td>
                                    <td><span class="run-status" [attr.data-status]="run.status">{{ statusLabel(run) }}</span></td>
                                    <td>
                                        <div class="table-progress">
                                            <progress max="100" [value]="run.progress" [attr.aria-label]="operationLabel(run) + ' progress'"></progress>
                                            <span>{{ formatPercent(run.progress) }}</span>
                                        </div>
                                    </td>
                                    <td class="num">{{ run.imported_count }}</td>
                                    <td class="num">{{ run.skipped_count }}</td>
                                    <td class="num">{{ run.failed_count }}</td>
                                    <td>{{ formatDate(run.created_at) }}</td>
                                    <td>{{ formatDate(run.finished_at) }}</td>
                                    <td class="col-actions">
                                        <div class="table-actions">
                                            @if (importedLink(run); as link) {
                                                <a class="table-action" [routerLink]="link.path" [queryParams]="link.queryParams">{{ link.label }}</a>
                                            }
                                            <button class="table-action" type="button" (click)="showDetails(run.id)">Details</button>
                                            @if (run.status === 'interrupted') {
                                                <button class="table-action" type="button" (click)="resumeRun(run.id)">Resume</button>
                                            }
                                        </div>
                                    </td>
                                </tr>
                            } @empty {
                                <tr><td colspan="10" class="data-table-empty">No acquisitions yet. Close this dialog to start a NIST collection fetch, PubChem lookup, or COD search.</td></tr>
                            }
                        </tbody>
                    </table>
                </div>
                <adsmod-data-pagination
                    [page]="historyPage()"
                    [pageSize]="historyPageSize()"
                    [total]="historyTotal()"
                    itemLabel="acquisition"
                    [disabled]="historyLoading()"
                    (pageChange)="changeHistoryPage($event)"
                    (pageSizeChange)="changeHistoryPageSize($event)"
                />
                @if (detailsRun(); as run) {
                    <div class="run-details-panel" role="region" aria-label="Acquisition details">
                        <div class="run-details-header">
                            <div>
                                <p class="eyebrow">{{ run.source_key }} · {{ statusLabel(run) }}</p>
                                <h5>{{ operationLabel(run) }}</h5>
                            </div>
                            <button class="detail-close" type="button" aria-label="Close details" (click)="closeDetails()">×</button>
                        </div>
                        <div class="run-details-grid">
                            <div><span>Progress</span><strong>{{ formatPercent(run.progress) }}</strong></div>
                            <div><span>Saved</span><strong>{{ run.imported_count }}</strong></div>
                            <div><span>Skipped</span><strong>{{ run.skipped_count }}</strong></div>
                            <div><span>Failed</span><strong>{{ run.failed_count }}</strong></div>
                            <div><span>Processed</span><strong>{{ run.processed_count }}</strong></div>
                            <div><span>Created</span><strong>{{ formatDate(run.created_at) }}</strong></div>
                            <div><span>Started</span><strong>{{ formatDate(run.started_at) }}</strong></div>
                            <div><span>Finished</span><strong>{{ formatDate(run.finished_at) }}</strong></div>
                        </div>
                        @if (pendingCount(run); as pending) {
                            <p class="run-note">{{ pending }} records remain; run the operation again to continue.</p>
                        }
                        @if (run.error_detail) {
                            <p class="run-details-error" role="alert">{{ run.error_detail }}</p>
                        }
                        @if (detailsFailuresLoading()) {
                            <p class="run-note" role="status">Loading failed records…</p>
                        } @else if (detailsFailures().length) {
                            <div class="run-failures">
                                <div class="run-list-head">
                                    <strong>Failed records</strong>
                                    <span>{{ detailsFailuresTotal() }} total</span>
                                </div>
                                <ul class="sync-failures">
                                    @for (failure of detailsFailures(); track failure.id) {
                                        <li><strong>{{ failure.external_id || 'Record' }}</strong><span>{{ failure.reason }}</span></li>
                                    }
                                </ul>
                                @if (detailsFailuresTotal() > detailsFailures().length) {
                                    <p class="run-note">Showing the first {{ detailsFailures().length }} of {{ detailsFailuresTotal() }} failed records.</p>
                                }
                            </div>
                        } @else if (run.failed_count) {
                            <p class="run-note">No record-level failure details were returned for this run.</p>
                        }
                        <div class="run-details-actions">
                            @if (importedLink(run); as link) {
                                <a class="button secondary" [routerLink]="link.path" [queryParams]="link.queryParams">{{ link.label }}</a>
                            }
                            @if (isActive(run)) {
                                <button class="button secondary" type="button" [disabled]="run.cancel_requested" (click)="cancelRun(run.id)">{{ run.cancel_requested ? 'Cancelling…' : 'Cancel' }}</button>
                            }
                            @if (run.status === 'interrupted') {
                                <button class="button primary" type="button" (click)="resumeRun(run.id)">Resume</button>
                            }
                            <button class="button quiet" type="button" (click)="closeDetails()">Close</button>
                        </div>
                    </div>
                }
            </div>
        </dialog>

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
    private readonly runsDialog = viewChild<ElementRef<HTMLDialogElement>>('runsDialog');
    private readonly targeted = viewChild<NistTargetedSearchComponent>('targeted');
    protected readonly runsOpen = signal(false);
    protected readonly runError = signal<string | null>(null);
    private previousOverflow: string | null = null;
    private mainScrollLock: { element: HTMLElement; overflow: string } | null = null;
    private pollTimer: number | null = null;
    private pollDelay = 0;
    private destroyed = false;

    protected readonly collectionRows = COLLECTIONS;
    protected readonly collections = signal<NISTCollectionView[]>([]);
    protected readonly fractions = signal<Record<NISTCategory, number>>({
        guest: readFraction('guest'),
        host: readFraction('host'),
        experiments: readFraction('experiments'),
    });
    protected readonly activeRuns = signal<PublicDataSyncRun[]>([]);
    protected readonly historyRuns = signal<PublicDataSyncRun[]>([]);
    protected readonly historyTotal = signal(0);
    protected readonly historyPage = signal(1);
    protected readonly historyPageSize = signal(10);
    protected readonly historyLoading = signal(false);
    protected readonly detailsRunId = signal<string | null>(null);
    protected readonly detailsRun = signal<PublicDataSyncRun | null>(null);
    protected readonly detailsFailures = signal<PublicDataSyncFailure[]>([]);
    protected readonly detailsFailuresTotal = signal(0);
    protected readonly detailsFailuresLoading = signal(false);
    protected readonly pollError = signal<string | null>(null);
    protected readonly error = signal<string | null>(null);
    private readonly pendingSubmissions = signal<ReadonlySet<string>>(new Set());

    protected readonly nistMode = signal<'bulk' | 'targeted'>('bulk');
    protected readonly pubchemMode = signal<'single' | 'bulk'>('single');
    protected readonly codMode = signal<'single' | 'bulk'>('single');
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

    protected readonly visibleActiveRuns = computed(() => this.activeRuns().slice(0, MAX_ACTIVE_ROWS));
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
            this.restorePageScroll();
            if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
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
        if (formula) {
            this.codQuery = formula;
            this.codQueryType = 'formula';
        }
        const materialId = Number(params.get('material_id'));
        if (Number.isInteger(materialId) && materialId > 0) {
            this.codLinkMaterial.set({ id: materialId, name: params.get('material_name') ?? `material ${materialId}` });
        }
        void this.refreshCollections();
        void this.refreshOverview();
        void this.refreshRuns();
    }

    ngAfterViewInit(): void {
        const panel = this.route.snapshot.queryParamMap.get('panel');
        if (panel === 'runs') {
            this.openRuns();
        } else if (panel) {
            this.host.nativeElement.querySelector(`#acquire-${panel}`)?.scrollIntoView({ block: 'start' });
        }
    }

    protected openRuns(): void {
        const dialog = this.runsDialog()?.nativeElement;
        if (!dialog || dialog.open) return;
        this.previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        const main = this.host.nativeElement.closest('main');
        if (main) {
            this.mainScrollLock = { element: main, overflow: main.style.overflow };
            main.style.overflow = 'hidden';
        }
        dialog.showModal();
        this.runsOpen.set(true);
    }

    protected closeRuns(): void {
        const dialog = this.runsDialog()?.nativeElement;
        if (dialog?.open) dialog.close();
        this.runsOpen.set(false);
        this.closeDetails();
        this.runError.set(null);
        this.restorePageScroll();
        this.host.nativeElement.querySelector<HTMLButtonElement>('[data-acquisitions-trigger]')?.focus();
    }

    protected trapRunFocus(event: KeyboardEvent): void {
        if (event.key !== 'Tab') return;
        const dialog = this.runsDialog()?.nativeElement;
        if (!dialog) return;
        const controls = Array.from(dialog.querySelectorAll<HTMLElement>(
            'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        )).filter((element) => element.getClientRects().length > 0 && element.tabIndex >= 0);
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first?.focus();
        }
    }

    protected dismissBackdrop(event: MouseEvent): void {
        const dialog = this.runsDialog()?.nativeElement;
        if (!dialog || event.target !== dialog) return;
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) this.closeRuns();
    }

    private restorePageScroll(): void {
        if (this.mainScrollLock) {
            this.mainScrollLock.element.style.overflow = this.mainScrollLock.overflow;
            this.mainScrollLock = null;
        }
        if (this.previousOverflow === null) return;
        document.body.style.overflow = this.previousOverflow;
        this.previousOverflow = null;
    }

    protected collectionState(category: NISTCategory): NISTCollectionView {
        return this.collections().find((item) => item.category === category) ?? { category, local_count: 0, available_count: null, indexed_at: null };
    }

    protected setFraction(category: NISTCategory, value: number | string): void {
        const parsed = Number(value);
        if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) return;
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
            while (!this.destroyed && this.isActive(run)) {
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

    private startCollectionFetch(category: NISTCategory, overwriteLocalEdits: boolean): Promise<void> {
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
        await this.queue(`pubchem:resolve:${cid}`, () => resolvePubChem(cid, this.pubchemTarget()?.id));
        this.pubchemTarget.set(null);
    }

    protected enrichChemicals(): Promise<void> {
        return this.queue('pubchem:enrich', () => startChemicalEnrichment(this.bulkFractions().pubchem));
    }

    protected fetchBulkStructures(): Promise<void> {
        return this.queue('cod:bulk', () => startBulkStructures(this.bulkFractions().cod));
    }

    protected async fetchTargeted(identifiers: string[]): Promise<void> {
        await this.queue('nist:targeted', () => fetchNistIsotherms(identifiers));
        this.targeted()?.reset();
    }

    protected setBulkFraction(key: 'pubchem' | 'cod', value: number | string): void {
        const parsed = Number(value);
        if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) return;
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
        await this.queue('cod:import', () => importCOD(selected, this.codLinkMaterial()?.id));
        this.selectedCodIds.set([]);
    }

    protected refreshAll(): void {
        void this.refreshCollections();
        void this.refreshOverview();
        void this.refreshRuns();
    }

    protected async changeHistoryPage(page: number): Promise<void> {
        if (page === this.historyPage()) return;
        this.historyPage.set(page);
        await this.refreshHistoryOnly();
    }

    protected async changeHistoryPageSize(size: number): Promise<void> {
        if (size === this.historyPageSize()) return;
        this.historyPageSize.set(size);
        this.historyPage.set(1);
        await this.refreshHistoryOnly();
    }

    protected async cancelRun(runId: string): Promise<void> {
        this.runError.set(null);
        const result = await cancelPublicDataSyncRun(runId);
        if (result.error) this.runError.set(result.error);
        if (result.data) this.upsertRun(result.data);
        await this.refreshRuns();
    }

    protected async resumeRun(runId: string): Promise<void> {
        this.runError.set(null);
        const result = await resumePublicDataSyncRun(runId);
        if (!result.data) {
            this.runError.set(result.error);
            return;
        }
        this.upsertRun(result.data);
        this.schedulePoll(POLL_INTERVAL_MS);
        await this.refreshRuns();
    }

    protected async showDetails(runId: string): Promise<void> {
        this.runError.set(null);
        this.detailsRunId.set(runId);
        this.detailsRun.set(
            [...this.activeRuns(), ...this.historyRuns()].find((run) => run.id === runId) ?? null,
        );
        this.detailsFailures.set([]);
        this.detailsFailuresTotal.set(0);
        this.detailsFailuresLoading.set(true);
        const [runResult, failureResult] = await Promise.all([
            fetchPublicDataSyncRun(runId),
            fetchPublicDataSyncFailures(runId),
        ]);
        if (this.destroyed || this.detailsRunId() !== runId) return;
        if (runResult.data) this.detailsRun.set(runResult.data);
        else if (runResult.error) this.runError.set(runResult.error);
        this.detailsFailuresLoading.set(false);
        if (failureResult.data) {
            this.detailsFailures.set(failureResult.data.failures);
            this.detailsFailuresTotal.set(failureResult.data.total);
        } else if (failureResult.error) {
            this.runError.set(failureResult.error);
        }
    }

    protected closeDetails(): void {
        this.detailsRunId.set(null);
        this.detailsRun.set(null);
        this.detailsFailures.set([]);
        this.detailsFailuresTotal.set(0);
        this.detailsFailuresLoading.set(false);
    }

    protected isActive(run: PublicDataSyncRun): boolean {
        return ACTIVE_STATUSES.has(run.status);
    }

    protected isBusy(source: string, key: string): boolean {
        return this.activeRuns().some(
            (run) => this.isActive(run) && run.source_key === source && (run.parameters['category'] === key || run.operation === key),
        );
    }

    protected isSubmitting(key: string): boolean {
        return this.pendingSubmissions().has(key);
    }

    protected statusLabel(run: PublicDataSyncRun): string {
        if (run.cancel_requested && run.status === 'running') return 'cancelling';
        return run.status.replaceAll('_', ' ');
    }

    protected operationLabel(run: PublicDataSyncRun): string {
        const category = COLLECTIONS.find((item) => item.category === run.parameters['category'])?.label ?? 'collection';
        switch (run.operation) {
            case 'nist_category': {
                const picked = run.parameters['identifiers'] as string[] | undefined;
                return picked ? `NIST targeted fetch · ${picked.length} isotherms` : `NIST fetch · ${category}`;
            }
            case 'pubchem_resolve': return `PubChem · ${String(run.parameters['query'] ?? 'compound')}`;
            case 'pubchem_enrich': return 'PubChem enrichment of stored chemicals';
            case 'cod_import': {
                if ('fraction' in run.parameters) return 'COD structures for stored materials';
                const ids = (run.parameters['cod_ids'] as string[] | undefined) ?? [String(run.parameters['cod_id'] ?? '')];
                return ids.length === 1 ? `COD import · ${ids[0]}` : `COD import · ${ids.length} structures`;
            }
            default: return 'Provider health check';
        }
    }

    protected importedLink(run: PublicDataSyncRun): ImportedLink | null {
        if (!['completed', 'partially_completed'].includes(run.status)) return null;
        const entityId = Number(run.checkpoint['entity_id']);
        const entityIds = run.checkpoint['entity_ids'];
        const single = Number.isInteger(entityId) ? entityId : Array.isArray(entityIds) && entityIds.length === 1 ? Number(entityIds[0]) : null;
        const category = run.parameters['category'];
        switch (run.operation) {
            case 'nist_category':
                if (!run.imported_count && !run.skipped_count) return null;
                if (category === 'experiments') return { label: 'View datasets', path: ['/datasets'], queryParams: {} };
                return { label: category === 'guest' ? 'View species' : 'View materials', path: ['/datasets'], queryParams: {} };
            case 'pubchem_resolve':
            case 'pubchem_enrich':
                return run.imported_count || run.skipped_count ? { label: 'View species', path: ['/datasets'], queryParams: {} } : null;
            case 'cod_import':
                if (single !== null) return { label: 'Open structure', path: ['/sources', 'structures'], queryParams: { id: single } };
                return run.imported_count || run.skipped_count ? { label: 'View structures', path: ['/sources', 'structures'], queryParams: { source: 'cod' } } : null;
            default:
                return null;
        }
    }

    protected pendingCount(run: PublicDataSyncRun): number {
        if (this.isActive(run) || (run.operation === 'cod_import' && !('fraction' in run.parameters))) return 0;
        return Number(run.checkpoint['pending_count'] ?? 0);
    }

    protected formatPercent(value: number): string {
        return `${Math.round(value)}%`;
    }

    protected formatDate(value: string | null): string {
        if (!value) return '—';
        const date = new Date(value);
        return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
    }

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

    private async refreshRuns(): Promise<void> {
        if (this.pollTimer !== null) {
            window.clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
        const previousActive = this.activeRuns();
        const activeRequest = fetchPublicDataSyncRuns({ scope: 'active', page: 1, pageSize: 100 });
        const [activeResult, historyOk] = await Promise.all([
            activeRequest,
            this.refreshHistoryOnly(),
        ]);
        if (this.destroyed) return;
        let failed = !historyOk;
        if (activeResult.data) {
            this.activeRuns.set(activeResult.data.runs);
        } else {
            failed = true;
        }
        this.syncDetailsRun();
        if (failed) {
            this.pollError.set(activeResult.error || 'Live updates are temporarily unavailable.');
            this.schedulePoll(this.nextPollDelay());
            return;
        }
        this.pollError.set(null);
        this.pollDelay = 0;
        const activeIds = new Set(this.activeRuns().map((run) => run.id));
        const finished = previousActive.filter((run) => !activeIds.has(run.id));
        if (finished.some((run) => run.source_key === 'nist')) {
            void this.refreshCollections();
        }
        if (finished.length) void this.refreshOverview();
        if (finished.length) {
            await this.refreshHistoryOnly();
        }
        if (this.activeRuns().length) {
            this.schedulePoll(POLL_INTERVAL_MS);
        }
    }

    private async refreshHistoryOnly(): Promise<boolean> {
        this.historyLoading.set(true);
        const result = await fetchPublicDataSyncRuns({
            scope: 'history',
            page: this.historyPage(),
            pageSize: this.historyPageSize(),
        });
        if (this.destroyed) return false;
        this.historyLoading.set(false);
        if (!result.data) {
            this.pollError.set(result.error || 'History could not be refreshed.');
            return false;
        }
        const pageCount = Math.max(1, Math.ceil(result.data.total / this.historyPageSize()));
        if (this.historyPage() > pageCount) {
            this.historyPage.set(pageCount);
            return this.refreshHistoryOnly();
        }
        this.historyRuns.set(result.data.runs);
        this.historyTotal.set(result.data.total);
        this.syncDetailsRun();
        return true;
    }

    private syncDetailsRun(): void {
        const runId = this.detailsRunId();
        if (!runId) return;
        const run = [...this.activeRuns(), ...this.historyRuns()].find((item) => item.id === runId);
        if (run) this.detailsRun.set(run);
    }

    private async queue(
        key: string,
        request: () => Promise<ApiResult<PublicDataSyncRun>>,
    ): Promise<void> {
        if (this.isSubmitting(key)) return;
        this.error.set(null);
        this.markSubmitting(key, true);
        try {
            const result = await request();
            if (!result.data) {
                this.error.set(result.error);
                return;
            }
            const run = result.data;
            this.upsertRun(run);
            if (this.isActive(run)) {
                this.schedulePoll(POLL_INTERVAL_MS);
            } else {
                if (run.source_key === 'nist') void this.refreshCollections();
                void this.refreshOverview();
            }
        } finally {
            this.markSubmitting(key, false);
        }
    }

    private upsertRun(run: PublicDataSyncRun): void {
        if (this.isActive(run)) {
            this.activeRuns.update((runs) => [run, ...runs.filter((item) => item.id !== run.id)]);
            this.historyRuns.update((runs) => runs.filter((item) => item.id !== run.id));
        } else {
            const known = this.historyRuns().some((item) => item.id === run.id);
            this.activeRuns.update((runs) => runs.filter((item) => item.id !== run.id));
            this.historyRuns.update((runs) => [run, ...runs.filter((item) => item.id !== run.id)]);
            if (!known) this.historyTotal.update((total) => total + 1);
        }
        this.syncDetailsRun();
    }

    private markSubmitting(key: string, active: boolean): void {
        const next = new Set(this.pendingSubmissions());
        if (active) next.add(key);
        else next.delete(key);
        this.pendingSubmissions.set(next);
    }

    private schedulePoll(delay: number): void {
        if (this.destroyed) return;
        if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
        this.pollTimer = window.setTimeout(() => {
            this.pollTimer = null;
            void this.refreshRuns();
        }, delay);
    }

    private nextPollDelay(): number {
        this.pollDelay = this.pollDelay
            ? Math.min(POLL_RETRY_MAX_MS, this.pollDelay * 2)
            : POLL_RETRY_MIN_MS;
        return this.pollDelay;
    }
}
