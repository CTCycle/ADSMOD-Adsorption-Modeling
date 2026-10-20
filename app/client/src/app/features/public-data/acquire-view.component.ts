// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { AfterViewInit, Component, DestroyRef, ElementRef, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import type {
    CODSearchResult,
    MaterialRecordView,
    NISTCategory,
    NISTCollectionView,
    PublicDataSyncFailure,
    PublicDataSyncRun,
} from '../../models/public-data.model';
import {
    cancelPublicDataSyncRun,
    fetchNistCollections,
    fetchPublicDataSyncFailures,
    fetchPublicDataSyncRuns,
    importCOD,
    resolvePubChem,
    searchCOD,
    startPublicDataSync,
} from '../../services/public-data.service';
import { MaterialPickerComponent } from './material-picker.component';

interface ImportedLink {
    label: string;
    path: string[];
    queryParams: Record<string, string | number>;
}

const COLLECTIONS: { category: NISTCategory; label: string; description: string }[] = [
    { category: 'guest', label: 'Adsorbate species', description: 'Guest molecules with InChIKey, formula, and molecular weight.' },
    { category: 'host', label: 'Adsorbent materials', description: 'Host materials keyed by the NIST material hashkey.' },
    { category: 'experiments', label: 'Adsorption isotherms', description: 'Measured isotherms converted to canonical pressure and uptake.' },
];
const ACTIVE_STATUSES = new Set(['queued', 'running']);
const POLL_INTERVAL_MS = 1500;
const MAX_COD_SELECTION = 50;

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
    imports: [FormsModule, RouterLink, MaterialPickerComponent],
    template: `
        <section class="public-section acquire-intro">
            <div class="section-heading">
                <div>
                    <p class="eyebrow">Acquire public data</p>
                    <h3>Fetch records from NIST, PubChem, and COD</h3>
                    <p>Every request runs in the background and is saved to the local database. Records already stored are reused, never downloaded twice. Follow progress below, then open the imported records.</p>
                </div>
            </div>
            @if (error()) {
                <div class="public-alert error" role="alert"><strong>Request failed.</strong><span>{{ error() }}</span><button type="button" class="button secondary" (click)="error.set(null)">Dismiss</button></div>
            }
        </section>

        <section class="public-section source-panel" id="acquire-nist">
            <div class="section-heading">
                <div>
                    <p class="eyebrow">NIST/ARPA-E adsorption database</p>
                    <h4>Adsorption collections</h4>
                    <p>Check how many records NIST publishes, choose the share to consider, and fetch. Each run is bounded; fetch again to continue where the previous run stopped.</p>
                </div>
            </div>
            <div class="collection-list">
                @for (collection of collectionRows; track collection.category) {
                    @let state = collectionState(collection.category);
                    @let busy = isBusy('nist', collection.category);
                    <div class="collection-row">
                        <div class="collection-name">
                            <strong>{{ collection.label }}</strong>
                            <span>{{ collection.description }}</span>
                        </div>
                        <div class="collection-count">
                            <strong>{{ state.local_count.toLocaleString() }}{{ state.available_count === null ? '' : ' / ' + state.available_count.toLocaleString() }}</strong>
                            <span>{{ state.available_count === null ? 'stored · availability not checked' : 'stored / available · checked ' + formatDate(state.indexed_at) }}</span>
                        </div>
                        <label class="collection-fraction">Share of index
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
                        <div class="filter-actions">
                            <button class="button secondary" type="button" [disabled]="busy" (click)="checkAvailability(collection.category)">Check availability</button>
                            <button class="button primary" type="button" [disabled]="busy" (click)="fetchCollection(collection.category)">{{ busy ? 'Running…' : 'Fetch' }}</button>
                        </div>
                    </div>
                }
            </div>
        </section>

        <div class="acquire-grid">
            <section class="public-section source-panel" id="acquire-pubchem">
                <div class="section-heading">
                    <div>
                        <p class="eyebrow">PubChem</p>
                        <h4>Chemical identity and properties</h4>
                        <p>Resolve one compound by CID, InChIKey, or name, or fill in PubChem identity for stored chemicals that do not have it yet.</p>
                    </div>
                </div>
                @if (pubchemTarget(); as target) {
                    <p class="target-chip">Attaching the result to <strong>{{ target.name }}</strong><button class="text-button" type="button" (click)="pubchemTarget.set(null)">Clear</button></p>
                }
                <div class="provider-query">
                    <div><label for="acquire-pubchem-query">Compound</label><span>CID, InChIKey, or compound name</span></div>
                    <input id="acquire-pubchem-query" [(ngModel)]="pubchemQuery" placeholder="e.g. carbon dioxide or 280" (keyup.enter)="resolveCompound()" />
                    <button class="button primary" type="button" [disabled]="!pubchemQuery.trim()" (click)="resolveCompound()">Resolve</button>
                </div>
                <div class="panel-footer">
                    <span>Stored chemicals without PubChem identity are resolved in bounded batches. Lookups that PubChem cannot answer are not retried.</span>
                    <button class="button secondary" type="button" [disabled]="isBusy('pubchem', 'pubchem_enrich')" (click)="enrichChemicals()">Enrich stored chemicals</button>
                </div>
            </section>

            <section class="public-section source-panel" id="acquire-cod">
                <div class="section-heading">
                    <div>
                        <p class="eyebrow">Crystallography Open Database</p>
                        <h4>Crystal structures</h4>
                        <p>Search with a bounded query, select results, and import the original CIF with unit cell and atom sites.</p>
                    </div>
                </div>
                <div class="provider-query structure-query">
                    <div><label for="acquire-cod-query">Search</label><span>Broad result sets are rejected by the server.</span></div>
                    <input id="acquire-cod-query" [(ngModel)]="codQuery" placeholder="Text, formula, or COD ID" (keyup.enter)="searchStructures()" />
                    <select [(ngModel)]="codQueryType" aria-label="COD query type"><option value="text">Text</option><option value="formula">Formula</option><option value="id">COD ID</option></select>
                    <button class="button primary" type="button" [disabled]="codSearching() || !codQuery.trim()" (click)="searchStructures()">{{ codSearching() ? 'Searching…' : 'Search' }}</button>
                </div>
                @if (codResults().length) {
                    <div class="table-frame cod-results">
                        <table>
                            <thead><tr><th class="select-column"><input type="checkbox" [checked]="allCodSelected()" aria-label="Select all results" (change)="toggleAllCod($event)" /></th><th>COD ID</th><th>Name</th><th>Formula</th><th>Space group</th><th>Reference</th></tr></thead>
                            <tbody>
                                @for (item of codResults(); track item.cod_id) {
                                    <tr>
                                        <td class="select-column"><input type="checkbox" [checked]="selectedCodIds().includes(item.cod_id)" [attr.aria-label]="'Select COD ' + item.cod_id" (change)="toggleCod(item.cod_id, $event)" /></td>
                                        <td><a [href]="item.source_url" target="_blank" rel="noopener noreferrer">{{ item.cod_id }}</a></td>
                                        <td class="truncate-cell" [title]="item.name || ''">{{ item.name || '—' }}</td>
                                        <td class="truncate-cell" [title]="item.formula || ''">{{ item.formula || '—' }}</td>
                                        <td>{{ item.space_group || '—' }}</td>
                                        <td class="truncate-cell" [title]="item.doi || ''">{{ item.year ?? '' }}{{ item.doi ? ' · ' + item.doi : '' }}</td>
                                    </tr>
                                }
                            </tbody>
                        </table>
                    </div>
                    <div class="cod-import-bar">
                        @if (codLinkMaterial(); as material) {
                            <p class="target-chip">Link imports to <strong>{{ material.name }}</strong><button class="text-button" type="button" (click)="codLinkMaterial.set(null)">Change</button></p>
                        } @else {
                            <adsmod-material-picker label="Link imports to material (optional)" (selected)="codLinkMaterial.set($event)" />
                        }
                        <button class="button primary" type="button" [disabled]="!selectedCodIds().length" (click)="importStructures()">Import selected ({{ selectedCodIds().length }})</button>
                    </div>
                } @else if (codSearchCompleted()) {
                    <p class="empty-inline" role="status">No COD records matched. Try a COD ID, a formula such as “C6 H6”, or a narrower text query.</p>
                }
            </section>
        </div>

        <section class="public-section" id="acquire-runs">
            <div class="section-heading">
                <div><p class="eyebrow">Run tracker</p><h4>Recent acquisitions</h4></div>
                <button class="button secondary" type="button" (click)="refreshRuns()">Refresh</button>
            </div>
            @if (trackedRuns().length) {
                <div class="sync-run-list">
                    @for (run of trackedRuns(); track run.id) {
                        <article class="sync-run-row">
                            <div><strong>{{ operationLabel(run) }}</strong><span>{{ run.source_key }} · {{ run.status.replace('_', ' ') }} · {{ formatDate(run.created_at) }}</span></div>
                            <progress max="100" [value]="run.progress" [attr.aria-label]="operationLabel(run) + ' progress'"></progress>
                            <span>{{ run.imported_count }} saved · {{ run.skipped_count }} already stored · {{ run.failed_count }} failed</span>
                            <span class="run-actions">
                                @if (isActive(run)) {
                                    <button class="text-button" type="button" (click)="cancelRun(run.id)">Cancel</button>
                                } @else if (importedLink(run); as link) {
                                    <a class="text-link" [routerLink]="link.path" [queryParams]="link.queryParams">{{ link.label }} →</a>
                                }
                            </span>
                            @if (run.error_detail) {<p class="sync-run-error">{{ run.error_detail }}</p>}
                            @if (pendingCount(run); as pending) {<p class="run-note">{{ pending }} records remain; run it again to continue.</p>}
                            @if (run.failed_count || run.error_detail) {
                                <button class="text-button run-failures-toggle" type="button" (click)="toggleFailures(run.id)">{{ failureRun() === run.id ? 'Hide' : 'Show' }} failed records</button>
                            }
                            @if (failureRun() === run.id) {
                                <div class="sync-run-details">
                                    @if (failures().length) {
                                        <ul class="sync-failures">@for (failure of failures(); track failure.id) {<li><strong>{{ failure.external_id || 'Record' }}</strong><span>{{ failure.reason }}</span></li>}</ul>
                                    } @else {<p>No record-level details were returned for this run.</p>}
                                </div>
                            }
                        </article>
                    }
                </div>
            } @else {
                <p class="empty-inline">No acquisitions yet. Start with a NIST collection, a PubChem compound, or a COD search above.</p>
            }
        </section>
    `,
    styleUrls: ['./public-data-workspace.component.css', './acquire-view.component.css'],
})
export class AcquireViewComponent implements OnInit, AfterViewInit {
    private readonly route = inject(ActivatedRoute);
    private readonly host = inject(ElementRef<HTMLElement>);
    private readonly destroyRef = inject(DestroyRef);
    private pollTimer: number | null = null;
    private destroyed = false;

    protected readonly collectionRows = COLLECTIONS;
    protected readonly collections = signal<NISTCollectionView[]>([]);
    protected readonly fractions = signal<Record<NISTCategory, number>>({
        guest: readFraction('guest'),
        host: readFraction('host'),
        experiments: readFraction('experiments'),
    });
    protected readonly runs = signal<PublicDataSyncRun[]>([]);
    protected readonly failureRun = signal<string | null>(null);
    protected readonly failures = signal<PublicDataSyncFailure[]>([]);
    protected readonly error = signal<string | null>(null);

    protected pubchemQuery = '';
    protected readonly pubchemTarget = signal<{ id: number; name: string } | null>(null);

    protected codQuery = '';
    protected codQueryType: 'text' | 'formula' | 'id' = 'text';
    protected readonly codSearching = signal(false);
    protected readonly codSearchCompleted = signal(false);
    protected readonly codResults = signal<CODSearchResult[]>([]);
    protected readonly selectedCodIds = signal<string[]>([]);
    protected readonly codLinkMaterial = signal<Pick<MaterialRecordView, 'id' | 'name'> | null>(null);

    protected readonly trackedRuns = computed(() => this.runs().filter((run) => run.operation !== 'health_check'));
    protected readonly allCodSelected = computed(
        () => this.codResults().length > 0 && this.selectedCodIds().length === Math.min(this.codResults().length, MAX_COD_SELECTION),
    );

    constructor() {
        this.destroyRef.onDestroy(() => {
            this.destroyed = true;
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
        void this.refreshRuns();
    }

    ngAfterViewInit(): void {
        const panel = this.route.snapshot.queryParamMap.get('panel');
        if (panel) {
            this.host.nativeElement.querySelector(`#acquire-${panel}`)?.scrollIntoView({ block: 'start' });
        }
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

    protected checkAvailability(category: NISTCategory): Promise<void> {
        return this.queue(startPublicDataSync({ source_key: 'nist', operation: 'nist_index', category }));
    }

    protected fetchCollection(category: NISTCategory): Promise<void> {
        return this.queue(
            startPublicDataSync({ source_key: 'nist', operation: 'nist_category', category, fraction: this.fractions()[category] }),
        );
    }

    protected async resolveCompound(): Promise<void> {
        const query = this.pubchemQuery.trim();
        if (!query) return;
        await this.queue(resolvePubChem(query, this.pubchemTarget()?.id));
        this.pubchemQuery = '';
        this.pubchemTarget.set(null);
    }

    protected enrichChemicals(): Promise<void> {
        return this.queue(startPublicDataSync({ source_key: 'pubchem', operation: 'pubchem_enrich' }));
    }

    protected async searchStructures(): Promise<void> {
        const query = this.codQuery.trim();
        if (!query) return;
        this.codSearching.set(true);
        this.codSearchCompleted.set(false);
        this.codResults.set([]);
        this.selectedCodIds.set([]);
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
        await this.queue(importCOD(selected, this.codLinkMaterial()?.id));
        this.selectedCodIds.set([]);
    }

    protected async refreshRuns(): Promise<void> {
        const result = await fetchPublicDataSyncRuns();
        if (this.destroyed) return;
        if (!result.data) {
            this.error.set(result.error);
            return;
        }
        const previouslyActive = new Set(this.runs().filter((run) => this.isActive(run)).map((run) => run.id));
        this.runs.set(result.data.runs);
        const finishedNist = result.data.runs.some(
            (run) => previouslyActive.has(run.id) && !this.isActive(run) && run.source_key === 'nist',
        );
        if (finishedNist) void this.refreshCollections();
        this.schedulePoll();
    }

    protected async cancelRun(runId: string): Promise<void> {
        const result = await cancelPublicDataSyncRun(runId);
        if (result.error) this.error.set(result.error);
        await this.refreshRuns();
    }

    protected async toggleFailures(runId: string): Promise<void> {
        if (this.failureRun() === runId) {
            this.failureRun.set(null);
            return;
        }
        this.failureRun.set(runId);
        this.failures.set([]);
        const result = await fetchPublicDataSyncFailures(runId);
        if (result.data) this.failures.set(result.data.failures);
        else this.error.set(result.error);
    }

    protected isActive(run: PublicDataSyncRun): boolean {
        return ACTIVE_STATUSES.has(run.status);
    }

    protected isBusy(source: string, key: string): boolean {
        return this.runs().some(
            (run) => this.isActive(run) && run.source_key === source && (run.parameters['category'] === key || run.operation === key),
        );
    }

    protected operationLabel(run: PublicDataSyncRun): string {
        const category = COLLECTIONS.find((item) => item.category === run.parameters['category'])?.label ?? 'collection';
        switch (run.operation) {
            case 'nist_index': return `NIST availability · ${category}`;
            case 'nist_category': return `NIST fetch · ${category}`;
            case 'pubchem_resolve': return `PubChem · ${String(run.parameters['query'] ?? 'compound')}`;
            case 'pubchem_enrich': return 'PubChem enrichment of stored chemicals';
            case 'cod_import': {
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
                if (category === 'experiments') return { label: 'View isotherms', path: ['/public-data', 'adsorption'], queryParams: { source: 'nist' } };
                if (category === 'guest') return { label: 'View chemicals', path: ['/public-data', 'chemicals'], queryParams: { source: 'nist' } };
                return { label: 'View materials', path: ['/public-data', 'materials'], queryParams: { source: 'nist' } };
            case 'pubchem_resolve':
                return single === null ? null : { label: 'Open chemical', path: ['/public-data', 'chemicals'], queryParams: { id: single } };
            case 'pubchem_enrich':
                return run.imported_count || run.skipped_count ? { label: 'View chemicals', path: ['/public-data', 'chemicals'], queryParams: { source: 'pubchem' } } : null;
            case 'cod_import':
                if (single !== null) return { label: 'Open structure', path: ['/public-data', 'structures'], queryParams: { id: single } };
                return run.imported_count || run.skipped_count ? { label: 'View structures', path: ['/public-data', 'structures'], queryParams: { source: 'cod' } } : null;
            default:
                return null;
        }
    }

    protected pendingCount(run: PublicDataSyncRun): number {
        if (this.isActive(run) || run.operation === 'cod_import') return 0;
        return Number(run.checkpoint['pending_count'] ?? 0);
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

    private async queue(request: Promise<{ data: PublicDataSyncRun | null; error: string | null }>): Promise<void> {
        this.error.set(null);
        const result = await request;
        if (!result.data) {
            this.error.set(result.error);
            return;
        }
        const queued = result.data;
        this.runs.update((runs) => [queued, ...runs.filter((run) => run.id !== queued.id)]);
        this.schedulePoll();
        this.host.nativeElement.querySelector('#acquire-runs')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    private schedulePoll(): void {
        if (this.pollTimer !== null || this.destroyed) return;
        if (!this.runs().some((run) => this.isActive(run))) return;
        this.pollTimer = window.setTimeout(() => {
            this.pollTimer = null;
            void this.refreshRuns();
        }, POLL_INTERVAL_MS);
    }
}
