// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, ParamMap, Router, RouterLink, RouterLinkActive } from '@angular/router';
import { combineLatest, debounceTime } from 'rxjs';
import type {
    PublicDataOverviewResponse,
    PublicDataSyncRun,
    PublicSourceSummary,
    SourcesView,
    StructurePageResponse,
    StructureRecordView,
} from '../../models/public-data.model';
import {
    checkPublicSourceHealth,
    fetchPublicDataSyncRun,
    fetchPublicOverview,
    fetchPublicSources,
    fetchPublicStructureDetail,
    fetchPublicStructures,
    setStructureMaterial,
    structureFileUrl,
} from '../../services/public-data.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';
import { AcquireViewComponent } from './acquire-view.component';
import { EntityPickerComponent, type PickedEntity } from '../../shared/components/entity-picker/entity-picker.component';

const VIEWS: readonly SourcesView[] = ['catalog', 'acquire', 'structures'];
const VIEW_LABELS: Record<SourcesView, string> = {
    catalog: 'Catalog',
    acquire: 'Acquire',
    structures: 'Structures',
};

/** Catalog sections, keyed by the primary data type a provider contributes. */
const CATEGORIES: readonly { capability: string; title: string; description: string }[] = [
    { capability: 'adsorption', title: 'Adsorption isotherms', description: 'Measured isotherms that become experiments in a local dataset.' },
    { capability: 'chemicals', title: 'Chemical identity and properties', description: 'Identifiers, synonyms, and properties that enrich local species.' },
    { capability: 'structures', title: 'Crystal structures', description: 'Crystallographic records that can be linked to local materials.' },
];
const CAPABILITY_LABELS: Record<string, string> = {
    adsorption: 'Isotherms',
    materials: 'Materials',
    chemicals: 'Species',
    structures: 'Structures',
    references: 'References',
};

const positiveId = (value: string | null): number | null => {
    const parsed = Number(value);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

@Component({
    selector: 'adsmod-sources-page',
    standalone: true,
    imports: [FormsModule, RouterLink, RouterLinkActive, AcquireViewComponent, EntityPickerComponent, DataPaginationComponent],
    template: `
        <div class="public-workspace">
            <nav class="public-tabs" aria-label="Sources">
                @for (tab of views; track tab) {
                    <a [routerLink]="['/sources', tab]" routerLinkActive="active">{{ viewLabel(tab) }}</a>
                }
            </nav>

            @if (error()) {
                <div class="public-alert error" role="alert">
                    <strong>Source request failed.</strong>
                    <span>{{ error() }}</span>
                    <button type="button" class="button secondary" (click)="reload()">Retry</button>
                </div>
            }

            @if (loading()) {
                <div class="public-loading" aria-live="polite">
                    <span class="public-spinner" aria-hidden="true"></span>
                    <span>Loading {{ viewLabel(view()) }}…</span>
                </div>
            }

            @switch (view()) {
                @case ('catalog') {
                    <section class="public-intro">
                        <div>
                            <h2>External data sources</h2>
                            <p>
                                Browse the providers ADSMOD integrates. Browsing reads the provider; acquiring copies records into your local database,
                                where isotherms appear as datasets and materials and species become reusable entities on the Datasets page.
                            </p>
                            <div class="catalog-actions">
                                <a class="button primary" routerLink="/sources/acquire">Acquire data</a>
                                <a class="button secondary" routerLink="/datasets">Open local datasets</a>
                            </div>
                        </div>
                        <div class="overview-totals" aria-label="Records acquired from providers">
                            <div><strong>{{ overview()?.adsorption_records ?? '—' }}</strong><span>acquired isotherms</span></div>
                            <div><strong>{{ overview()?.materials ?? '—' }}</strong><span>provider materials</span></div>
                            <div><strong>{{ overview()?.chemicals ?? '—' }}</strong><span>provider species</span></div>
                            <a routerLink="/sources/structures"><strong>{{ overview()?.structures ?? '—' }}</strong><span>structures</span></a>
                        </div>
                        <p class="acquisition-summary" aria-live="polite">
                            <a routerLink="/sources/acquire">{{ overview()?.active_sync_runs ?? 0 }} active</a>
                            · <a routerLink="/sources/acquire">{{ overview()?.failed_sync_runs ?? 0 }} failed</a>
                            · latest successful acquisition {{ formatDate(overview()?.latest_successful_sync_at ?? null) }}
                        </p>
                    </section>
                    <div class="catalog-toolbar">
                        <p>Status reflects the last saved provider check.</p>
                        <button class="button secondary" type="button" [disabled]="sourceHealthLoading()" (click)="refreshSourceHealth()">{{ sourceHealthLoading() ? 'Checking…' : 'Refresh status' }}</button>
                    </div>
                    @for (group of catalog(); track group.title) {
                        <section class="public-section catalog-group" [attr.aria-labelledby]="'catalog-' + group.capability">
                            <div class="section-heading">
                                <div>
                                    <h3 [id]="'catalog-' + group.capability">{{ group.title }}</h3>
                                    <p>{{ group.description }}</p>
                                </div>
                            </div>
                            <div class="source-summary-grid">
                                @for (source of group.sources; track source.key) {
                                    <article class="source-summary">
                                        <div class="source-summary-head">
                                            <div><strong>{{ source.name }}</strong><span class="source-key">{{ source.key }}</span></div>
                                            <span class="source-status" [class.available]="source.health_status === 'available'" [class.unavailable]="source.health_status === 'unavailable'">
                                                {{ source.health_status === 'unknown' ? 'Not checked' : source.health_status }}
                                            </span>
                                        </div>
                                        <p>{{ source.description }}</p>
                                        @if (source.health_detail) {
                                            <p class="source-detail" [title]="source.health_detail">{{ source.health_detail }}</p>
                                        }
                                        <div class="capability-list" aria-label="Available data types">
                                            @for (capability of source.capabilities; track capability) { <span>{{ capabilityLabel(capability) }}</span> }
                                        </div>
                                        <dl class="source-facts">
                                            <div><dt>Local records</dt><dd>{{ source.record_count }}</dd></div>
                                            <div><dt>Last acquisition</dt><dd>{{ formatDate(source.last_successful_sync_at) }}</dd></div>
                                            <div><dt>Last checked</dt><dd>{{ formatDate(source.last_health_checked_at) }}@if (!source.health_fresh && source.last_health_checked_at) { <span class="source-stale">stale</span> }</dd></div>
                                        </dl>
                                        <div class="source-summary-foot">
                                            <span class="source-links">
                                                <a [href]="source.homepage_url" target="_blank" rel="noopener noreferrer">Website</a>
                                                @if (source.license_url) {
                                                    <a [href]="source.license_url" target="_blank" rel="noopener noreferrer">{{ source.license_name && source.license_name.length <= 24 ? source.license_name : 'License' }}</a>
                                                }
                                            </span>
                                            <a [routerLink]="['/sources', 'acquire']" [queryParams]="{ panel: source.key }">Acquire</a>
                                        </div>
                                    </article>
                                }
                            </div>
                        </section>
                    } @empty {
                        @if (!loading()) {
                            <p class="empty-inline">No providers are registered.</p>
                        }
                    }
                }
                @case ('acquire') {
                    <adsmod-public-data-acquire />
                }
                @case ('structures') {
                    <section class="public-section">
                        <div class="section-heading"><div><h3>Structures</h3><p>Imported COD structures with their original CIF, unit cell, and atom sites. Link a structure to the material it describes.</p></div><a class="button secondary" [routerLink]="['/sources', 'acquire']" [queryParams]="{ panel: 'cod' }">Search COD</a></div>
                        @if (structureMaterialFilter()) {
                            <p class="filter-chip">Showing structures linked to material #{{ structureMaterialFilter() }}<a class="text-link" [routerLink]="[]" [queryParams]="{ material_id: null, id: null }" queryParamsHandling="merge">Clear</a></p>
                        }
                        <div class="filter-bar compact"><label>Search<input [(ngModel)]="structureQuery" placeholder="Name or formula" /></label><label>Source<select [(ngModel)]="structureSource"><option value="">All sources</option>@for (source of sources(); track source.key) {<option [value]="source.key">{{ source.name }}</option>}</select></label><label>Link status<select [(ngModel)]="structureLink"><option value="">Any</option><option value="linked">Linked to material</option><option value="unlinked">Unlinked</option></select></label><div></div><div class="filter-actions"><button class="button primary" type="button" (click)="applyStructureFilters()">Apply</button><button class="button secondary" type="button" (click)="resetStructureFilters()">Reset</button></div></div>
                        <div class="data-table-frame structures-table-frame">
                            <table class="data-table structures-table">
                                <caption class="sr-only">Crystal structures</caption>
                                <thead>
                                    <tr>
                                        <th class="col-source" scope="col">Source</th>
                                        <th class="col-external" scope="col">Source ID</th>
                                        <th class="col-material" scope="col">Material</th>
                                        <th class="col-formula" scope="col">Formula</th>
                                        <th class="col-space-group" scope="col">Space group</th>
                                        <th class="col-count num" scope="col">Atoms</th>
                                        <th class="col-date" scope="col">Retrieved</th>
                                        <th class="col-actions" scope="col"></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @for (item of structures()?.items ?? []; track item.id) {
                                        <tr>
                                            <td><span class="source-key">{{ item.source }}</span></td>
                                            <td class="truncate-cell" [title]="item.external_id">{{ item.external_id }}</td>
                                            <td class="truncate-cell" [title]="item.material_name || 'Unlinked'">{{ item.material_name || 'Unlinked' }}</td>
                                            <td class="truncate-cell" [title]="item.formula || ''">{{ item.formula || '—' }}</td>
                                            <td class="truncate-cell" [title]="item.space_group || ''">{{ item.space_group || '—' }}</td>
                                            <td class="num">{{ item.atom_count }}</td>
                                            <td>{{ formatDate(item.retrieved_at) }}</td>
                                            <td class="col-actions">
                                                <div class="table-actions">
                                                    <button class="table-action" type="button" (click)="openDetail(item.id)">Inspect</button>
                                                </div>
                                            </td>
                                        </tr>
                                    } @empty {
                                        <tr><td colspan="8" class="data-table-empty">No imported structures match the current filters. <a class="text-link" [routerLink]="['/sources', 'acquire']" [queryParams]="{ panel: 'cod' }">Search COD</a></td></tr>
                                    }
                                </tbody>
                            </table>
                        </div>
                        <adsmod-data-pagination
                            [page]="page()"
                            [pageSize]="pageSize()"
                            [total]="structures()?.pagination?.total ?? 0"
                            itemLabel="structure"
                            [disabled]="loading()"
                            (pageChange)="pageChanged($event)"
                            (pageSizeChange)="pageSizeChanged($event)"
                        />
                    </section>
                    @if (structureDetail(); as detail) {
                        <aside class="detail-panel" aria-label="Structure details">
                            <div class="detail-header"><div><p class="eyebrow">{{ detail.source }} · {{ detail.external_id }}</p><h3>{{ detail.name || detail.formula || 'Structure record' }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="closeDetail()">×</button></div>
                            <div class="detail-grid"><div><span>Formula</span><strong>{{ detail.formula || '—' }}</strong></div><div><span>Space group</span><strong>{{ detail.space_group || '—' }}</strong></div><div><span>Unit-cell volume</span><strong>{{ detail.cell_volume_angstrom3 === null ? '—' : formatNumber(detail.cell_volume_angstrom3) + ' Å³' }}</strong></div><div><span>Normalized atoms</span><strong>{{ detail.atom_count }}</strong></div></div>
                            <div class="detail-section"><h4>Material</h4>
                                @if (detail.material_id) {
                                    <p class="link-row"><span>{{ detail.material_name }}</span><button class="button secondary" type="button" [disabled]="linkBusy()" (click)="linkStructure(detail.id, null)">Unlink</button></p>
                                } @else {
                                    <div class="link-row"><adsmod-entity-picker kind="material" [disabled]="linkBusy()" (selected)="structureLinkCandidate.set($event)" /><button class="button primary" type="button" [disabled]="linkBusy() || !structureLinkCandidate()" (click)="linkStructure(detail.id, structureLinkCandidate()?.id ?? null)">Link</button></div>
                                }
                            </div>
                            <div class="detail-section"><h4>Unit cell</h4><p>a {{ valueOrDash(detail.cell_a_angstrom) }} Å · b {{ valueOrDash(detail.cell_b_angstrom) }} Å · c {{ valueOrDash(detail.cell_c_angstrom) }} Å</p><p>α {{ valueOrDash(detail.cell_alpha_deg) }}° · β {{ valueOrDash(detail.cell_beta_deg) }}° · γ {{ valueOrDash(detail.cell_gamma_deg) }}°</p></div>
                            <div class="detail-section"><h4>Atomic coordinates</h4><div class="atom-list">@for (atom of detail.atoms.slice(0, 50); track atom.sequence_index) {<code>{{ atom.label }} {{ atom.element }} ({{ formatNumber(atom.fractional_x) }}, {{ formatNumber(atom.fractional_y) }}, {{ formatNumber(atom.fractional_z) }})</code>} @empty {<p class="empty-inline">No fractional atom coordinates were normalized from this CIF.</p>}</div>@if (detail.atoms.length > 50) {<p class="detail-note">Showing the first 50 of {{ detail.atoms.length }} atom sites.</p>}</div>
                            <div class="detail-section"><h4>Reference</h4>@if (detail.doi) {<p><a class="text-link" [href]="'https://doi.org/' + detail.doi" target="_blank" rel="noopener noreferrer">DOI {{ detail.doi }}</a>{{ detail.reference_title ? ' · ' + detail.reference_title : '' }}</p>} @else {<p class="empty-inline">No publication reference was supplied by the source.</p>}</div>
                            <div class="detail-links"><a class="button secondary" [href]="structureFile(detail.id)" download>Download {{ detail.format.toUpperCase() }}</a>@if (detail.source_url) {<a class="button secondary" [href]="detail.source_url" target="_blank" rel="noopener noreferrer">Open source record</a>}</div>
                        </aside>
                    }
                }
            }
        </div>
    `,
    styleUrl: './sources-page.component.css',
})
export class SourcesPageComponent {
    private readonly route = inject(ActivatedRoute);
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);
    private loadSequence = 0;
    private detailSequence = 0;
    private sourceHealthSequence = 0;
    private listKey = '';

    protected readonly views = VIEWS;
    protected readonly view = signal<SourcesView>('catalog');
    protected readonly loading = signal(false);
    protected readonly sourceHealthLoading = signal(false);
    protected readonly linkBusy = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly sources = signal<PublicSourceSummary[]>([]);
    protected readonly overview = signal<PublicDataOverviewResponse | null>(null);
    protected readonly structures = signal<StructurePageResponse | null>(null);
    protected readonly structureDetail = signal<StructureRecordView | null>(null);
    protected readonly structureLinkCandidate = signal<PickedEntity | null>(null);
    protected readonly structureMaterialFilter = signal<number | null>(null);
    protected readonly page = signal(1);
    protected readonly pageSize = signal(25);

    protected structureQuery = '';
    protected structureSource = '';
    protected structureLink = '';

    /** Each provider is listed once, under the first category it contributes to. */
    protected readonly catalog = computed(() => {
        const placed = new Set<string>();
        const groups = CATEGORIES.map((category) => {
            const sources = this.sources().filter(
                (source) => !placed.has(source.key) && source.capabilities.includes(category.capability),
            );
            sources.forEach((source) => placed.add(source.key));
            return { ...category, sources };
        });
        const other = this.sources().filter((source) => !placed.has(source.key));
        if (other.length) {
            groups.push({ capability: 'other', title: 'Other sources', description: 'Additional providers.', sources: other });
        }
        return groups.filter((group) => group.sources.length);
    });

    constructor() {
        // Path and query parameters emit separately within one navigation; settle
        // them first so a new view never pairs with the previous view's record id.
        combineLatest([this.route.paramMap, this.route.queryParamMap])
            .pipe(debounceTime(0), takeUntilDestroyed(this.destroyRef))
            .subscribe(([params, query]) => this.applyRoute(params, query));
    }

    protected viewLabel(view: SourcesView): string {
        return VIEW_LABELS[view];
    }

    protected capabilityLabel(capability: string): string {
        return CAPABILITY_LABELS[capability] ?? capability;
    }

    protected async reload(): Promise<void> {
        await this.loadCurrent();
    }

    private applyRoute(params: ParamMap, query: ParamMap): void {
        const rawView = params.get('view') ?? 'catalog';
        if (!VIEWS.includes(rawView as SourcesView)) {
            void this.router.navigate(['/sources', 'catalog'], { replaceUrl: true });
            return;
        }
        const nextView = rawView as SourcesView;
        const source = query.get('source') ?? '';
        const materialId = positiveId(query.get('material_id'));
        const listKey = JSON.stringify([nextView, source, materialId]);
        if (listKey !== this.listKey) {
            const viewChanged = nextView !== this.view() || !this.listKey;
            this.listKey = listKey;
            this.view.set(nextView);
            this.page.set(1);
            if (viewChanged) this.closeDetails();
            this.structureMaterialFilter.set(nextView === 'structures' ? materialId : null);
            if (source && nextView === 'structures') this.structureSource = source;
            void this.loadCurrent();
        }
        if (nextView === 'structures') void this.loadDetail(positiveId(query.get('id')));
    }

    private isCurrentLoad(requestId: number, expectedView: SourcesView): boolean {
        return requestId === this.loadSequence && this.view() === expectedView;
    }

    private async loadCurrent(): Promise<void> {
        const requestId = ++this.loadSequence;
        const currentView = this.view();
        if (currentView !== 'catalog') {
            this.sourceHealthSequence += 1;
            this.sourceHealthLoading.set(false);
        }
        this.error.set(null);
        if (currentView === 'acquire') return;
        this.loading.set(true);
        try {
            if (currentView === 'catalog') {
                await Promise.all([this.refreshSources(), this.loadOverview(requestId)]);
            } else {
                await Promise.all([this.sources().length ? Promise.resolve() : this.refreshSources(), this.loadStructures(requestId)]);
            }
        } finally {
            if (this.isCurrentLoad(requestId, currentView)) this.loading.set(false);
        }
    }

    private async loadOverview(requestId: number): Promise<void> {
        const result = await fetchPublicOverview();
        if (!this.isCurrentLoad(requestId, 'catalog')) return;
        this.overview.set(result.data);
        if (result.error) this.error.set(result.error);
    }

    private async loadStructures(requestId = this.loadSequence): Promise<void> {
        const linked = this.structureLink === 'linked' ? true : this.structureLink === 'unlinked' ? false : null;
        const result = await fetchPublicStructures({ page: this.page(), page_size: this.pageSize(), q: this.structureQuery, source: this.structureSource, linked_only: linked, material_id: this.structureMaterialFilter() });
        if (!this.isCurrentLoad(requestId, 'structures')) return;
        this.structures.set(result.data);
        this.error.set(result.error);
    }

    protected async applyStructureFilters(): Promise<void> { this.page.set(1); await this.loadStructures(); }
    protected async resetStructureFilters(): Promise<void> {
        this.structureQuery = ''; this.structureSource = ''; this.structureLink = '';
        if (this.structureMaterialFilter()) {
            await this.router.navigate([], { relativeTo: this.route, queryParams: { material_id: null, source: null, id: null }, queryParamsHandling: 'merge' });
            return;
        }
        await this.applyStructureFilters();
    }

    protected async pageChanged(page: number): Promise<void> {
        if (page < 1 || page === this.page()) return;
        this.page.set(page);
        await this.loadCurrent();
    }

    protected async pageSizeChanged(size: number): Promise<void> {
        if (size === this.pageSize()) return;
        this.pageSize.set(size);
        this.page.set(1);
        await this.loadCurrent();
    }

    protected openDetail(id: number): void {
        void this.router.navigate([], { relativeTo: this.route, queryParams: { id }, queryParamsHandling: 'merge' });
    }

    protected closeDetail(): void {
        void this.router.navigate([], { relativeTo: this.route, queryParams: { id: null }, queryParamsHandling: 'merge' });
    }

    private async loadDetail(id: number | null): Promise<void> {
        const requestId = ++this.detailSequence;
        if (id === null) {
            this.closeDetails();
            return;
        }
        if (this.structureDetail()?.id === id) return;
        this.structureLinkCandidate.set(null);
        const result = await fetchPublicStructureDetail(id);
        if (requestId !== this.detailSequence) return;
        this.structureDetail.set(result.data);
        if (result.error) this.error.set(result.error);
    }

    protected async linkStructure(structureId: number, materialId: number | null): Promise<void> {
        this.linkBusy.set(true);
        try {
            const result = await setStructureMaterial(structureId, materialId);
            if (!result.data) {
                this.error.set(result.error);
                return;
            }
            this.structureDetail.set(result.data);
            this.structureLinkCandidate.set(null);
            await this.loadStructures();
        } finally {
            this.linkBusy.set(false);
        }
    }

    protected structureFile(id: number): string {
        return structureFileUrl(id);
    }

    private async refreshSources(): Promise<void> {
        const result = await fetchPublicSources();
        if (result.data) this.sources.set(result.data.sources);
        else this.error.set(result.error);
    }

    protected async refreshSourceHealth(): Promise<void> {
        const requestId = ++this.sourceHealthSequence;
        const isCurrent = (): boolean => requestId === this.sourceHealthSequence && this.view() === 'catalog';
        this.sourceHealthLoading.set(true);
        this.error.set(null);
        try {
            const queued = await Promise.all(this.sources().map((source) => checkPublicSourceHealth(source.key)));
            const errors = queued.flatMap((result) => (result.error ? [result.error] : []));
            if (errors.length) this.error.set(errors.join(' '));
            let pending: PublicDataSyncRun[] = queued.flatMap((result) => (result.data ? [result.data] : []));
            while (pending.length && isCurrent()) {
                await new Promise((resolve) => window.setTimeout(resolve, 800));
                const updated = await Promise.all(pending.map((run) => fetchPublicDataSyncRun(run.id)));
                pending = updated.flatMap((result) => (result.data && ['queued', 'running'].includes(result.data.status) ? [result.data] : []));
            }
            if (isCurrent()) await this.refreshSources();
        } finally {
            if (isCurrent()) this.sourceHealthLoading.set(false);
        }
    }

    private closeDetails(): void {
        this.structureDetail.set(null);
        this.structureLinkCandidate.set(null);
    }

    protected formatNumber(value: number): string { return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(value); }
    protected valueOrDash(value: number | null): string { return value === null ? '—' : this.formatNumber(value); }
    protected formatDate(value: string | null): string { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date); }
}
