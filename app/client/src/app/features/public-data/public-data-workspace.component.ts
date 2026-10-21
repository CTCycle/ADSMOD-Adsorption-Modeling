// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, ParamMap, Router, RouterLink, RouterLinkActive } from '@angular/router';
import { combineLatest, debounceTime } from 'rxjs';
import type {
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialDetailResponse,
    MaterialPageResponse,
    MaterialRecordView,
    PublicDataOverviewResponse,
    PublicDataSyncRun,
    PublicDataView,
    PublicSourceSummary,
    StructurePageResponse,
    StructureRecordView,
} from '../../models/public-data.model';
import {
    checkPublicSourceHealth,
    fetchPublicAdsorption,
    fetchPublicAdsorptionDetail,
    fetchPublicChemicalDetail,
    fetchPublicChemicals,
    fetchPublicDataSyncRun,
    fetchPublicMaterialDetail,
    fetchPublicMaterials,
    fetchPublicOverview,
    fetchPublicSources,
    fetchPublicStructureDetail,
    fetchPublicStructures,
    setStructureMaterial,
    structureFileUrl,
} from '../../services/public-data.service';
import { AcquireViewComponent } from './acquire-view.component';
import { MaterialPickerComponent } from './material-picker.component';

const VIEWS: readonly PublicDataView[] = [
    'overview',
    'acquire',
    'adsorption',
    'materials',
    'chemicals',
    'structures',
    'sources',
];
const VIEW_LABELS: Record<PublicDataView, string> = {
    overview: 'Overview',
    acquire: 'Acquire',
    adsorption: 'Adsorption Data',
    materials: 'Materials',
    chemicals: 'Chemicals',
    structures: 'Structures',
    sources: 'Sources',
};

const positiveId = (value: string | null): number | null => {
    const parsed = Number(value);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

@Component({
    selector: 'adsmod-public-data-workspace',
    standalone: true,
    imports: [FormsModule, RouterLink, RouterLinkActive, AcquireViewComponent, MaterialPickerComponent],
    template: `
        <div class="public-workspace">
            <nav class="public-tabs" aria-label="Public data workspace">
                @for (tab of views; track tab) {
                    <a [routerLink]="['/public-data', tab]" routerLinkActive="active">{{ viewLabel(tab) }}</a>
                }
            </nav>

            @if (error()) {
                <div class="public-alert error" role="alert">
                    <strong>Public data request failed.</strong>
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
                @case ('overview') {
                    <section class="public-intro">
                        <div>
                            <h2>Local public data</h2>
                            <p>Records fetched from NIST, PubChem, and COD, with source identifiers, retrieval times, original units, and links between materials, chemicals, isotherms, and structures.</p>
                            <a class="button primary overview-cta" routerLink="/public-data/acquire">Acquire public data</a>
                        </div>
                        <div class="overview-totals" aria-label="Local public data counts">
                            <a routerLink="/public-data/adsorption"><strong>{{ overviewAdsorptionTotal() ?? '—' }}</strong><span>adsorption records</span></a>
                            <a routerLink="/public-data/materials"><strong>{{ overviewMaterialTotal() ?? '—' }}</strong><span>materials</span></a>
                            <a routerLink="/public-data/chemicals"><strong>{{ overviewChemicalTotal() ?? '—' }}</strong><span>chemicals</span></a>
                            <a routerLink="/public-data/structures"><strong>{{ overviewStructureTotal() ?? '—' }}</strong><span>structures</span></a>
                        </div>
                        <p class="acquisition-summary" aria-live="polite">Acquisition activity: {{ activeSyncRuns() }} active · {{ failedSyncRuns() }} failed · latest successful sync {{ formatDate(latestSyncAt()) }}</p>
                    </section>
                    <section class="public-section">
                        <div class="section-heading">
                            <h3>Integrated sources</h3>
                            <a class="text-link" routerLink="/public-data/sources">Inspect sources</a>
                        </div>
                        <div class="source-summary-grid">
                            @for (source of sources(); track source.key) {
                                <article class="source-summary">
                                    <div class="source-summary-head">
                                        <div><strong>{{ source.name }}</strong><span class="source-key">{{ source.key }}</span></div>
                                        <span class="source-status" [class.available]="source.health_status === 'available'" [class.unavailable]="source.health_status === 'unavailable'">{{ source.health_status }}</span>
                                    </div>
                                    <p>{{ source.description }}</p>
                                    <div class="capability-list">
                                        @for (capability of source.capabilities; track capability) { <span>{{ capability }}</span> }
                                    </div>
                                    <div class="source-summary-foot"><span>{{ source.record_count }} local source records</span><a [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: source.key }">Fetch data</a></div>
                                </article>
                            }
                        </div>
                    </section>
                }
                @case ('acquire') {
                    <adsmod-public-data-acquire />
                }
                @case ('adsorption') {
                    <section class="public-section">
                        <div class="section-heading"><div><h3>Adsorption data</h3><p>Filter locally stored isotherms; follow a material or adsorbate to its own record.</p></div></div>
                        @if (materialIdFilter() || adsorbateIdFilter()) {
                            <p class="filter-chip">Showing isotherms for {{ materialIdFilter() ? 'material' : 'adsorbate' }} <strong>{{ idFilterLabel() }}</strong><a class="text-link" [routerLink]="[]" [queryParams]="{ material_id: null, adsorbate_id: null, id: null }" queryParamsHandling="merge">Clear</a></p>
                        }
                        <div class="filter-bar">
                            <label>Source<select [(ngModel)]="adsorptionSource"><option value="">All sources</option>@for (source of sources(); track source.key) {<option [value]="source.key">{{ source.name }}</option>}</select></label>
                            <label>Material<input [(ngModel)]="adsorptionMaterial" placeholder="e.g. carbon, MOF-5" /></label>
                            <label>Adsorbate<input [(ngModel)]="adsorptionAdsorbate" placeholder="e.g. methane" /></label>
                            <label>Min temperature (K)<input [(ngModel)]="temperatureMin" type="number" min="0" /></label>
                            <label>Max temperature (K)<input [(ngModel)]="temperatureMax" type="number" min="0" /></label>
                            <div class="filter-actions"><button class="button primary" type="button" (click)="applyAdsorptionFilters()">Apply</button><button class="button secondary" type="button" (click)="resetAdsorptionFilters()">Reset</button></div>
                        </div>
                        <div class="table-frame">
                            <table>
                                <thead><tr><th>Source</th><th>Source ID</th><th>Material</th><th>Adsorbate</th><th>Temperature</th><th>Pressure range</th><th>Uptake range</th><th>Points</th><th></th></tr></thead>
                                <tbody>
                                    @for (item of adsorption()?.items ?? []; track item.id) {
                                        <tr>
                                            <td><span class="source-key">{{ item.source }}</span></td>
                                            <td class="truncate-cell" [title]="item.external_id">{{ item.external_id }}</td>
                                            <td class="truncate-cell" [title]="item.material"><a class="text-link" [routerLink]="['/public-data', 'materials']" [queryParams]="{ id: item.material_id }">{{ item.material }}</a></td>
                                            <td class="truncate-cell" [title]="adsorbateNames(item.adsorbates)">@for (adsorbate of item.adsorbates; track adsorbate.id; let last = $last) {<a class="text-link" [routerLink]="['/public-data', 'chemicals']" [queryParams]="{ id: adsorbate.id }">{{ adsorbate.name }}</a>@if (!last) {, }}</td>
                                            <td>{{ formatNumber(item.temperature_k) }} K</td>
                                            <td>{{ range(item.pressure_min_pa, item.pressure_max_pa, 'Pa') }}</td>
                                            <td>{{ range(item.uptake_min_mol_kg, item.uptake_max_mol_kg, 'mol/kg') }}</td>
                                            <td>{{ item.point_count }}</td>
                                            <td><button class="row-action" type="button" (click)="openDetail(item.id)">Inspect</button></td>
                                        </tr>
                                    } @empty {
                                        <tr><td colspan="9" class="empty-table">No adsorption records match the current filters. <a class="text-link" [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: 'nist' }">Fetch NIST isotherms</a></td></tr>
                                    }
                                </tbody>
                            </table>
                        </div>
                        <div class="pagination-row">{{ pageSummary(adsorption()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(adsorption()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (adsorptionDetail(); as detail) {
                        <aside class="detail-panel" aria-label="Adsorption record details">
                            <div class="detail-header"><div><p class="eyebrow">{{ detail.source }} · {{ detail.external_id }}</p><h3>{{ detail.material }} / {{ adsorbateNames(detail.adsorbates) }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="closeDetail()">×</button></div>
                            <div class="detail-links">
                                <a class="text-link" [routerLink]="['/public-data', 'materials']" [queryParams]="{ id: detail.material_id }">Material: {{ detail.material }}</a>
                                @for (adsorbate of detail.adsorbates; track adsorbate.id) {<a class="text-link" [routerLink]="['/public-data', 'chemicals']" [queryParams]="{ id: adsorbate.id }">Adsorbate: {{ adsorbate.name }}</a>}
                                @if (detail.doi) {<a class="text-link" [href]="'https://doi.org/' + detail.doi" target="_blank" rel="noopener noreferrer">DOI {{ detail.doi }}</a>}
                                @if (detail.source_url) {<a class="text-link" [href]="detail.source_url" target="_blank" rel="noopener noreferrer">Source record</a>}
                            </div>
                            <div class="detail-grid"><div><span>Temperature</span><strong>{{ formatNumber(detail.temperature_k) }} K</strong></div><div><span>Pressure basis</span><strong>{{ detail.pressure_basis }}</strong></div><div><span>Measurements</span><strong>{{ detail.measurements.length }}</strong></div><div><span>Retrieved</span><strong>{{ formatDate(detail.retrieved_at) }}</strong></div></div>
                            <div class="plot-panel">
                                <div class="plot-title"><strong>Isotherm</strong><span>Canonical pressure and uptake</span></div>
                                @if (detail.measurements.length > 1) {
                                    <svg viewBox="0 0 720 300" role="img" aria-label="Pressure versus uptake isotherm plot">
                                        <line x1="62" y1="24" x2="62" y2="250" class="plot-axis"/><line x1="62" y1="250" x2="700" y2="250" class="plot-axis"/>
                                        @for (series of plotSeries(); track series.adsorbate) {
                                            <polyline [attr.points]="series.points" class="plot-line" />
                                        }
                                        <text x="360" y="286" text-anchor="middle">Pressure (Pa)</text><text x="18" y="150" text-anchor="middle" transform="rotate(-90 18 150)">Uptake (mol/kg)</text>
                                    </svg>
                                } @else { <p class="empty-inline">At least two measurements are required for a plot.</p> }
                            </div>
                            <div class="detail-section"><h4>Provenance</h4>@for (identifier of detail.external_identifiers; track identifier.source + identifier.external_id) {<div class="provenance-row"><span>{{ identifier.source }}</span><code>{{ identifier.external_id }}</code>@if (identifier.source_url) {<a [href]="identifier.source_url" target="_blank" rel="noopener noreferrer">Open</a>}</div>}</div>
                        </aside>
                    }
                }
                @case ('materials') {
                    <section class="public-section">
                        <div class="section-heading"><div><h3>Materials</h3><p>One row per material identity; each lists its source records, isotherms, and linked crystal structures.</p></div></div>
                        <div class="filter-bar compact">
                            <label>Search<input [(ngModel)]="materialQuery" placeholder="Material name" /></label><label>Formula<input [(ngModel)]="materialFormula" placeholder="Formula" /></label><label>Source<select [(ngModel)]="materialSource"><option value="">All sources</option>@for (source of sources(); track source.key) {<option [value]="source.key">{{ source.name }}</option>}</select></label><label>Structure<select [(ngModel)]="materialStructure"><option value="">Any</option><option value="yes">Available</option><option value="no">Not available</option></select></label><div class="filter-actions"><button class="button primary" type="button" (click)="applyMaterialFilters()">Apply</button><button class="button secondary" type="button" (click)="resetMaterialFilters()">Reset</button></div>
                        </div>
                        <div class="table-frame"><table><thead><tr><th>Material</th><th>Formula</th><th>Molar mass</th><th>Sources</th><th>Isotherms</th><th>Structures</th><th></th></tr></thead><tbody>@for (item of materials()?.items ?? []; track item.id) {<tr><td class="truncate-cell" [title]="item.name">{{ item.name }}</td><td>{{ item.formula || '—' }}</td><td>{{ item.molar_mass_g_mol === null ? '—' : formatNumber(item.molar_mass_g_mol) + ' g/mol' }}</td><td>{{ sourceKeys(item.external_identifiers) }}</td><td>{{ item.isotherm_count }}</td><td>{{ item.structure_count }}</td><td><button class="row-action" type="button" (click)="openDetail(item.id)">Inspect</button></td></tr>} @empty {<tr><td colspan="7" class="empty-table">No materials match the current filters. <a class="text-link" [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: 'nist' }">Fetch NIST materials</a></td></tr>}</tbody></table></div>
                        <div class="pagination-row">{{ pageSummary(materials()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(materials()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (materialDetail(); as detail) {
                        <aside class="detail-panel" aria-label="Material details">
                            <div class="detail-header"><div><p class="eyebrow">Material record</p><h3>{{ detail.name }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="closeDetail()">×</button></div>
                            <div class="detail-grid"><div><span>Formula</span><strong>{{ detail.formula || '—' }}</strong></div><div><span>Molar mass</span><strong>{{ detail.molar_mass_g_mol === null ? '—' : formatNumber(detail.molar_mass_g_mol) + ' g/mol' }}</strong></div><div><span>Isotherms</span><strong>{{ detail.isotherm_count }}</strong></div><div><span>Structures</span><strong>{{ detail.structure_count }}</strong></div></div>
                            <div class="detail-links">
                                @if (detail.isotherm_count) {<a class="button secondary" [routerLink]="['/public-data', 'adsorption']" [queryParams]="{ material_id: detail.id }">View {{ detail.isotherm_count }} isotherms</a>}
                                <a class="button secondary" [routerLink]="['/public-data', 'acquire']" [queryParams]="codSearchParams(detail)">Find COD structures</a>
                            </div>
                            <div class="detail-section"><h4>Linked structures</h4>
                                @for (structure of detail.structures; track structure.id) {
                                    <div class="provenance-row"><span>{{ structure.source }}</span><code>{{ structure.external_id }} · {{ structure.formula || 'no formula' }} · {{ structure.space_group || 'no space group' }}</code><a [routerLink]="['/public-data', 'structures']" [queryParams]="{ id: structure.id }">Open</a></div>
                                } @empty {<p class="empty-inline">No crystal structure is linked to this material yet.</p>}
                            </div>
                            <div class="detail-section"><h4>Provenance</h4>@for (identifier of detail.external_identifiers; track identifier.source + identifier.external_id) {<div class="provenance-row"><span>{{ identifier.source }}</span><code>{{ identifier.external_id }}</code>@if (identifier.source_url) {<a [href]="identifier.source_url" target="_blank" rel="noopener noreferrer">Open</a>}</div>} @empty {<p class="empty-inline">No source record of its own; it is known through the isotherms or datasets that use it.</p>}</div>
                        </aside>
                    }
                }
                @case ('chemicals') {
                    <section class="public-section">
                        <div class="section-heading"><div><h3>Chemicals & adsorbates</h3><p>Search stored chemicals by name or any PubChem synonym. Exact identifiers decide identity; names never merge records.</p></div><a class="button secondary" [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: 'pubchem' }">Resolve on PubChem</a></div>
                        <div class="filter-bar"><label>Search<input [(ngModel)]="chemicalQuery" placeholder="Name or synonym" /></label><label>Formula<input [(ngModel)]="chemicalFormula" placeholder="Formula" /></label><label>Source<select [(ngModel)]="chemicalSource"><option value="">All sources</option>@for (source of sources(); track source.key) {<option [value]="source.key">{{ source.name }}</option>}</select></label><label>Min mass<input [(ngModel)]="chemicalMassMin" type="number" min="0" /></label><label>Max mass<input [(ngModel)]="chemicalMassMax" type="number" min="0" /></label><div class="filter-actions"><button class="button primary" type="button" (click)="applyChemicalFilters()">Apply</button><button class="button secondary" type="button" (click)="resetChemicalFilters()">Reset</button></div></div>
                        <div class="table-frame"><table><thead><tr><th>Name</th><th>Formula</th><th>Molecular weight</th><th>InChIKey</th><th>PubChem CID</th><th>Sources</th><th></th></tr></thead><tbody>@for (item of chemicals()?.items ?? []; track item.id) {<tr><td class="truncate-cell" [title]="item.name">{{ item.name }}</td><td>{{ item.formula || '—' }}</td><td>{{ item.molecular_weight === null ? '—' : formatNumber(item.molecular_weight) + ' g/mol' }}</td><td class="mono truncate-cell" [title]="item.inchi_key || ''">{{ item.inchi_key || '—' }}</td><td>{{ item.pubchem_cid || '—' }}</td><td>{{ sourceKeys(item.external_identifiers) }}</td><td><button class="row-action" type="button" (click)="openDetail(item.id)">Inspect</button></td></tr>} @empty {<tr><td colspan="7" class="empty-table">No chemicals match the current filters. <a class="text-link" [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: 'pubchem' }">Resolve a compound</a></td></tr>}</tbody></table></div>
                        <div class="pagination-row">{{ pageSummary(chemicals()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(chemicals()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (chemicalDetail(); as detail) {
                        <aside class="detail-panel" aria-label="Chemical details"><div class="detail-header"><div><p class="eyebrow">Chemical record</p><h3>{{ detail.preferred_name || detail.name }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="closeDetail()">×</button></div>
                            <div class="detail-links">
                                @if (detail.isotherm_count) {<a class="button secondary" [routerLink]="['/public-data', 'adsorption']" [queryParams]="{ adsorbate_id: detail.id }">View {{ detail.isotherm_count }} isotherms</a>}
                                @if (!detail.pubchem_cid) {<a class="button secondary" [routerLink]="['/public-data', 'acquire']" [queryParams]="pubchemResolveParams(detail)">Resolve on PubChem</a>}
                            </div>
                            <div class="chemical-detail-layout">@if (detail.structure_2d_url) {<div class="molecule-preview"><img [src]="detail.structure_2d_url" [alt]="'2D molecular structure for ' + detail.name"/></div>}<div class="detail-grid"><div><span>Formula</span><strong>{{ detail.formula || '—' }}</strong></div><div><span>Molecular weight</span><strong>{{ detail.molecular_weight === null ? '—' : formatNumber(detail.molecular_weight) + ' g/mol' }}</strong></div><div><span>PubChem CID</span><strong>{{ detail.pubchem_cid || '—' }}</strong></div><div><span>3D conformer</span><strong>{{ detail.conformer_3d_url ? 'Available' : 'Not reported' }}</strong></div></div></div>
                            <div class="detail-section"><h4>Stable identifiers</h4><dl class="identifier-list"><dt>InChIKey</dt><dd>{{ detail.inchi_key || '—' }}</dd><dt>InChI</dt><dd>{{ detail.inchi || '—' }}</dd><dt>SMILES</dt><dd>{{ detail.smiles || '—' }}</dd><dt>Connectivity SMILES</dt><dd>{{ detail.connectivity_smiles || '—' }}</dd></dl></div>
                            <div class="detail-section"><h4>Physicochemical properties</h4><div class="property-grid">@for (property of detail.properties; track property.key) {<div><span>{{ property.key }}</span><strong>{{ property.value_number ?? property.value_text }}{{ property.unit ? ' ' + property.unit : '' }}</strong></div>} @empty {<p class="empty-inline">No additional normalized properties are stored.</p>}</div></div>
                            <div class="detail-section"><h4>Synonyms</h4><p class="wrap-list">{{ detail.synonyms.slice(0, 20).join(' · ') || 'No synonyms stored.' }}</p></div>
                            <div class="detail-section"><h4>Provenance</h4>@for (identifier of detail.external_identifiers; track identifier.source + identifier.external_id) {<div class="provenance-row"><span>{{ identifier.source }}</span><code>{{ identifier.external_id }}</code>@if (identifier.source_url) {<a [href]="identifier.source_url" target="_blank" rel="noopener noreferrer">Open</a>}</div>} @empty {<p class="empty-inline">No source record of its own; it is known through the isotherms or datasets that use it.</p>}</div>
                        </aside>
                    }
                }
                @case ('structures') {
                    <section class="public-section">
                        <div class="section-heading"><div><h3>Structures</h3><p>Imported COD structures with their original CIF, unit cell, and atom sites. Link a structure to the material it describes.</p></div><a class="button secondary" [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: 'cod' }">Search COD</a></div>
                        @if (structureMaterialFilter()) {
                            <p class="filter-chip">Showing structures linked to material #{{ structureMaterialFilter() }}<a class="text-link" [routerLink]="[]" [queryParams]="{ material_id: null, id: null }" queryParamsHandling="merge">Clear</a></p>
                        }
                        <div class="filter-bar compact"><label>Search<input [(ngModel)]="structureQuery" placeholder="Name or formula" /></label><label>Source<select [(ngModel)]="structureSource"><option value="">All sources</option>@for (source of sources(); track source.key) {<option [value]="source.key">{{ source.name }}</option>}</select></label><label>Link status<select [(ngModel)]="structureLink"><option value="">Any</option><option value="linked">Linked to material</option><option value="unlinked">Unlinked</option></select></label><div></div><div class="filter-actions"><button class="button primary" type="button" (click)="applyStructureFilters()">Apply</button><button class="button secondary" type="button" (click)="resetStructureFilters()">Reset</button></div></div>
                        <div class="table-frame"><table><thead><tr><th>Source</th><th>Source ID</th><th>Material</th><th>Formula</th><th>Space group</th><th>Atoms</th><th>Retrieved</th><th></th></tr></thead><tbody>@for (item of structures()?.items ?? []; track item.id) {<tr><td><span class="source-key">{{ item.source }}</span></td><td>{{ item.external_id }}</td><td class="truncate-cell">@if (item.material_id) {<a class="text-link" [routerLink]="['/public-data', 'materials']" [queryParams]="{ id: item.material_id }">{{ item.material_name }}</a>} @else {Unlinked}</td><td class="truncate-cell" [title]="item.formula || ''">{{ item.formula || '—' }}</td><td>{{ item.space_group || '—' }}</td><td>{{ item.atom_count }}</td><td>{{ formatDate(item.retrieved_at) }}</td><td><button class="row-action" type="button" (click)="openDetail(item.id)">Inspect</button></td></tr>} @empty {<tr><td colspan="8" class="empty-table">No imported structures match the current filters. <a class="text-link" [routerLink]="['/public-data', 'acquire']" [queryParams]="{ panel: 'cod' }">Search COD</a></td></tr>}</tbody></table></div>
                        <div class="pagination-row">{{ pageSummary(structures()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(structures()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (structureDetail(); as detail) {
                        <aside class="detail-panel" aria-label="Structure details">
                            <div class="detail-header"><div><p class="eyebrow">{{ detail.source }} · {{ detail.external_id }}</p><h3>{{ detail.name || detail.formula || 'Structure record' }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="closeDetail()">×</button></div>
                            <div class="detail-grid"><div><span>Formula</span><strong>{{ detail.formula || '—' }}</strong></div><div><span>Space group</span><strong>{{ detail.space_group || '—' }}</strong></div><div><span>Unit-cell volume</span><strong>{{ detail.cell_volume_angstrom3 === null ? '—' : formatNumber(detail.cell_volume_angstrom3) + ' Å³' }}</strong></div><div><span>Normalized atoms</span><strong>{{ detail.atom_count }}</strong></div></div>
                            <div class="detail-section"><h4>Material</h4>
                                @if (detail.material_id) {
                                    <p class="link-row"><a class="text-link" [routerLink]="['/public-data', 'materials']" [queryParams]="{ id: detail.material_id }">{{ detail.material_name }}</a><button class="button secondary" type="button" [disabled]="linkBusy()" (click)="linkStructure(detail.id, null)">Unlink</button></p>
                                } @else {
                                    <div class="link-row"><adsmod-material-picker [disabled]="linkBusy()" (selected)="structureLinkCandidate.set($event)" /><button class="button primary" type="button" [disabled]="linkBusy() || !structureLinkCandidate()" (click)="linkStructure(detail.id, structureLinkCandidate()?.id ?? null)">Link</button></div>
                                }
                            </div>
                            <div class="detail-section"><h4>Unit cell</h4><p>a {{ valueOrDash(detail.cell_a_angstrom) }} Å · b {{ valueOrDash(detail.cell_b_angstrom) }} Å · c {{ valueOrDash(detail.cell_c_angstrom) }} Å</p><p>α {{ valueOrDash(detail.cell_alpha_deg) }}° · β {{ valueOrDash(detail.cell_beta_deg) }}° · γ {{ valueOrDash(detail.cell_gamma_deg) }}°</p></div>
                            <div class="detail-section"><h4>Atomic coordinates</h4><div class="atom-list">@for (atom of detail.atoms.slice(0, 50); track atom.sequence_index) {<code>{{ atom.label }} {{ atom.element }} ({{ formatNumber(atom.fractional_x) }}, {{ formatNumber(atom.fractional_y) }}, {{ formatNumber(atom.fractional_z) }})</code>} @empty {<p class="empty-inline">No fractional atom coordinates were normalized from this CIF.</p>}</div>@if (detail.atoms.length > 50) {<p class="detail-note">Showing the first 50 of {{ detail.atoms.length }} atom sites.</p>}</div>
                            <div class="detail-section"><h4>Reference</h4>@if (detail.doi) {<p><a class="text-link" [href]="'https://doi.org/' + detail.doi" target="_blank" rel="noopener noreferrer">DOI {{ detail.doi }}</a>{{ detail.reference_title ? ' · ' + detail.reference_title : '' }}</p>} @else {<p class="empty-inline">No publication reference was supplied by the source.</p>}</div>
                            <div class="detail-links"><a class="button secondary" [href]="structureFile(detail.id)" download>Download {{ detail.format.toUpperCase() }}</a>@if (detail.source_url) {<a class="button secondary" [href]="detail.source_url" target="_blank" rel="noopener noreferrer">Open source record</a>}</div>
                        </aside>
                    }
                }
                @case ('sources') {
                    <section class="public-section">
                        <div class="section-heading"><div><h3>Sources</h3><p>Provider availability is independent. The status shown here is the last saved check; refreshing queues a bounded background check. Fetch data from the Acquire view.</p></div><div class="filter-actions"><a class="button secondary" routerLink="/public-data/acquire">Open Acquire</a><button class="button secondary" type="button" [disabled]="sourceHealthLoading()" (click)="refreshSourceHealth()">{{ sourceHealthLoading() ? 'Checking…' : 'Refresh status' }}</button></div></div>
                        <div class="source-table">
                            @for (source of sources(); track source.key) {
                                <article class="source-row"><div class="source-health"><span class="health-dot" [class.available]="source.health_status === 'available'" [class.degraded]="source.health_status === 'degraded'" [class.unavailable]="source.health_status === 'unavailable'"></span><div><strong>{{ source.name }}</strong><code>{{ source.key }}</code></div><span>{{ source.health_status === 'unknown' ? 'Not checked' : source.health_status }}</span></div><p>{{ source.description }}</p><div class="capability-list">@for (capability of source.capabilities; track capability) {<span>{{ capability }}</span>}</div><div class="source-meta"><span>{{ source.record_count }} stored source records</span><span>Checked {{ formatDate(source.last_health_checked_at) }}{{ source.health_fresh ? '' : source.last_health_checked_at ? ' · stale' : '' }}</span><span>Last sync {{ formatDate(source.last_successful_sync_at) }}</span>@if (source.license_name) {<span>{{ source.license_name }}</span>}</div><div class="source-actions"><a [href]="source.homepage_url" target="_blank" rel="noopener noreferrer">Source</a>@if (source.license_url) {<a [href]="source.license_url" target="_blank" rel="noopener noreferrer">License</a>}</div>@if (source.health_detail) {<p class="source-detail">{{ source.health_detail }}</p>}</article>
                            }
                        </div>
                    </section>
                }
            }
        </div>
    `,
    styleUrl: './public-data-workspace.component.css',
})
export class PublicDataWorkspaceComponent {
    private readonly route = inject(ActivatedRoute);
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);
    private loadSequence = 0;
    private detailSequence = 0;
    private sourceHealthSequence = 0;
    private listKey = '';

    protected readonly views = VIEWS;
    protected readonly view = signal<PublicDataView>('overview');
    protected readonly loading = signal(false);
    protected readonly sourceHealthLoading = signal(false);
    protected readonly linkBusy = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly sources = signal<PublicSourceSummary[]>([]);
    protected readonly adsorption = signal<AdsorptionPageResponse | null>(null);
    protected readonly adsorptionDetail = signal<AdsorptionDetailResponse | null>(null);
    protected readonly materials = signal<MaterialPageResponse | null>(null);
    protected readonly materialDetail = signal<MaterialDetailResponse | null>(null);
    protected readonly chemicals = signal<ChemicalPageResponse | null>(null);
    protected readonly chemicalDetail = signal<ChemicalRecordView | null>(null);
    protected readonly structures = signal<StructurePageResponse | null>(null);
    protected readonly structureDetail = signal<StructureRecordView | null>(null);
    protected readonly structureLinkCandidate = signal<MaterialRecordView | null>(null);
    protected readonly materialIdFilter = signal<number | null>(null);
    protected readonly adsorbateIdFilter = signal<number | null>(null);
    protected readonly structureMaterialFilter = signal<number | null>(null);
    protected readonly page = signal(1);
    protected readonly pageSize = 25;

    protected readonly overviewAdsorptionTotal = signal<number | null>(null);
    protected readonly overviewMaterialTotal = signal<number | null>(null);
    protected readonly overviewChemicalTotal = signal<number | null>(null);
    protected readonly overviewStructureTotal = signal<number | null>(null);
    protected readonly activeSyncRuns = signal(0);
    protected readonly failedSyncRuns = signal(0);
    protected readonly latestSyncAt = signal<string | null>(null);

    protected adsorptionSource = '';
    protected adsorptionMaterial = '';
    protected adsorptionAdsorbate = '';
    protected temperatureMin: number | null = null;
    protected temperatureMax: number | null = null;
    protected materialQuery = '';
    protected materialFormula = '';
    protected materialSource = '';
    protected materialStructure = '';
    protected chemicalQuery = '';
    protected chemicalFormula = '';
    protected chemicalSource = '';
    protected chemicalMassMin: number | null = null;
    protected chemicalMassMax: number | null = null;
    protected structureQuery = '';
    protected structureSource = '';
    protected structureLink = '';

    protected readonly idFilterLabel = computed(() => {
        const first = this.adsorption()?.items[0];
        const materialId = this.materialIdFilter();
        if (materialId) return first?.material_id === materialId ? first.material : `#${materialId}`;
        const adsorbateId = this.adsorbateIdFilter();
        return first?.adsorbates.find((item) => item.id === adsorbateId)?.name ?? `#${adsorbateId}`;
    });

    protected readonly plotSeries = computed(() => {
        const measurements = this.adsorptionDetail()?.measurements ?? [];
        if (measurements.length < 2) return [];
        const pressures = measurements.map((item) => item.pressure_pa);
        const uptakes = measurements.map((item) => item.uptake_mol_kg);
        const minX = Math.min(...pressures);
        const maxX = Math.max(...pressures);
        const minY = Math.min(...uptakes);
        const maxY = Math.max(...uptakes);
        const spanX = Math.max(maxX - minX, Number.EPSILON);
        const spanY = Math.max(maxY - minY, Number.EPSILON);
        const grouped = new Map<string, typeof measurements>();
        for (const item of measurements) {
            const series = grouped.get(item.adsorbate) ?? [];
            series.push(item);
            grouped.set(item.adsorbate, series);
        }
        return [...grouped.entries()].map(([adsorbate, series]) => ({
            adsorbate,
            points: series
                .sort((left, right) => left.pressure_pa - right.pressure_pa)
                .map((item) => {
                    const x = 62 + ((item.pressure_pa - minX) / spanX) * 638;
                    const y = 250 - ((item.uptake_mol_kg - minY) / spanY) * 226;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(' '),
        }));
    });

    constructor() {
        // Path and query parameters emit separately within one navigation; settle
        // them first so a new view never pairs with the previous view's record id.
        combineLatest([this.route.paramMap, this.route.queryParamMap])
            .pipe(debounceTime(0), takeUntilDestroyed(this.destroyRef))
            .subscribe(([params, query]) => this.applyRoute(params, query));
    }

    protected viewLabel(view: PublicDataView): string {
        return VIEW_LABELS[view];
    }

    protected async reload(): Promise<void> {
        await this.loadCurrent();
    }

    private applyRoute(params: ParamMap, query: ParamMap): void {
        const rawView = params.get('view') ?? 'overview';
        if (!VIEWS.includes(rawView as PublicDataView)) {
            void this.router.navigate(['/public-data', 'overview'], { replaceUrl: true });
            return;
        }
        const nextView = rawView as PublicDataView;
        const source = query.get('source') ?? '';
        const materialId = positiveId(query.get('material_id'));
        const adsorbateId = positiveId(query.get('adsorbate_id'));
        const listKey = JSON.stringify([nextView, source, materialId, adsorbateId]);
        if (listKey !== this.listKey) {
            const viewChanged = nextView !== this.view() || !this.listKey;
            this.listKey = listKey;
            this.view.set(nextView);
            this.page.set(1);
            if (viewChanged) this.closeDetails();
            this.applyLinkFilters(nextView, source, materialId, adsorbateId);
            void this.loadCurrent();
        }
        void this.loadDetail(nextView, positiveId(query.get('id')));
    }

    private applyLinkFilters(view: PublicDataView, source: string, materialId: number | null, adsorbateId: number | null): void {
        this.materialIdFilter.set(view === 'adsorption' ? materialId : null);
        this.adsorbateIdFilter.set(view === 'adsorption' ? adsorbateId : null);
        this.structureMaterialFilter.set(view === 'structures' ? materialId : null);
        if (!source) return;
        if (view === 'adsorption') this.adsorptionSource = source;
        if (view === 'materials') this.materialSource = source;
        if (view === 'chemicals') this.chemicalSource = source;
        if (view === 'structures') this.structureSource = source;
    }

    private isCurrentLoad(requestId: number, expectedView: PublicDataView): boolean {
        return requestId === this.loadSequence && this.view() === expectedView;
    }

    private async ensureSources(requestId: number, expectedView: PublicDataView): Promise<void> {
        if (this.sources().length) return;
        const result = await fetchPublicSources();
        if (!this.isCurrentLoad(requestId, expectedView)) return;
        if (result.data) this.sources.set(result.data.sources);
        else this.error.set(result.error);
    }

    private async loadCurrent(): Promise<void> {
        const requestId = ++this.loadSequence;
        const currentView = this.view();
        if (currentView !== 'sources') {
            this.sourceHealthSequence += 1;
            this.sourceHealthLoading.set(false);
        }
        this.error.set(null);
        if (currentView === 'acquire') return;
        this.loading.set(true);
        try {
            const sourcesLoad = this.ensureSources(requestId, currentView);
            switch (currentView) {
                case 'overview': await Promise.all([sourcesLoad, this.loadOverview(requestId)]); break;
                case 'adsorption': await Promise.all([sourcesLoad, this.loadAdsorption(requestId)]); break;
                case 'materials': await Promise.all([sourcesLoad, this.loadMaterials(requestId)]); break;
                case 'chemicals': await Promise.all([sourcesLoad, this.loadChemicals(requestId)]); break;
                case 'structures': await Promise.all([sourcesLoad, this.loadStructures(requestId)]); break;
                case 'sources': await this.refreshSources(); break;
            }
        } finally {
            if (this.isCurrentLoad(requestId, currentView)) this.loading.set(false);
        }
    }

    private async loadOverview(requestId: number): Promise<void> {
        this.setOverviewTotals(null);
        const result = await fetchPublicOverview();
        if (!this.isCurrentLoad(requestId, 'overview')) return;
        this.setOverviewTotals(result.data);
        this.error.set(result.error);
    }

    private setOverviewTotals(overview: PublicDataOverviewResponse | null): void {
        this.overviewAdsorptionTotal.set(overview?.adsorption_records ?? null);
        this.overviewMaterialTotal.set(overview?.materials ?? null);
        this.overviewChemicalTotal.set(overview?.chemicals ?? null);
        this.overviewStructureTotal.set(overview?.structures ?? null);
        this.activeSyncRuns.set(overview?.active_sync_runs ?? 0);
        this.failedSyncRuns.set(overview?.failed_sync_runs ?? 0);
        this.latestSyncAt.set(overview?.latest_successful_sync_at ?? null);
    }

    private async loadAdsorption(requestId = this.loadSequence): Promise<void> {
        const result = await fetchPublicAdsorption({
            page: this.page(),
            page_size: this.pageSize,
            source: this.adsorptionSource,
            material: this.adsorptionMaterial,
            adsorbate: this.adsorptionAdsorbate,
            material_id: this.materialIdFilter(),
            adsorbate_id: this.adsorbateIdFilter(),
            temperature_min_k: this.temperatureMin,
            temperature_max_k: this.temperatureMax,
        });
        if (!this.isCurrentLoad(requestId, 'adsorption')) return;
        this.adsorption.set(result.data);
        this.error.set(result.error);
    }

    private async loadMaterials(requestId = this.loadSequence): Promise<void> {
        const hasStructure = this.materialStructure === 'yes' ? true : this.materialStructure === 'no' ? false : null;
        const result = await fetchPublicMaterials({ page: this.page(), page_size: this.pageSize, q: this.materialQuery, formula: this.materialFormula, source: this.materialSource, has_structure: hasStructure });
        if (!this.isCurrentLoad(requestId, 'materials')) return;
        this.materials.set(result.data);
        this.error.set(result.error);
    }

    private async loadChemicals(requestId = this.loadSequence): Promise<void> {
        const result = await fetchPublicChemicals({ page: this.page(), page_size: this.pageSize, q: this.chemicalQuery, formula: this.chemicalFormula, source: this.chemicalSource, molecular_weight_min: this.chemicalMassMin, molecular_weight_max: this.chemicalMassMax });
        if (!this.isCurrentLoad(requestId, 'chemicals')) return;
        this.chemicals.set(result.data);
        this.error.set(result.error);
    }

    private async loadStructures(requestId = this.loadSequence): Promise<void> {
        const linked = this.structureLink === 'linked' ? true : this.structureLink === 'unlinked' ? false : null;
        const result = await fetchPublicStructures({ page: this.page(), page_size: this.pageSize, q: this.structureQuery, source: this.structureSource, linked_only: linked, material_id: this.structureMaterialFilter() });
        if (!this.isCurrentLoad(requestId, 'structures')) return;
        this.structures.set(result.data);
        this.error.set(result.error);
    }

    protected async applyAdsorptionFilters(): Promise<void> { this.page.set(1); await this.loadAdsorption(); }
    protected async resetAdsorptionFilters(): Promise<void> {
        this.adsorptionSource = ''; this.adsorptionMaterial = ''; this.adsorptionAdsorbate = ''; this.temperatureMin = null; this.temperatureMax = null;
        if (this.materialIdFilter() || this.adsorbateIdFilter()) {
            await this.router.navigate([], { relativeTo: this.route, queryParams: { material_id: null, adsorbate_id: null, source: null, id: null }, queryParamsHandling: 'merge' });
            return;
        }
        await this.applyAdsorptionFilters();
    }
    protected async applyMaterialFilters(): Promise<void> { this.page.set(1); await this.loadMaterials(); }
    protected async resetMaterialFilters(): Promise<void> { this.materialQuery = ''; this.materialFormula = ''; this.materialSource = ''; this.materialStructure = ''; await this.applyMaterialFilters(); }
    protected async applyChemicalFilters(): Promise<void> { this.page.set(1); await this.loadChemicals(); }
    protected async resetChemicalFilters(): Promise<void> { this.chemicalQuery = ''; this.chemicalFormula = ''; this.chemicalSource = ''; this.chemicalMassMin = null; this.chemicalMassMax = null; await this.applyChemicalFilters(); }
    protected async applyStructureFilters(): Promise<void> { this.page.set(1); await this.loadStructures(); }
    protected async resetStructureFilters(): Promise<void> {
        this.structureQuery = ''; this.structureSource = ''; this.structureLink = '';
        if (this.structureMaterialFilter()) {
            await this.router.navigate([], { relativeTo: this.route, queryParams: { material_id: null, source: null, id: null }, queryParamsHandling: 'merge' });
            return;
        }
        await this.applyStructureFilters();
    }

    protected async previousPage(): Promise<void> { if (this.page() > 1) { this.page.update((value) => value - 1); await this.loadCurrent(); } }
    protected async nextPage(): Promise<void> { this.page.update((value) => value + 1); await this.loadCurrent(); }

    protected openDetail(id: number): void {
        void this.router.navigate([], { relativeTo: this.route, queryParams: { id }, queryParamsHandling: 'merge' });
    }

    protected closeDetail(): void {
        void this.router.navigate([], { relativeTo: this.route, queryParams: { id: null }, queryParamsHandling: 'merge' });
    }

    private async loadDetail(view: PublicDataView, id: number | null): Promise<void> {
        const requestId = ++this.detailSequence;
        if (id === null) {
            this.closeDetails();
            return;
        }
        const current = { adsorption: this.adsorptionDetail(), materials: this.materialDetail(), chemicals: this.chemicalDetail(), structures: this.structureDetail() } as Record<string, { id: number } | null>;
        if (current[view]?.id === id) return;
        this.structureLinkCandidate.set(null);
        switch (view) {
            case 'adsorption': await this.setDetail(requestId, fetchPublicAdsorptionDetail(id), (data) => this.adsorptionDetail.set(data)); break;
            case 'materials': await this.setDetail(requestId, fetchPublicMaterialDetail(id), (data) => this.materialDetail.set(data)); break;
            case 'chemicals': await this.setDetail(requestId, fetchPublicChemicalDetail(id), (data) => this.chemicalDetail.set(data)); break;
            case 'structures': await this.setDetail(requestId, fetchPublicStructureDetail(id), (data) => this.structureDetail.set(data)); break;
        }
    }

    private async setDetail<T>(requestId: number, request: Promise<{ data: T | null; error: string | null }>, assign: (data: T | null) => void): Promise<void> {
        const result = await request;
        if (requestId !== this.detailSequence) return;
        assign(result.data);
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

    protected codSearchParams(material: MaterialDetailResponse): Record<string, string | number> {
        return {
            panel: 'cod',
            material_id: material.id,
            material_name: material.name,
            ...(material.formula ? { formula: material.formula } : {}),
        };
    }

    protected pubchemResolveParams(chemical: ChemicalRecordView): Record<string, string | number> {
        return {
            panel: 'pubchem',
            query: chemical.inchi_key || chemical.name,
            adsorbate_id: chemical.id,
            adsorbate_name: chemical.name,
        };
    }

    private async refreshSources(): Promise<void> {
        const result = await fetchPublicSources();
        if (result.data) this.sources.set(result.data.sources);
        else this.error.set(result.error);
    }

    protected async refreshSourceHealth(): Promise<void> {
        const requestId = ++this.sourceHealthSequence;
        const isCurrent = (): boolean => requestId === this.sourceHealthSequence && this.view() === 'sources';
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
        this.adsorptionDetail.set(null);
        this.materialDetail.set(null);
        this.chemicalDetail.set(null);
        this.structureDetail.set(null);
        this.structureLinkCandidate.set(null);
    }

    protected adsorbateNames(adsorbates: { name: string }[]): string { return adsorbates.map((item) => item.name).join(', '); }
    protected sourceKeys(identifiers: { source: string }[]): string { return [...new Set(identifiers.map((item) => item.source))].join(', ') || '—'; }
    protected formatNumber(value: number): string { return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(value); }
    protected valueOrDash(value: number | null): string { return value === null ? '—' : this.formatNumber(value); }
    protected formatDate(value: string | null): string { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date); }
    protected range(min: number | null, max: number | null, unit: string): string { return min === null || max === null ? '—' : `${this.formatNumber(min)}–${this.formatNumber(max)} ${unit}`; }
    protected pageSummary(pagination: { page: number; page_size: number; total: number } | undefined): string { if (!pagination || pagination.total === 0) return '0 records'; const first = (pagination.page - 1) * pagination.page_size + 1; const last = Math.min(pagination.total, pagination.page * pagination.page_size); return `${first}–${last} of ${pagination.total}`; }
    protected hasNextPage(pagination: { page: number; page_size: number; total: number } | undefined): boolean { return !!pagination && pagination.page * pagination.page_size < pagination.total; }
}
