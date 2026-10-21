// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, effect, input, output, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { AdsorptionDetailResponse, ChemicalRecordView, MaterialDetailResponse } from '../../../models/public-data.model';
import { fetchPublicAdsorptionDetail, fetchPublicChemicalDetail, fetchPublicMaterialDetail } from '../../../services/public-data.service';
import { formatValue, pressureLabel } from '../dashboard-charts';

@Component({
    selector: 'adsmod-explore-inspector',
    standalone: true,
    imports: [RouterLink],
    styleUrl: '../dashboards.css',
    host: { '(document:keydown.escape)': 'closed.emit()' },
    template: `
        <aside class="detail-panel explore-inspector" aria-label="Isotherm details" aria-live="polite">
            <div class="detail-header">
                <div>
                    <p class="eyebrow">{{ isotherm()?.source ?? 'Isotherm' }} · {{ isotherm()?.external_id ?? '#' + isothermId() }}</p>
                    <h3>{{ isotherm()?.material ?? 'Loading…' }}</h3>
                </div>
                <button class="detail-close" type="button" aria-label="Close details" (click)="closed.emit()">×</button>
            </div>
            @if (error()) {
                <p class="inline-error" role="alert">{{ error() }}</p>
            }
            @if (isotherm(); as detail) {
                <div class="detail-grid">
                    <div><span>Temperature</span><strong>{{ format(detail.temperature_k) }} K</strong></div>
                    <div><span>Pressure basis</span><strong>{{ detail.pressure_basis }}</strong></div>
                    <div><span>Measurements</span><strong>{{ detail.measurements.length }}</strong></div>
                    <div><span>Pressure range</span><strong>{{ format(detail.pressure_min) }}–{{ format(detail.pressure_max) }} {{ unitLabel(detail.pressure_unit) }}</strong></div>
                </div>
                <div class="detail-links">
                    <a class="text-link" [routerLink]="['/public-data', 'adsorption']" [queryParams]="{ id: detail.id }">Open adsorption record</a>
                    @if (detail.doi) {<a class="text-link" [href]="'https://doi.org/' + detail.doi" target="_blank" rel="noopener noreferrer">DOI {{ detail.doi }}</a>}
                    @if (detail.source_url) {<a class="text-link" [href]="detail.source_url" target="_blank" rel="noopener noreferrer">Source record</a>}
                </div>
            }
            @if (material(); as item) {
                <section class="detail-section">
                    <h4>Material</h4>
                    <dl class="property-list">
                        <div><dt>Name</dt><dd>{{ item.name }}</dd></div>
                        <div><dt>Formula</dt><dd>{{ item.formula ?? '—' }}</dd></div>
                        <div><dt>Molar mass</dt><dd>{{ item.molar_mass_g_mol === null ? '—' : format(item.molar_mass_g_mol) + ' g/mol' }}</dd></div>
                        <div><dt>Isotherms</dt><dd>{{ item.isotherm_count }}</dd></div>
                    </dl>
                    @for (structure of item.structures.slice(0, 3); track structure.id) {
                        <p class="structure-line">
                            <span class="source-key">{{ structure.source }}</span>
                            {{ structure.formula ?? structure.name ?? structure.external_id }}
                            @if (structure.space_group) { · {{ structure.space_group }} }
                            @if (structure.cell_volume_angstrom3 !== null) { · {{ format(structure.cell_volume_angstrom3) }} Å³ }
                        </p>
                    } @empty {
                        <p class="empty-inline">No crystal structure is linked to this material.</p>
                    }
                    <a class="text-link" [routerLink]="['/public-data', 'materials']" [queryParams]="{ id: item.id }">Open material record</a>
                </section>
            }
            @for (chemical of chemicals(); track chemical.id) {
                <section class="detail-section">
                    <h4>Adsorbate · {{ chemical.preferred_name ?? chemical.name }}</h4>
                    @if (chemical.structure_2d_url) {
                        <img class="molecule-preview" [src]="chemical.structure_2d_url" [alt]="'2D structure of ' + chemical.name" loading="lazy" />
                    }
                    <dl class="property-list">
                        <div><dt>Formula</dt><dd>{{ chemical.formula ?? '—' }}</dd></div>
                        <div><dt>Molecular weight</dt><dd>{{ chemical.molecular_weight === null ? '—' : format(chemical.molecular_weight) + ' g/mol' }}</dd></div>
                        <div><dt>InChIKey</dt><dd><code>{{ chemical.inchi_key ?? '—' }}</code></dd></div>
                        <div><dt>SMILES</dt><dd><code>{{ chemical.smiles ?? chemical.connectivity_smiles ?? '—' }}</code></dd></div>
                        <div><dt>PubChem CID</dt><dd>{{ chemical.pubchem_cid ?? '—' }}</dd></div>
                    </dl>
                    <a class="text-link" [routerLink]="['/public-data', 'chemicals']" [queryParams]="{ id: chemical.id }">Open chemical record</a>
                </section>
            }
        </aside>
    `,
})
export class ExploreInspectorComponent {
    readonly isothermId = input.required<number>();
    readonly closed = output<void>();

    protected readonly isotherm = signal<AdsorptionDetailResponse | null>(null);
    protected readonly material = signal<MaterialDetailResponse | null>(null);
    protected readonly chemicals = signal<ChemicalRecordView[]>([]);
    protected readonly error = signal<string | null>(null);

    private token = 0;

    constructor() {
        effect(() => {
            void this.load(this.isothermId());
        });
    }

    protected format(value: number | null): string {
        return formatValue(value);
    }

    protected unitLabel(unit: string): string {
        return pressureLabel(unit);
    }

    private async load(isothermId: number): Promise<void> {
        const token = ++this.token;
        this.isotherm.set(null);
        this.material.set(null);
        this.chemicals.set([]);
        this.error.set(null);
        const detail = await fetchPublicAdsorptionDetail(isothermId);
        if (token !== this.token) return;
        if (!detail.data) {
            this.error.set(detail.error ?? 'The isotherm could not be loaded.');
            return;
        }
        this.isotherm.set(detail.data);
        const [material, ...chemicals] = await Promise.all([
            fetchPublicMaterialDetail(detail.data.material_id),
            ...detail.data.adsorbates.map((adsorbate) => fetchPublicChemicalDetail(adsorbate.id)),
        ]);
        if (token !== this.token) return;
        this.material.set(material.data as MaterialDetailResponse | null);
        this.chemicals.set(chemicals.flatMap((result) => (result.data ? [result.data as ChemicalRecordView] : [])));
        const failure = [material, ...chemicals].find((result) => result.error);
        if (failure) this.error.set(failure.error);
    }
}
