// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, effect, input, output, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { ExperimentDetail } from '../../../models/dataset.model';
import type { MaterialDetail, SpeciesDetail } from '../../../models/materials.model';
import { fetchExperimentDetail } from '../../../services/dataset.service';
import { fetchMaterialDetail, fetchSpeciesDetail } from '../../../services/materials.service';
import { formatValue, pressureLabel } from '../dashboard-charts';

export interface InspectorTarget {
    datasetId: number;
    experimentId: number;
}

@Component({
    selector: 'adsmod-explore-inspector',
    standalone: true,
    imports: [RouterLink],
    styleUrl: '../dashboards.css',
    host: { '(document:keydown.escape)': 'closed.emit()' },
    template: `
        <aside class="detail-panel explore-inspector" aria-label="Experiment details" aria-live="polite">
            <div class="detail-header">
                <div>
                    <p class="eyebrow">{{ detail()?.dataset_name ?? 'Experiment' }} · {{ detail()?.experiment?.external_key ?? '#' + target().experimentId }}</p>
                    <h3>{{ detail()?.experiment?.adsorbent ?? 'Loading…' }}</h3>
                </div>
                <button class="detail-close" type="button" aria-label="Close details" (click)="closed.emit()">×</button>
            </div>
            @if (error()) {
                <p class="inline-error" role="alert">{{ error() }}</p>
            }
            @if (detail(); as info) {
                <div class="detail-grid">
                    <div><span>Temperature</span><strong>{{ format(info.experiment.temperature_k) }} K</strong></div>
                    <div><span>Pressure basis</span><strong>{{ info.experiment.pressure_basis }}</strong></div>
                    <div><span>Measurements</span><strong>{{ info.experiment.observation_count }}</strong></div>
                    <div><span>Pressure range</span><strong>{{ format(info.experiment.pressure_min) }}–{{ format(info.experiment.pressure_max) }} {{ unitLabel(info.experiment.pressure_canonical_unit ?? 'Pa') }}</strong></div>
                </div>
                <div class="detail-links">
                    <a class="text-link" [routerLink]="['/datasets', target().datasetId]" [queryParams]="{ experiment: target().experimentId }">Open in dataset</a>
                    @for (record of info.provenance; track record.source + record.external_id) {
                        @for (doi of record.doi; track doi) {<a class="text-link" [href]="'https://doi.org/' + doi" target="_blank" rel="noopener noreferrer">DOI {{ doi }}</a>}
                        @if (record.source_url) {<a class="text-link" [href]="record.source_url" target="_blank" rel="noopener noreferrer">Source record</a>}
                    }
                </div>
            }
            @if (material(); as item) {
                <section class="detail-section">
                    <h4>Material</h4>
                    <dl class="property-list">
                        <div><dt>Name</dt><dd>{{ item.name }}</dd></div>
                        <div><dt>Formula</dt><dd>{{ item.formula ?? '—' }}</dd></div>
                        <div><dt>Molar mass</dt><dd>{{ item.molar_mass_g_mol === null ? '—' : format(item.molar_mass_g_mol) + ' g/mol' }}</dd></div>
                        <div><dt>Experiments</dt><dd>{{ item.isotherm_count }}</dd></div>
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
                </section>
            }
            @for (chemical of species(); track chemical.id) {
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
                </section>
            }
    `,
})
export class ExploreInspectorComponent {
    readonly target = input.required<InspectorTarget>();
    readonly closed = output<void>();

    protected readonly detail = signal<ExperimentDetail | null>(null);
    protected readonly material = signal<MaterialDetail | null>(null);
    protected readonly species = signal<SpeciesDetail[]>([]);
    protected readonly error = signal<string | null>(null);

    private token = 0;

    constructor() {
        effect(() => {
            void this.load(this.target());
        });
    }

    protected format(value: number | null): string {
        return formatValue(value);
    }

    protected unitLabel(unit: string): string {
        return pressureLabel(unit);
    }

    private async load(target: InspectorTarget): Promise<void> {
        const token = ++this.token;
        this.detail.set(null);
        this.material.set(null);
        this.species.set([]);
        this.error.set(null);
        const detail = await fetchExperimentDetail(target.datasetId, target.experimentId);
        if (token !== this.token) return;
        if (!detail.data) {
            this.error.set(detail.error ?? 'The experiment could not be loaded.');
            return;
        }
        this.detail.set(detail.data);
        const experiment = detail.data.experiment;
        const [material, ...species] = await Promise.all([
            fetchMaterialDetail(experiment.adsorbent_id),
            ...experiment.components.map((component) => fetchSpeciesDetail(component.adsorbate_id)),
        ]);
        if (token !== this.token) return;
        this.material.set(material.data as MaterialDetail | null);
        this.species.set(species.flatMap((result) => (result.data ? [result.data as SpeciesDetail] : [])));
        const failure = [material, ...species].find((result) => result.error);
        if (failure) this.error.set(failure.error);
    }
}
