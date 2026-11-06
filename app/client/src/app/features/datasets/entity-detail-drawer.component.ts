// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, computed, effect, input, output, signal, untracked, viewChild } from '@angular/core';
import { DatePipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import type { DatasetUsage, MaterialDetail, SpeciesDetail } from '../../models/materials.model';
import type { ExternalIdentifierView } from '../../models/public-data.model';
import {
    deleteMaterial,
    deleteSpecies,
    fetchMaterialDetail,
    fetchSpeciesDetail,
} from '../../services/materials.service';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';
import type { EntityKind } from './entity-panel.component';
import { EntityStructureComponent } from './entity-structure.component';

type FieldState = 'value' | 'missing' | 'na';

interface EntityField {
    label: string;
    value: string | null;
    state: FieldState;
    /** Why a value is missing or not applicable; shown as a tooltip. */
    hint?: string;
}

interface EntitySection {
    id: 'identity' | 'identifiers' | 'properties' | 'structure' | 'usage' | 'provenance';
    title: string;
    open: boolean;
    fields: EntityField[];
}

interface EntityView {
    name: string;
    sections: EntitySection[];
    datasets: DatasetUsage[];
    experimentCount: number;
    identifiers: ExternalIdentifierView[];
}

const PROPERTY_LABELS: Record<string, string> = {
    xlogp: 'XLogP',
    tpsa_angstrom2: 'Polar surface area',
    h_bond_donor_count: 'H-bond donors',
    h_bond_acceptor_count: 'H-bond acceptors',
    rotatable_bond_count: 'Rotatable bonds',
    complexity: 'Complexity',
    exact_mass_da: 'Exact mass',
    monoisotopic_mass_da: 'Monoisotopic mass',
};

function present(value: string | number | null | undefined): value is string | number {
    return value !== null && value !== undefined && value !== '';
}

/** A stored value, or an explicit "Not available" when the record could carry one but does not. */
function field(label: string, value: string | number | null | undefined, hint?: string): EntityField {
    return present(value) ? { label, value: String(value), state: 'value' } : { label, value: null, state: 'missing', hint };
}

/** A field that does not apply to this kind of entity at all. */
function notApplicable(label: string, hint: string): EntityField {
    return { label, value: null, state: 'na', hint };
}

function formatProperty(prop: SpeciesDetail['properties'][number]): string {
    const value = prop.value_text ?? prop.value_number;
    return `${value}${prop.unit ? ' ' + prop.unit : ''}`;
}

function materialView(item: MaterialDetail): EntityView {
    const molecularOnly = 'Molecular identifiers describe discrete molecules, not adsorbent materials.';
    return {
        name: item.name,
        experimentCount: item.isotherm_count,
        datasets: item.datasets,
        identifiers: item.external_identifiers,
        sections: [
            {
                id: 'identity',
                title: 'Identity and composition',
                open: true,
                fields: [
                    field('Name', item.name),
                    field('Formula', item.formula, 'No composition was recorded for this material.'),
                    field('Molar mass', item.molar_mass_g_mol ? `${item.molar_mass_g_mol} g/mol` : null, 'No molar mass was recorded for this material.'),
                    item.smiles ? field('SMILES', item.smiles) : notApplicable('SMILES', molecularOnly),
                ],
            },
            { id: 'structure', title: 'Crystal structure', open: true, fields: [] },
            {
                id: 'identifiers',
                title: 'Molecular identifiers',
                open: false,
                fields: ['InChI', 'InChIKey', 'PubChem CID'].map((label) => notApplicable(label, molecularOnly)),
            },
            { id: 'usage', title: `Experiments and datasets (${item.isotherm_count})`, open: false, fields: [] },
            { id: 'provenance', title: 'Provenance and sources', open: false, fields: [] },
        ],
    };
}

function speciesView(item: SpeciesDetail): EntityView {
    const notEnriched = 'Not stored. Resolve the species in PubChem from Sources to add it.';
    const elements = item.properties.filter((prop) => prop.key.startsWith('element:'));
    const descriptors = item.properties.filter((prop) => !prop.key.startsWith('element:'));
    const crystalOnly = 'Unit cells and crystal structures describe solid materials, not molecular species.';
    return {
        name: item.preferred_name || item.name,
        experimentCount: item.isotherm_count,
        datasets: item.datasets,
        identifiers: item.external_identifiers,
        sections: [
            {
                id: 'identity',
                title: 'Identity',
                open: true,
                fields: [
                    field('Name', item.name),
                    field('Preferred name', item.preferred_name, notEnriched),
                    field('Formula', item.formula),
                    field('Molecular weight', item.molecular_weight ? `${item.molecular_weight} g/mol` : null),
                    field('Synonyms', item.synonyms.length ? item.synonyms.slice(0, 12).join(', ') : null, notEnriched),
                ],
            },
            {
                id: 'identifiers',
                title: 'Identifiers',
                open: true,
                fields: [
                    field('PubChem CID', item.pubchem_cid, notEnriched),
                    field('InChIKey', item.inchi_key),
                    field('InChI', item.inchi, notEnriched),
                    field('SMILES', item.smiles),
                    field('Connectivity SMILES', item.connectivity_smiles, notEnriched),
                ],
            },
            { id: 'structure', title: 'Structure', open: true, fields: [] },
            {
                id: 'properties',
                title: 'Molecular properties',
                open: false,
                fields: item.properties.length
                    ? [
                          ...descriptors.map((prop) => field(PROPERTY_LABELS[prop.key] ?? prop.key, formatProperty(prop))),
                          ...(elements.length
                              ? [field('Elemental composition', elements.map((prop) => `${prop.key.slice('element:'.length)}${prop.value_number === 1 ? '' : prop.value_number}`).join(' '))]
                              : []),
                          notApplicable('Unit cell and space group', crystalOnly),
                      ]
                    : [field('Computed properties', null, notEnriched), notApplicable('Unit cell and space group', crystalOnly)],
            },
            { id: 'usage', title: `Experiments and datasets (${item.isotherm_count})`, open: false, fields: [] },
            { id: 'provenance', title: 'Provenance and sources', open: false, fields: [] },
        ],
    };
}

@Component({
    selector: 'adsmod-entity-detail-drawer',
    standalone: true,
    imports: [ConfirmDialogComponent, DatePipe, EntityStructureComponent, RouterLink],
    template: `
        <div class="drawer-backdrop" (click)="closed.emit()"></div>
        <aside
            #panel
            class="entity-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="entity-drawer-title"
            tabindex="-1"
            (keydown.escape)="closed.emit()"
        >
            <header class="entity-drawer-head">
                <div>
                    <p class="eyebrow">{{ kind() === 'material' ? 'Adsorbent material' : 'Chemical species' }}</p>
                    <h2 id="entity-drawer-title">{{ view()?.name ?? 'Loading…' }}</h2>
                </div>
                <button class="button quiet" type="button" aria-label="Close details" (click)="closed.emit()">✕</button>
            </header>

            @if (error()) {
                <p class="dataset-status error" role="alert">{{ error() }}</p>
            }
            @if (view(); as entity) {
                <div class="entity-drawer-body">
                    @for (section of entity.sections; track section.id) {
                        <details class="entity-section" [open]="section.open">
                            <summary>{{ section.title }}</summary>
                            @switch (section.id) {
                                @case ('structure') {
                                    <adsmod-entity-structure [kind]="kind()" [detail]="detail()!" />
                                }
                                @case ('usage') {
                                    @if (entity.datasets.length) {
                                        <ul class="entity-list">
                                            @for (dataset of entity.datasets; track dataset.id) {
                                                <li><a class="text-link" [routerLink]="['/datasets', dataset.id]">{{ dataset.name }}</a><span>{{ dataset.experiment_count }} exp.</span></li>
                                            }
                                        </ul>
                                    } @else {
                                        <p class="muted">Not referenced by any dataset.</p>
                                    }
                                }
                                @case ('provenance') {
                                    @if (entity.identifiers.length) {
                                        <ul class="entity-list plain">
                                            @for (identifier of entity.identifiers; track identifier.source + identifier.external_id) {
                                                <li>
                                                    <strong>{{ identifier.source.toUpperCase() }}</strong>
                                                    @if (identifier.source_url) {
                                                        <a class="text-link" [href]="identifier.source_url" target="_blank" rel="noopener">{{ identifier.external_id }}</a>
                                                    } @else {
                                                        {{ identifier.external_id }}
                                                    }
                                                    @if (identifier.retrieved_at) {
                                                        <span class="entity-date">retrieved {{ identifier.retrieved_at | date: 'mediumDate' }}</span>
                                                    }
                                                </li>
                                            }
                                        </ul>
                                    } @else {
                                        <p class="muted">No provider record. This {{ kind() === 'material' ? 'material' : 'species' }} was created from an uploaded dataset.</p>
                                    }
                                }
                                @default {
                                    <dl class="entity-facts">
                                        @for (item of section.fields; track item.label) {
                                            <div>
                                                <dt>{{ item.label }}</dt>
                                                <dd [class.absent]="item.state !== 'value'" [title]="item.hint ?? ''">{{ item.state === 'value' ? item.value : item.state === 'na' ? 'Not applicable' : 'Not available' }}</dd>
                                            </div>
                                        }
                                    </dl>
                                }
                            }
                        </details>
                    }
                </div>
                <footer class="entity-drawer-foot">
                    @if (entity.experimentCount) {
                        <p class="muted">In use, so it cannot be deleted. Delete the datasets that use it, or assign another {{ kind() === 'material' ? 'material' : 'species' }} to those experiments first.</p>
                    }
                    <button class="button quiet danger" type="button" [disabled]="entity.experimentCount > 0 || busy()" (click)="confirmOpen.set(true)">
                        Delete {{ kind() === 'material' ? 'material' : 'species' }}
                    </button>
                </footer>
            }
        </aside>
        <adsmod-confirm-dialog
            [open]="confirmOpen()"
            [title]="'Delete ' + (view()?.name ?? '')"
            message="This permanently removes the record and its provider links. Datasets are not affected because none reference it."
            confirmLabel="Delete"
            (confirmed)="remove()"
            (closed)="confirmOpen.set(false)"
        />
    `,
    styles: [
        `
            .drawer-backdrop { position: fixed; inset: 0; z-index: 40; background: rgb(3 24 47 / 0.18); }
            .entity-drawer {
                position: fixed;
                top: 0;
                right: 0;
                bottom: 0;
                z-index: 41;
                display: grid;
                grid-template-rows: auto minmax(0, 1fr) auto;
                width: min(26rem, 100vw);
                border-left: 1px solid var(--border-color);
                background: var(--surface);
                box-shadow: -18px 0 48px rgb(3 24 47 / 0.18);
            }
            .entity-drawer:focus { outline: none; }
            .entity-drawer-head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--spacing-sm); padding: var(--spacing-md) var(--spacing-md) var(--spacing-sm); border-bottom: 1px solid var(--border-color); }
            .entity-drawer-head h2 { margin: 0; font-size: 1.05rem; overflow-wrap: anywhere; }
            .eyebrow { margin: 0 0 2px; color: var(--text-tertiary); font-size: 0.66rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
            .entity-drawer-body { display: grid; align-content: start; gap: var(--spacing-md); padding: var(--spacing-md); overflow-y: auto; }
            .entity-section { border-bottom: 1px solid var(--border-color); padding-bottom: var(--spacing-sm); }
            .entity-section summary { margin-bottom: 0.4rem; font-size: 0.78rem; font-weight: 700; cursor: pointer; }
            .entity-section > :not(summary) { margin-top: 0.4rem; }
            .entity-facts dd.absent { color: var(--text-tertiary); font-style: italic; }
            .entity-date { color: var(--text-tertiary); font-size: 0.72rem; }
            .entity-facts { display: grid; gap: 0.35rem; margin: 0; }
            .entity-facts div { display: grid; grid-template-columns: 7.5rem minmax(0, 1fr); gap: 0.5rem; }
            .entity-facts dt { color: var(--text-tertiary); font-size: 0.74rem; }
            .entity-facts dd { margin: 0; font-size: 0.8rem; overflow-wrap: anywhere; }
            .entity-list { display: grid; gap: 0.2rem; margin: 0; padding: 0; list-style: none; font-size: 0.8rem; }
            .entity-list li { display: flex; justify-content: space-between; gap: 0.5rem; }
            .entity-list.plain li { display: block; color: var(--text-secondary); overflow-wrap: anywhere; }
            .entity-list span { color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
            .muted { margin: 0; color: var(--text-secondary); font-size: 0.78rem; }
            .entity-drawer-foot { display: grid; gap: 0.5rem; padding: var(--spacing-sm) var(--spacing-md) var(--spacing-md); border-top: 1px solid var(--border-color); }
            .entity-drawer-foot button { justify-self: start; }
            .dataset-status { margin: var(--spacing-sm) var(--spacing-md) 0; }
        `,
    ],
})
export class EntityDetailDrawerComponent {
    readonly kind = input.required<EntityKind>();
    readonly entityId = input.required<number>();
    readonly closed = output<void>();
    readonly deleted = output<void>();

    private readonly panel = viewChild<ElementRef<HTMLElement>>('panel');
    protected readonly detail = signal<MaterialDetail | SpeciesDetail | null>(null);
    protected readonly error = signal<string | null>(null);
    protected readonly busy = signal(false);
    protected readonly confirmOpen = signal(false);
    protected readonly view = computed(() => {
        const detail = this.detail();
        if (!detail) return null;
        return this.kind() === 'material' ? materialView(detail as MaterialDetail) : speciesView(detail as SpeciesDetail);
    });

    private requestId = 0;

    constructor() {
        effect(() => {
            const kind = this.kind();
            const id = this.entityId();
            untracked(() => void this.load(kind, id));
        });
        effect(() => {
            this.panel()?.nativeElement.focus();
        });
    }

    private async load(kind: EntityKind, id: number): Promise<void> {
        const requestId = ++this.requestId;
        this.detail.set(null);
        this.error.set(null);
        const result = kind === 'material' ? await fetchMaterialDetail(id) : await fetchSpeciesDetail(id);
        if (requestId !== this.requestId) return;
        this.detail.set(result.data);
        this.error.set(result.error);
    }

    protected async remove(): Promise<void> {
        this.confirmOpen.set(false);
        this.busy.set(true);
        const result = this.kind() === 'material' ? await deleteMaterial(this.entityId()) : await deleteSpecies(this.entityId());
        this.busy.set(false);
        if (result.error) {
            this.error.set(result.error);
            return;
        }
        this.deleted.emit();
    }
}
