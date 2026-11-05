// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, computed, effect, input, output, signal, untracked, viewChild } from '@angular/core';
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

interface EntityFact {
    label: string;
    value: string;
}

interface EntityView {
    name: string;
    facts: EntityFact[];
    datasets: DatasetUsage[];
    experimentCount: number;
    identifiers: ExternalIdentifierView[];
    extras: { title: string; items: string[] }[];
}

function present(value: string | number | null | undefined): value is string | number {
    return value !== null && value !== undefined && value !== '';
}

function facts(entries: [string, string | number | null | undefined][]): EntityFact[] {
    return entries.filter(([, value]) => present(value)).map(([label, value]) => ({ label, value: String(value) }));
}

function materialView(item: MaterialDetail): EntityView {
    return {
        name: item.name,
        facts: facts([
            ['Formula', item.formula],
            ['Molar mass', item.molar_mass_g_mol ? `${item.molar_mass_g_mol} g/mol` : null],
            ['SMILES', item.smiles],
        ]),
        datasets: item.datasets,
        experimentCount: item.isotherm_count,
        identifiers: item.external_identifiers,
        extras: item.structures.length
            ? [{ title: 'Linked structures', items: item.structures.map((structure) => `${structure.name || structure.external_id} · ${structure.formula ?? ''} ${structure.space_group ? '· ' + structure.space_group : ''}`.trim()) }]
            : [],
    };
}

function speciesView(item: SpeciesDetail): EntityView {
    return {
        name: item.preferred_name || item.name,
        facts: facts([
            ['Name', item.preferred_name && item.preferred_name !== item.name ? item.name : null],
            ['Formula', item.formula],
            ['Molecular weight', item.molecular_weight ? `${item.molecular_weight} g/mol` : null],
            ['InChIKey', item.inchi_key],
            ['SMILES', item.smiles || item.connectivity_smiles],
            ['PubChem CID', item.pubchem_cid],
        ]),
        datasets: item.datasets,
        experimentCount: item.isotherm_count,
        identifiers: item.external_identifiers,
        extras: [
            ...(item.synonyms.length ? [{ title: 'Synonyms', items: item.synonyms.slice(0, 12) }] : []),
            ...(item.properties.length
                ? [{ title: 'Properties', items: item.properties.map((prop) => `${prop.key}: ${prop.value_text ?? prop.value_number ?? ''}${prop.unit ? ' ' + prop.unit : ''}`) }]
                : []),
        ],
    };
}

@Component({
    selector: 'adsmod-entity-detail-drawer',
    standalone: true,
    imports: [ConfirmDialogComponent, RouterLink],
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
                    @if (entity.facts.length) {
                        <dl class="entity-facts">
                            @for (fact of entity.facts; track fact.label) {
                                <div><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div>
                            }
                        </dl>
                    }
                    <section>
                        <h3>Used in {{ entity.experimentCount }} experiment{{ entity.experimentCount === 1 ? '' : 's' }}</h3>
                        @if (entity.datasets.length) {
                            <ul class="entity-list">
                                @for (dataset of entity.datasets; track dataset.id) {
                                    <li><a class="text-link" [routerLink]="['/datasets', dataset.id]">{{ dataset.name }}</a><span>{{ dataset.experiment_count }} exp.</span></li>
                                }
                            </ul>
                        } @else {
                            <p class="muted">Not referenced by any dataset.</p>
                        }
                    </section>
                    @for (extra of entity.extras; track extra.title) {
                        <section>
                            <h3>{{ extra.title }}</h3>
                            <ul class="entity-list plain">
                                @for (item of extra.items; track $index) {
                                    <li>{{ item }}</li>
                                }
                            </ul>
                        </section>
                    }
                    @if (entity.identifiers.length) {
                        <section>
                            <h3>Provenance</h3>
                            <ul class="entity-list plain">
                                @for (identifier of entity.identifiers; track identifier.source + identifier.external_id) {
                                    <li>
                                        <strong>{{ identifier.source.toUpperCase() }}</strong>
                                        @if (identifier.source_url) {
                                            <a class="text-link" [href]="identifier.source_url" target="_blank" rel="noopener">{{ identifier.external_id }}</a>
                                        } @else {
                                            {{ identifier.external_id }}
                                        }
                                    </li>
                                }
                            </ul>
                        </section>
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
            .entity-drawer-body h3 { margin: 0 0 0.35rem; font-size: 0.78rem; }
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
