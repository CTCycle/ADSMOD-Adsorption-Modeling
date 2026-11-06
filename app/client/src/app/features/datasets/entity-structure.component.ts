// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, input, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { MaterialDetail, SpeciesDetail } from '../../models/materials.model';
import type { StructureRecordView } from '../../models/public-data.model';
import { fetchSpeciesConformer, fetchStructureFileText, structureFileUrl } from '../../services/public-data.service';
import {
    StructureViewerDialogComponent,
    type StructureFormat,
} from '../../shared/components/structure-viewer/structure-viewer-dialog.component';
import type { EntityKind } from './entity-panel.component';

interface ViewerRequest {
    title: string;
    subtitle: string | null;
    format: StructureFormat;
    content: string;
}

/** Structure preview for the detail drawer: a compact 2D image for species, structure records for materials, and a 3D viewer dialog. */
@Component({
    selector: 'adsmod-entity-structure',
    standalone: true,
    imports: [RouterLink, StructureViewerDialogComponent],
    template: `
        @if (kind() === 'species') {
            @if (species(); as chemical) {
                @if (chemical.pubchem_cid) {
                    @if (imageFailed()) {
                        <p class="muted">The 2D image could not be loaded from PubChem. Check the connection and reopen the panel.</p>
                    } @else if (chemical.structure_2d_url) {
                        <img class="structure-2d" [src]="chemical.structure_2d_url" [alt]="'2D structure of ' + chemical.name" width="300" height="300" loading="lazy" (error)="imageFailed.set(true)" />
                    }
                    <div class="structure-actions">
                        @if (cached()) {
                            <button class="button secondary" type="button" (click)="openSpecies(chemical)">View 3D</button>
                        } @else if (chemical.conformer_3d_url) {
                            <button class="button secondary" type="button" [disabled]="busy()" (click)="openSpecies(chemical)">
                                {{ busy() ? 'Fetching 3D structure…' : 'Fetch and view 3D' }}
                            </button>
                        } @else {
                            <p class="muted">PubChem publishes no 3D conformer for this compound.</p>
                        }
                    </div>
                } @else {
                    <p class="muted">Not available: this species is not linked to PubChem, so no structure is known for it.</p>
                    <a class="text-link" routerLink="/sources/acquire" [queryParams]="{ panel: 'pubchem', query: chemical.name, adsorbate_id: chemical.id }">Resolve in PubChem</a>
                }
            }
        } @else {
            @if (material(); as adsorbent) {
                @if (adsorbent.structures.length) {
                    <ul class="structure-list">
                        @for (structure of adsorbent.structures; track structure.id) {
                            <li>
                                <div>
                                    <strong>{{ structure.name || structure.source.toUpperCase() + ' ' + structure.external_id }}</strong>
                                    <span>{{ describe(structure) }}</span>
                                </div>
                                <div class="structure-row-actions">
                                    <button class="button secondary" type="button" [disabled]="!structure.has_coordinates || busy()" [title]="structure.has_coordinates ? 'Open the 3D viewer' : 'This record has no atom coordinates'" (click)="openStructure(structure)">View 3D</button>
                                    <a class="text-link" [href]="fileUrl(structure.id)">CIF</a>
                                </div>
                            </li>
                        }
                    </ul>
                } @else {
                    <p class="muted">Not available: no crystal structure is linked to this material. Amorphous, composite and polymeric adsorbents often have none.</p>
                    <a class="text-link" routerLink="/sources/acquire" [queryParams]="{ panel: 'cod', material_id: adsorbent.id, material_name: adsorbent.name }">Search COD for a structure</a>
                }
            }
        }
        @if (error()) {
            <p class="structure-error" role="alert">{{ error() }}</p>
        }
        @if (viewer(); as request) {
            <adsmod-structure-viewer-dialog [title]="request.title" [subtitle]="request.subtitle" [format]="request.format" [content]="request.content" (closed)="viewer.set(null)" />
        }
    `,
    styles: [
        `
            :host { display: grid; gap: 0.5rem; }
            .muted { margin: 0; color: var(--text-secondary); font-size: 0.78rem; }
            .structure-2d { justify-self: center; width: min(14rem, 100%); height: auto; border: 1px solid var(--border-color); border-radius: 6px; background: #fff; }
            .structure-actions { display: flex; flex-wrap: wrap; gap: 0.5rem; }
            .structure-list { display: grid; gap: 0.5rem; margin: 0; padding: 0; list-style: none; }
            .structure-list li { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.5rem; padding: 0.45rem 0.55rem; border: 1px solid var(--border-color); border-radius: 6px; }
            .structure-list div:first-child { display: grid; min-width: 0; font-size: 0.8rem; overflow-wrap: anywhere; }
            .structure-list span { color: var(--text-secondary); font-size: 0.74rem; }
            .structure-row-actions { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; }
            .structure-error { margin: 0; color: var(--danger-text); font-size: 0.78rem; }
        `,
    ],
})
export class EntityStructureComponent {
    readonly kind = input.required<EntityKind>();
    readonly detail = input.required<MaterialDetail | SpeciesDetail>();

    protected readonly species = computed(() => (this.kind() === 'species' ? (this.detail() as SpeciesDetail) : null));
    protected readonly material = computed(() => (this.kind() === 'material' ? (this.detail() as MaterialDetail) : null));
    protected readonly busy = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly imageFailed = signal(false);
    protected readonly viewer = signal<ViewerRequest | null>(null);
    private readonly fetchedIds = signal<ReadonlySet<number>>(new Set());
    protected readonly cached = computed(() => {
        const chemical = this.species();
        return !!chemical && (chemical.conformer_3d_cached || this.fetchedIds().has(chemical.id));
    });

    protected fileUrl = structureFileUrl;

    protected describe(structure: StructureRecordView): string {
        const cell = [structure.cell_a_angstrom, structure.cell_b_angstrom, structure.cell_c_angstrom];
        return [
            structure.formula,
            structure.space_group,
            cell.every((value) => value !== null) ? `a,b,c ${cell.map((value) => Number(value).toFixed(2)).join(' / ')} Å` : null,
            structure.has_coordinates ? `${structure.atom_count} atoms` : 'no coordinates',
        ]
            .filter(Boolean)
            .join(' · ');
    }

    protected async openSpecies(chemical: SpeciesDetail): Promise<void> {
        await this.open(() => fetchSpeciesConformer(chemical.id), {
            title: chemical.preferred_name || chemical.name,
            subtitle: `PubChem CID ${chemical.pubchem_cid} · 3D conformer`,
            format: 'sdf',
        }, () => this.fetchedIds.update((ids) => new Set(ids).add(chemical.id)));
    }

    protected async openStructure(structure: StructureRecordView): Promise<void> {
        await this.open(() => fetchStructureFileText(structure.id), {
            title: structure.name || `${structure.source.toUpperCase()} ${structure.external_id}`,
            subtitle: this.describe(structure),
            format: 'cif',
        });
    }

    private async open(
        load: () => Promise<{ data: string | null; error: string | null }>,
        request: Omit<ViewerRequest, 'content'>,
        onLoaded?: () => void,
    ): Promise<void> {
        this.busy.set(true);
        this.error.set(null);
        const result = await load();
        this.busy.set(false);
        if (result.error || !result.data) {
            this.error.set(result.error ?? 'No structure data was returned.');
            return;
        }
        onLoaded?.();
        this.viewer.set({ ...request, content: result.data });
    }
}
