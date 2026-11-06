// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { EntityDetailDrawerComponent } from './entity-detail-drawer.component';

const speciesDetail = {
    id: 5,
    name: 'Methane',
    preferred_name: null,
    formula: 'CH4',
    molecular_weight: 16.04,
    inchi: null,
    inchi_key: 'VNWKTOKETHGBQD-UHFFFAOYSA-N',
    connectivity_smiles: null,
    smiles: null,
    pubchem_cid: null,
    synonyms: [],
    properties: [],
    external_identifiers: [],
    structure_2d_url: null,
    conformer_3d_url: null,
    conformer_3d_cached: false,
    isotherm_count: 0,
    retrieved_at: null,
    datasets: [],
};

const materialDetail = {
    id: 8,
    name: 'ZIF-8',
    formula: 'C8H10N4Zn',
    molar_mass_g_mol: null,
    smiles: null,
    structure_count: 0,
    isotherm_count: 0,
    external_identifiers: [],
    structures: [],
    datasets: [],
};

async function render(kind: 'species' | 'material', payload: unknown): Promise<HTMLElement> {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => payload })));
    TestBed.configureTestingModule({ providers: [provideRouter([])] });
    const fixture = TestBed.createComponent(EntityDetailDrawerComponent);
    fixture.componentRef.setInput('kind', kind);
    fixture.componentRef.setInput('entityId', 1);
    fixture.detectChanges();
    await vi.waitFor(() => {
        fixture.detectChanges();
        expect((fixture.nativeElement as HTMLElement).querySelector('.entity-section')).not.toBeNull();
    });
    return fixture.nativeElement as HTMLElement;
}

const row = (element: HTMLElement, label: string): string =>
    Array.from(element.querySelectorAll('.entity-facts > div'))
        .find((node) => node.querySelector('dt')?.textContent === label)
        ?.querySelector('dd')?.textContent?.trim() ?? '';

describe('EntityDetailDrawerComponent', () => {
    afterEach(() => vi.unstubAllGlobals());

    it('separates missing species data from fields that do not apply to molecules', async () => {
        const element = await render('species', speciesDetail);

        expect(row(element, 'InChIKey')).toBe('VNWKTOKETHGBQD-UHFFFAOYSA-N');
        expect(row(element, 'PubChem CID')).toBe('Not available');
        expect(row(element, 'InChI')).toBe('Not available');
        expect(row(element, 'Unit cell and space group')).toBe('Not applicable');
    });

    it('marks molecular identifiers as not applicable to materials while keeping missing composition data distinct', async () => {
        const element = await render('material', materialDetail);

        expect(row(element, 'Formula')).toBe('C8H10N4Zn');
        expect(row(element, 'Molar mass')).toBe('Not available');
        expect(row(element, 'InChIKey')).toBe('Not applicable');
        expect(row(element, 'PubChem CID')).toBe('Not applicable');
        expect(row(element, 'SMILES')).toBe('Not applicable');
    });
});
