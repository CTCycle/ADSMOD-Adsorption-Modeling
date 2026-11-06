// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MaterialDetail, SpeciesDetail } from '../../models/materials.model';
import { EntityStructureComponent } from './entity-structure.component';

const species = (overrides: Partial<SpeciesDetail> = {}): SpeciesDetail => ({
    id: 5,
    name: 'Methane',
    preferred_name: null,
    formula: 'CH4',
    molecular_weight: 16.04,
    inchi: null,
    inchi_key: 'VNWKTOKETHGBQD-UHFFFAOYSA-N',
    connectivity_smiles: null,
    smiles: 'C',
    pubchem_cid: '297',
    synonyms: [],
    properties: [],
    external_identifiers: [],
    structure_2d_url: 'https://pubchem.example/297/PNG',
    conformer_3d_url: 'https://pubchem.example/297/SDF',
    conformer_3d_cached: false,
    isotherm_count: 0,
    retrieved_at: null,
    datasets: [],
    ...overrides,
});

const material = (overrides: Partial<MaterialDetail> = {}): MaterialDetail => ({
    id: 8,
    name: 'ZIF-8',
    formula: null,
    molar_mass_g_mol: null,
    smiles: null,
    structure_count: 0,
    isotherm_count: 0,
    external_identifiers: [],
    structures: [],
    datasets: [],
    ...overrides,
});

describe('EntityStructureComponent', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({ ok: true, status: 200, text: async () => 'methane\n  3D\n\nM  END' });
        vi.stubGlobal('fetch', fetchMock);
        TestBed.configureTestingModule({ providers: [provideRouter([])] });
    });
    afterEach(() => vi.unstubAllGlobals());

    function render(kind: 'species' | 'material', detail: SpeciesDetail | MaterialDetail): HTMLElement {
        const fixture = TestBed.createComponent(EntityStructureComponent);
        fixture.componentRef.setInput('kind', kind);
        fixture.componentRef.setInput('detail', detail);
        fixture.detectChanges();
        return fixture.nativeElement as HTMLElement;
    }

    it('never guesses a structure for a species without a PubChem link', () => {
        const element = render('species', species({ pubchem_cid: null, structure_2d_url: null, conformer_3d_url: null }));

        expect(element.querySelector('img')).toBeNull();
        expect(element.textContent).toContain('not linked to PubChem');
        expect(element.querySelector('a')?.getAttribute('href')).toContain('/sources/acquire?panel=pubchem');
    });

    it('downloads the 3D conformer only on request and skips the request once cached', () => {
        const uncached = render('species', species());
        expect(uncached.querySelector('img')?.getAttribute('src')).toBe('https://pubchem.example/297/PNG');
        expect(fetchMock).not.toHaveBeenCalled();
        expect(uncached.querySelector('button')?.textContent).toContain('Fetch and view 3D');
        (uncached.querySelector('button') as HTMLButtonElement).click();
        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(String(fetchMock.mock.calls[0][0])).toContain('/public-data/chemicals/5/conformer');
        expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST' });

        TestBed.resetTestingModule();
        TestBed.configureTestingModule({ providers: [provideRouter([])] });
        expect(render('species', species({ conformer_3d_cached: true })).querySelector('button')?.textContent).toContain('View 3D');
    });

    it('offers a COD search for materials without a linked crystal structure', () => {
        const element = render('material', material());

        expect(element.textContent).toContain('no crystal structure is linked');
        expect(element.querySelector('a')?.getAttribute('href')).toContain('/sources/acquire?panel=cod&material_id=8');
    });
});
