// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NistTargetedSearchComponent } from './nist-targeted-search.component';

describe('NistTargetedSearchComponent', () => {
    const fetchMock = vi.fn();

    beforeEach(async () => {
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                total: 1,
                items: [{ filename: '10.1002adfm.200800058.Isotherm17', doi: null, adsorbent_name: 'Co3(HCOO)6', adsorbent_hashkey: 'k', adsorbate_names: ['Methanol'], adsorbate_inchi_keys: ['i'], temperature_k: 333, local: false }],
            }),
        });
        vi.stubGlobal('fetch', fetchMock);
        TestBed.resetTestingModule();
        await TestBed.configureTestingModule({ imports: [NistTargetedSearchComponent] }).compileComponents();
    });

    afterEach(() => vi.unstubAllGlobals());

    it('stays disabled without a criterion and sends every filled filter in one request', async () => {
        const fixture = TestBed.createComponent(NistTargetedSearchComponent);
        fixture.detectChanges();
        const element = fixture.nativeElement as HTMLElement;
        const search = element.querySelector<HTMLButtonElement>('.query-row button')!;
        expect(search.disabled).toBe(true);

        const type = (input: HTMLInputElement, value: string): void => {
            input.value = value;
            input.dispatchEvent(new Event('input'));
        };
        const inputs = element.querySelectorAll<HTMLInputElement>('input:not([type="checkbox"])');
        type(inputs[0], 'methanol');
        type(inputs[1], 'Co3');
        type(inputs[3], '300');
        type(inputs[4], '350');
        fixture.detectChanges();
        expect(search.disabled).toBe(false);

        search.click();
        await fixture.whenStable();
        fixture.detectChanges();

        const url = new URL(String(fetchMock.mock.calls[0][0]), 'http://localhost');
        expect(Object.fromEntries(url.searchParams)).toMatchObject({
            adsorbate: 'methanol',
            adsorbent: 'Co3',
            temperature_min: '300',
            temperature_max: '350',
            limit: '25',
        });
        expect(element.querySelectorAll('tbody tr')).toHaveLength(1);
    });
});
