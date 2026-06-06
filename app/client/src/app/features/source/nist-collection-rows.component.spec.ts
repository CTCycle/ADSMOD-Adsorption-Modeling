import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NistCollectionRowsComponent } from './nist-collection-rows.component';

describe('NistCollectionRowsComponent', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        window.localStorage.clear();
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'success',
                categories: [
                    { category: 'experiments', local_count: 2, available_count: 10, last_update: null, server_ok: true, server_checked_at: null, supports_enrichment: false },
                    { category: 'guest', local_count: 1, available_count: 5, last_update: null, server_ok: false, server_checked_at: null, supports_enrichment: true },
                    { category: 'host', local_count: 0, available_count: 4, last_update: null, server_ok: null, server_checked_at: null, supports_enrichment: true },
                ],
            }),
        });
        vi.stubGlobal('fetch', fetchMock);
    });

    it('hydrates fraction inputs from localStorage using the documented keys', async () => {
        window.localStorage.setItem('adsmod.nist.fraction.experiments', '0.125');

        await TestBed.configureTestingModule({
            imports: [NistCollectionRowsComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(NistCollectionRowsComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const experimentsInput = (fixture.nativeElement as HTMLElement).querySelector<HTMLInputElement>('#fraction-experiments');
        expect(experimentsInput?.value).toBe('0.125');
    });

    it('persists normalized fractions on blur', async () => {
        await TestBed.configureTestingModule({
            imports: [NistCollectionRowsComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(NistCollectionRowsComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const experimentsInput = (fixture.nativeElement as HTMLElement).querySelector<HTMLInputElement>('#fraction-experiments');
        expect(experimentsInput).not.toBeNull();

        experimentsInput!.value = '2';
        experimentsInput!.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        experimentsInput!.dispatchEvent(new Event('blur'));
        fixture.detectChanges();

        expect(window.localStorage.getItem('adsmod.nist.fraction.experiments')).toBe('1.000');
        expect(experimentsInput!.value).toBe('1.000');
    });
});
