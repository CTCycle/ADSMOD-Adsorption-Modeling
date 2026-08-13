import { TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { of } from 'rxjs';
import { MachineLearningPageComponent } from './machine-learning-page.component';

describe('MachineLearningPageComponent', () => {
    const fetchMock = vi.fn();

    beforeEach(async () => {
        TestBed.resetTestingModule();
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({
            ok: false,
            status: 503,
            statusText: 'Service Unavailable',
            json: async () => ({ detail: 'ML service unavailable' }),
        });
        vi.stubGlobal('fetch', fetchMock);
        await TestBed.configureTestingModule({
            imports: [MachineLearningPageComponent],
            providers: [
                provideRouter([]),
                {
                    provide: ActivatedRoute,
                    useValue: { paramMap: of(convertToParamMap({ view: 'processing' })) },
                },
            ],
        }).compileComponents();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('keeps every training view visible when the ML service is unavailable', async () => {
        const fixture = TestBed.createComponent(MachineLearningPageComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const tabs = Array.from(
            root.querySelectorAll<HTMLAnchorElement>('.training-view-tab'),
        );
        expect(tabs.map((tab) => tab.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
            'Data Processing',
            'Train datasets',
            'Checkpoints',
            'Training Dashboard',
        ]);
        expect(tabs.every((tab) => tab.getAttribute('href')?.startsWith('/training/'))).toBe(true);
        expect(root.textContent).toContain('Training requires the optional ML service.');
    });
});
