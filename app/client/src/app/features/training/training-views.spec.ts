// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { TrainingHistoryChartPanelComponent } from './components/training-history-chart-panel.component';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { of } from 'rxjs';
import { MachineLearningPageComponent } from './pages/machine-learning-page.component';

// Training history chart behavior

describe('TrainingHistoryChartPanelComponent', () => {
    it('shows the empty placeholder when there is no history', async () => {
        await TestBed.configureTestingModule({
            imports: [TrainingHistoryChartPanelComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(TrainingHistoryChartPanelComponent);
        fixture.componentRef.setInput('title', 'LOSS');
        fixture.componentRef.setInput('hasHistory', false);
        fixture.componentRef.setInput('history', []);
        fixture.componentRef.setInput('primaryLine', { dataKey: 'loss', color: '#f59e0b', name: 'Train Loss' });
        fixture.componentRef.setInput('secondaryLine', { dataKey: 'val_loss', color: '#2563eb', name: 'Val Loss' });
        fixture.componentRef.setInput('placeholderHint', 'Loss metrics will appear once training starts.');
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelector('.chart-placeholder')?.textContent).toContain('Waiting for training data');
    });

    it('renders svg paths and legend entries when history exists', async () => {
        await TestBed.configureTestingModule({
            imports: [TrainingHistoryChartPanelComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(TrainingHistoryChartPanelComponent);
        fixture.componentRef.setInput('title', 'LOSS');
        fixture.componentRef.setInput('hasHistory', true);
        fixture.componentRef.setInput('history', [
            { epoch: 1, loss: 0.8, val_loss: 1.2 },
            { epoch: 2, loss: 0.4, val_loss: 0.9 },
        ]);
        fixture.componentRef.setInput('primaryLine', { dataKey: 'loss', color: '#f59e0b', name: 'Train Loss' });
        fixture.componentRef.setInput('secondaryLine', { dataKey: 'val_loss', color: '#2563eb', name: 'Val Loss' });
        fixture.componentRef.setInput('placeholderHint', 'Loss metrics will appear once training starts.');
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelectorAll('path').length).toBeGreaterThanOrEqual(2);
        expect(root.textContent).toContain('Train Loss');
        expect(root.textContent).toContain('Val Loss');
    });
});

// Machine learning page behavior

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
        await vi.waitFor(() => {
            fixture.detectChanges();
            expect(fixture.nativeElement.textContent).toContain(
                'Training requires the optional ML service.',
            );
        });

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
