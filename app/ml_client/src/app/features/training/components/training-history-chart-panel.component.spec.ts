import { TestBed } from '@angular/core/testing';
import { TrainingHistoryChartPanelComponent } from './training-history-chart-panel.component';

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
