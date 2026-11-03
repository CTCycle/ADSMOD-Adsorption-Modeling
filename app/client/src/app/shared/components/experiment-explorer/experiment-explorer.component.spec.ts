// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import {
    ExperimentExplorerComponent,
    type ExplorerExperiment,
} from './experiment-explorer.component';

function experiment(key: string, overrides: Partial<ExplorerExperiment> = {}): ExplorerExperiment {
    return {
        key,
        name: `Experiment ${key}`,
        subtitle: 'CO2 on 13X',
        temperature: '298.15 K',
        pointCount: 2,
        pressureRange: '1–2 bar',
        status: 'valid',
        errors: 0,
        warnings: 0,
        issues: [],
        rows: [
            { point: 1, pressure: '100,000 Pa', pressureSource: '1 bar', uptake: '0.1 mol/kg', uptakeSource: null, status: 'valid', note: null },
            { point: 2, pressure: '200,000 Pa', pressureSource: '2 bar', uptake: '0.2 mol/kg', uptakeSource: null, status: 'valid', note: null },
        ],
        totalRows: 2,
        paged: false,
        page: 1,
        pageSize: 2,
        loading: false,
        error: null,
        ...overrides,
    };
}

async function render(experiments: ExplorerExperiment[]) {
    await TestBed.configureTestingModule({ imports: [ExperimentExplorerComponent] }).compileComponents();
    const fixture = TestBed.createComponent(ExperimentExplorerComponent);
    fixture.componentRef.setInput('experiments', experiments);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    return fixture;
}

describe('ExperimentExplorerComponent', () => {
    it('opens experiments with errors first, collapses the valid ones, and toggles independently', async () => {
        const fixture = await render([
            experiment('a'),
            experiment('b', { status: 'error', errors: 1, issues: [{ code: 'invalid_row', severity: 'error', message: 'Row is invalid', column: null, source_row: 4, experiment: 'b', remediation: null }] }),
        ]);
        const root = fixture.nativeElement as HTMLElement;
        const headers = Array.from(root.querySelectorAll<HTMLButtonElement>('.explorer-header'));

        expect(headers.map((header) => header.getAttribute('aria-expanded'))).toEqual(['false', 'true']);
        expect(root.querySelector('.explorer-issues')?.textContent).toContain('row 4');

        headers[0].click();
        fixture.detectChanges();
        expect(root.querySelectorAll('.explorer-table')).toHaveLength(2);
        expect(root.textContent).toContain('source: 1 bar');

        headers[1].click();
        fixture.detectChanges();
        expect(root.querySelectorAll('.explorer-table')).toHaveLength(1);
    });

    it('expands and collapses every experiment from the toolbar', async () => {
        const fixture = await render([experiment('a'), experiment('b'), experiment('c')]);
        const root = fixture.nativeElement as HTMLElement;
        const [expandAll, collapseAll] = Array.from(root.querySelectorAll<HTMLButtonElement>('.explorer-toolbar button'));

        expandAll.click();
        fixture.detectChanges();
        expect(root.querySelectorAll('.explorer-table')).toHaveLength(3);

        collapseAll.click();
        fixture.detectChanges();
        expect(root.querySelectorAll('.explorer-table')).toHaveLength(0);
    });

    it('states when only part of an experiment is previewed', async () => {
        const fixture = await render([experiment('a', { totalRows: 120 })]);

        expect((fixture.nativeElement as HTMLElement).textContent).toContain('first 2 of 120 observations');
    });
});
