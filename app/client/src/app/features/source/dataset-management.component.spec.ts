import { TestBed } from '@angular/core/testing';
import { describe, expect, it } from 'vitest';
import type { DatasetRowsPage, DatasetSummary } from '../../models/dataset.model';
import { DatasetManagementComponent } from './dataset-management.component';

const dataset: DatasetSummary = {
    name: 'silica.csv', source: 'uploaded', created_at: '2026-07-12T00:00:00Z', row_count: 2, column_count: 2,
    tags: ['demo'], description: 'Example measurements',
};

const page: DatasetRowsPage = {
    dataset_name: dataset.name, columns: ['row_id', 'pressure', 'capacity'],
    rows: [{ row_id: 1, pressure: '1', capacity: '0.4' }], offset: 0, limit: 100, total_rows: 1,
};

async function createComponent(): Promise<ReturnType<typeof TestBed.createComponent<DatasetManagementComponent>>> {
    await TestBed.configureTestingModule({ imports: [DatasetManagementComponent] }).compileComponents();
    return TestBed.createComponent(DatasetManagementComponent);
}

describe('DatasetManagementComponent', () => {
    it('explains how to begin when the dataset list is empty and keeps refresh available', async () => {
        const fixture = await createComponent();
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.textContent).toContain('Your workspace is ready');
        expect(root.textContent).toContain('Upload a CSV, TXT, XLSX, or JSON file above');
        expect(root.querySelector('button')?.textContent).toContain('Refresh');
        expect(root.textContent).not.toContain('No user datasets uploaded yet');
    });

    it('uses compact placeholders when no dataset is selected', async () => {
        const fixture = await createComponent();
        fixture.componentRef.setInput('datasets', [dataset]);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.textContent).toContain('Select a dataset to edit tags and description.');
        expect(root.textContent).toContain('Select a dataset to edit its rows.');
        expect(root.querySelector('.dataset-metadata-editor')).toBeNull();
        expect(root.querySelector('.spreadsheet-table')).toBeNull();
    });

    it('renders metadata and spreadsheet editing for the selected dataset', async () => {
        const fixture = await createComponent();
        fixture.componentRef.setInput('datasets', [dataset]);
        fixture.componentRef.setInput('selected', dataset.name);
        fixture.componentRef.setInput('page', page);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.querySelector('.dataset-metadata-editor')).not.toBeNull();
        expect(root.querySelector('.spreadsheet-table')).not.toBeNull();
        expect(root.textContent).toContain('pressure');
        expect(root.textContent).toContain('capacity');
    });
});
