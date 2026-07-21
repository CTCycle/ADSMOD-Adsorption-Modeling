import { TestBed } from '@angular/core/testing';
import { describe, expect, it } from 'vitest';
import type { DatasetSummary } from '../../models/dataset.model';
import { DatasetManagementComponent } from './dataset-management.component';

const dataset: DatasetSummary = {
    name: 'silica.csv', source: 'uploaded', created_at: '2026-07-12T00:00:00Z', row_count: 2, column_count: 2,
    tags: ['demo'], description: 'Example measurements',
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
        expect(root.textContent).toContain('Use + to add a CSV, TXT, XLSX, or JSON dataset.');
        expect(root.querySelector('button[aria-label="Add dataset"]')).not.toBeNull();
        expect(root.querySelector('button')?.textContent).toContain('Refresh');
        expect(root.textContent).not.toContain('No user datasets uploaded yet');
    });

    it('renders each dataset as a record with row actions', async () => {
        const fixture = await createComponent();
        fixture.componentRef.setInput('datasets', [dataset]);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.querySelector('.dataset-record')).not.toBeNull();
        expect(root.textContent).toContain('silica.csv');
        expect(root.textContent).toContain('2 rows');
        expect(root.textContent).toContain('2 columns');
        expect(root.querySelector('button[aria-label="Open spreadsheet for silica.csv"]')).not.toBeNull();
        expect(root.querySelector('button[aria-label="Edit metadata for silica.csv"]')).not.toBeNull();
        expect(root.querySelector('.dataset-metadata-editor')).toBeNull();
        expect(root.querySelector('.spreadsheet-table')).toBeNull();
    });

    it('opens metadata editing from a dataset action', async () => {
        const fixture = await createComponent();
        fixture.componentRef.setInput('datasets', [dataset]);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        root.querySelector<HTMLButtonElement>('button[aria-label="Edit metadata for silica.csv"]')?.click();
        fixture.detectChanges();
        expect(root.querySelector('.dataset-metadata-editor')).not.toBeNull();
        expect(root.textContent).toContain('Edit metadata');
    });
});
