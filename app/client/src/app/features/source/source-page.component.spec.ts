import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { SourcePageComponent } from './source-page.component';

function storeMock() {
    return {
        pendingFile: signal<File | null>(null), isDatasetUploading: signal(false), userDatasets: signal([]),
        selectedDataset: signal<string | null>(null), editorPage: signal(null), datasetStats: signal('Select a dataset to view its summary.'),
        managementStatus: signal(''), setPendingFile: vi.fn(), uploadPendingDataset: vi.fn(), refreshUserDatasets: vi.fn(),
        openDataset: vi.fn(), deleteDataset: vi.fn(), renameDataset: vi.fn(), saveMetadata: vi.fn(), loadDatasetPage: vi.fn(), saveMutations: vi.fn(),
    } as unknown as CoreWorkspaceStore;
}

describe('SourcePageComponent', () => {
    it('shows the compact add action and statistics placeholder without datasets', async () => {
        const store = storeMock();
        await TestBed.configureTestingModule({ imports: [SourcePageComponent], providers: [{ provide: CoreWorkspaceStore, useValue: store }] }).compileComponents();
        const fixture = TestBed.createComponent(SourcePageComponent);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.querySelector('button[aria-label="Add dataset"]')).not.toBeNull();
        expect(root.textContent).not.toContain('Choose a dataset file');
        expect(root.textContent).toContain('Select a dataset to see row and column summaries.');
        expect(root.querySelector('[role="alert"]')).toBeNull();
    });

    it('shows accessible status feedback', async () => {
        const store = storeMock();
        store.pendingFile.set(new File(['pressure,capacity'], 'measurements.csv', { type: 'text/csv' }));
        store.managementStatus.set('Upload failed.');
        await TestBed.configureTestingModule({ imports: [SourcePageComponent], providers: [{ provide: CoreWorkspaceStore, useValue: store }] }).compileComponents();
        const fixture = TestBed.createComponent(SourcePageComponent);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.querySelector('[role="alert"]')?.textContent).toContain('Upload failed.');
        expect(root.querySelector('.source-upload-submit')).toBeNull();
    });

    it('renders selected dataset statistics without the removed upload card', async () => {
        const store = storeMock();
        store.userDatasets.set([{ name: 'measurements.csv', source: 'uploaded', created_at: '', row_count: 1, column_count: 2, tags: [], description: '' }]);
        store.selectedDataset.set('measurements.csv');
        store.datasetStats.set('Rows: 1\nColumns: 2');
        await TestBed.configureTestingModule({ imports: [SourcePageComponent], providers: [{ provide: CoreWorkspaceStore, useValue: store }] }).compileComponents();
        const fixture = TestBed.createComponent(SourcePageComponent);
        fixture.detectChanges();
        const root = fixture.nativeElement as HTMLElement;

        expect(root.textContent).toContain('Rows: 1');
        expect(root.textContent).not.toContain('Select a dataset to see row and column summaries.');
        expect(root.querySelector('.source-upload-card')).toBeNull();
    });
});
