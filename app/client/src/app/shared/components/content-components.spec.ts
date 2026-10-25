// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { InfoModalComponent } from './info-modal/info-modal.component';
import { MarkdownRendererComponent } from './markdown-renderer/markdown-renderer.component';

// Information modal behavior

describe('InfoModalComponent', () => {
    it('renders only defined data entries', async () => {
        await TestBed.configureTestingModule({
            imports: [InfoModalComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(InfoModalComponent);
        fixture.componentRef.setInput('isOpen', true);
        fixture.componentRef.setInput('title', 'Dataset Metadata');
        fixture.componentRef.setInput('data', {
            Name: 'Dataset A',
            Count: 12,
            Empty: null,
        });
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.textContent).toContain('Dataset Metadata');
        expect(root.textContent).toContain('Dataset A');
        expect(root.textContent).toContain('12');
        expect(root.textContent).not.toContain('Empty');
    });
});

// Markdown rendering behavior

describe('MarkdownRendererComponent', () => {
    it('renders markdown tables and headings', async () => {
        await TestBed.configureTestingModule({
            imports: [MarkdownRendererComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(MarkdownRendererComponent);
        fixture.componentRef.setInput('content', '# Title\n\n| A | B |\n| - | - |\n| 1 | 2 |');
        fixture.componentRef.setInput('className', 'markdown-body');
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelector('.markdown-body h1')?.textContent).toContain('Title');
        expect(root.querySelector('.markdown-body table')).not.toBeNull();
    });

    it('sanitizes unsafe html content', async () => {
        await TestBed.configureTestingModule({
            imports: [MarkdownRendererComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(MarkdownRendererComponent);
        fixture.componentRef.setInput('content', 'Safe<script>window.__bad = true</script>');
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelector('script')).toBeNull();
        expect(root.textContent).toContain('Safe');
    });
});
