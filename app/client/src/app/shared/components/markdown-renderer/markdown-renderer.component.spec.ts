import { TestBed } from '@angular/core/testing';
import { MarkdownRendererComponent } from './markdown-renderer.component';

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
