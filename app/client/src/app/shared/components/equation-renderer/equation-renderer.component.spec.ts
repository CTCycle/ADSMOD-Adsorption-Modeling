import { TestBed } from '@angular/core/testing';
import katex from 'katex';
import { vi } from 'vitest';
import { EquationRendererComponent } from './equation-renderer.component';

describe('EquationRendererComponent', () => {
    it('renders latex through KaTeX', async () => {
        await TestBed.configureTestingModule({
            imports: [EquationRendererComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(EquationRendererComponent);
        fixture.componentRef.setInput('latex', 'x^2 + y^2');
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelector('.equation-container .katex')).not.toBeNull();
    });

    it('falls back to raw latex when KaTeX throws', async () => {
        const renderSpy = vi.spyOn(katex, 'renderToString').mockImplementation(() => {
            throw new Error('forced render failure');
        });

        await TestBed.configureTestingModule({
            imports: [EquationRendererComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(EquationRendererComponent);
        fixture.componentRef.setInput('latex', '\\badlatex');
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelector('.equation-fallback code')?.textContent).toContain('\\badlatex');
        renderSpy.mockRestore();
    });
});
