import { Component, Input, OnChanges, inject } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import DOMPurify from 'dompurify';
import katex from 'katex';

@Component({
    selector: 'adsmod-equation-renderer',
    standalone: true,
    template: `
        @if (renderedHtml) {
            <div class="equation-container" [innerHTML]="renderedHtml"></div>
        } @else {
            <div class="equation-fallback">
                <code>{{ latex }}</code>
            </div>
        }
    `,
})
export class EquationRendererComponent implements OnChanges {
    @Input({ required: true }) latex = '';
    @Input() displayMode = true;
    protected renderedHtml: SafeHtml | null = null;
    private readonly sanitizer = inject(DomSanitizer);

    ngOnChanges(): void {
        try {
            const rawHtml = katex.renderToString(this.latex, {
                throwOnError: false,
                displayMode: this.displayMode,
                output: 'html',
            });
            const sanitized = DOMPurify.sanitize(rawHtml);
            this.renderedHtml = sanitized ? this.sanitizer.bypassSecurityTrustHtml(sanitized) : null;
        } catch (error) {
            console.error('KaTeX rendering error:', error);
            this.renderedHtml = null;
        }
    }
}
