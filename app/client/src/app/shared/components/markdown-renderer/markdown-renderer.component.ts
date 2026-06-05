import { Component, Input, OnChanges, inject } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

@Component({
    selector: 'adsmod-markdown-renderer',
    standalone: true,
    template: '<div [class]="className" [innerHTML]="html"></div>',
})
export class MarkdownRendererComponent implements OnChanges {
    @Input() content = '';
    @Input() className = '';
    protected html: SafeHtml = '';
    private readonly sanitizer = inject(DomSanitizer);

    ngOnChanges(): void {
        const rendered = marked.parse(this.content, { gfm: true, async: false }) as string;
        const purified = DOMPurify.sanitize(rendered);
        this.html = this.sanitizer.bypassSecurityTrustHtml(purified);
    }
}
