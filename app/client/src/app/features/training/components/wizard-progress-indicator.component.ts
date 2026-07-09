import { Component, computed, input } from '@angular/core';

@Component({
    selector: 'adsmod-wizard-progress-indicator',
    standalone: true,
    template: `
        <div class="wizard-page-indicator">
            @for (page of pageNumbers(); track page; let index = $index) {
                <span class="wizard-dot" [class.active]="currentPage() === index">{{ page }}</span>
                @if (index < pageNumbers().length - 1) {
                    <span class="wizard-dot-line" [class.active]="currentPage() > index"></span>
                }
            }
        </div>
    `,
})
export class WizardProgressIndicatorComponent {
    readonly currentPage = input.required<number>();
    readonly totalPages = input.required<number>();
    protected readonly pageNumbers = computed(() => Array.from({ length: this.totalPages() }, (_, index) => index + 1));
}
