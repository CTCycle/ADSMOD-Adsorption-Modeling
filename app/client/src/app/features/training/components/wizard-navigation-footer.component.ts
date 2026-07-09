import { Component, input, output } from '@angular/core';

@Component({
    selector: 'adsmod-wizard-navigation-footer',
    standalone: true,
    template: `
        <div class="wizard-footer">
            <button class="secondary" type="button" [disabled]="isLoading()" (click)="closed.emit()">Cancel</button>
            @if (!isFirstPage()) {
                <button class="secondary" type="button" [disabled]="isLoading()" (click)="previous.emit()">Previous</button>
            }
            @if (!isLastPage()) {
                <button class="primary" type="button" [disabled]="isLoading()" (click)="next.emit()">Next</button>
            }
            @if (isLastPage()) {
                <button class="primary" type="button" [disabled]="isLoading()" (click)="confirmed.emit()">
                    {{ isLoading() ? confirmLoadingLabel() : confirmIdleLabel() }}
                </button>
            }
        </div>
    `,
})
export class WizardNavigationFooterComponent {
    readonly isLoading = input(false);
    readonly isFirstPage = input(false);
    readonly isLastPage = input(false);
    readonly confirmIdleLabel = input.required<string>();
    readonly confirmLoadingLabel = input.required<string>();
    readonly closed = output<void>();
    readonly next = output<void>();
    readonly previous = output<void>();
    readonly confirmed = output<void>();
}
