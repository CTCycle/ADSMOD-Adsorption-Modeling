import { Component, input, output } from '@angular/core';

@Component({
    selector: 'adsmod-split-selection-card',
    standalone: true,
    template: `
        <div class="section-container">
            @if (!hideHeader()) {
                <h3 class="split-selection-title">{{ title() }}</h3>
            }
            @if (!hideHeader() && subtitle()) {
                <p class="split-selection-subtitle">{{ subtitle() }}</p>
            }
            <div class="split-selection-card">
                <div class="split-selection-card-left">
                    <div class="split-selection-card-toolbar">
                        @if (showRefresh()) {
                            <button
                                type="button"
                                class="split-selection-refresh-button"
                                (click)="refresh.emit()"
                            >
                                Refresh
                            </button>
                        }
                    </div>
                    <div class="split-selection-card-content">
                        <ng-content select="[card-left]"></ng-content>
                    </div>
                </div>
                <div class="split-selection-card-right">
                    <ng-content select="[card-right]"></ng-content>
                </div>
            </div>
        </div>
    `,
})
export class SplitSelectionCardComponent {
    readonly title = input.required<string>();
    readonly subtitle = input('');
    readonly hideHeader = input(false);
    readonly showRefresh = input(false);
    readonly refresh = output<void>();
}
