import { Component, computed, input } from '@angular/core';

@Component({
    selector: 'adsmod-progress-bar',
    standalone: true,
    template: `
        <div class="inline-progress-bar-container">
            @if (label()) {
                <div class="inline-progress-bar-header">
                    <span class="progress-bar-label">{{ label() }}</span>
                    @if (showPercent()) {
                        <span class="progress-bar-percent">{{ clampedValue().toFixed(0) }}%</span>
                    }
                </div>
            }
            <div class="inline-progress-bar-track">
                <div
                    class="inline-progress-bar-fill"
                    [style.width.%]="clampedValue()"
                    [style.background-color]="color()"
                ></div>
            </div>
        </div>
    `,
})
export class ProgressBarComponent {
    readonly value = input(0);
    readonly label = input('');
    readonly color = input('var(--primary-600)');
    readonly showPercent = input(true);
    protected readonly clampedValue = computed(() => Math.min(100, Math.max(0, this.value())));
}
