import { Component, input } from '@angular/core';

@Component({
    selector: 'adsmod-metric-card',
    standalone: true,
    template: `
        <div class="metric-card">
            <div class="metric-label">{{ label() }}</div>
            <div class="metric-value" [style.color]="color()">{{ value() }}</div>
        </div>
    `,
})
export class MetricCardComponent {
    readonly label = input.required<string>();
    readonly value = input.required<string>();
    readonly color = input('var(--primary-600)');
}
