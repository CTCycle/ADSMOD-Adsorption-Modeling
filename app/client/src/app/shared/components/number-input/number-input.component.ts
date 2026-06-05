import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
    selector: 'adsmod-number-input',
    standalone: true,
    template: `
        <div style="flex: 1; min-width: 140px;">
            <label>{{ label }}</label>
            <input
                type="number"
                [value]="value"
                [min]="min"
                [max]="max"
                [step]="step"
                (input)="handleInput($event)"
            />
        </div>
    `,
})
export class NumberInputComponent {
    @Input({ required: true }) label = '';
    @Input({ required: true }) value = 0;
    @Input() min?: number;
    @Input() max?: number;
    @Input() step = 0.0001;
    @Input() precision = 4;
    @Output() readonly valueChange = new EventEmitter<number>();

    protected handleInput(event: Event): void {
        const input = event.target as HTMLInputElement;
        const nextValue = Number.parseFloat(input.value);
        if (!Number.isNaN(nextValue)) {
            this.valueChange.emit(Number(nextValue.toFixed(this.precision)));
        }
    }
}
