import { Component, EventEmitter, Input, Output } from '@angular/core';

let numberInputIdCounter = 0;

@Component({
    selector: 'adsmod-number-input',
    standalone: true,
    template: `
        <div class="number-input-wrapper">
            <label class="field-label" [for]="inputId">{{ label }}</label>
            <input
                [id]="inputId"
                class="number-input-field"
                type="number"
                [value]="value"
                [min]="min"
                [max]="max"
                [step]="step"
                [disabled]="disabled"
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
    @Input() disabled = false;
    @Output() readonly valueChange = new EventEmitter<number>();

    protected readonly inputId = `adsmod-number-input-${numberInputIdCounter++}`;

    protected handleInput(event: Event): void {
        const input = event.target as HTMLInputElement;
        const nextValue = Number.parseFloat(input.value);
        if (!Number.isNaN(nextValue)) {
            this.valueChange.emit(Number(nextValue.toFixed(this.precision)));
        }
    }
}