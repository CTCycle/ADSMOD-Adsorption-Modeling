import { Component, input, output } from '@angular/core';

let numberInputIdCounter = 0;

@Component({
    selector: 'adsmod-number-input',
    standalone: true,
    template: `
        <div class="number-input-wrapper">
            <label class="field-label" [for]="inputId">{{ label() }}</label>
            <input
                [id]="inputId"
                class="number-input-field"
                type="number"
                [value]="value()"
                [min]="min()"
                [max]="max()"
                [step]="step()"
                [disabled]="disabled()"
                (change)="handleChange($event)"
            />
        </div>
    `,
})
export class NumberInputComponent {
    readonly label = input.required<string>();
    readonly value = input.required<number>();
    readonly min = input<number | undefined>(undefined);
    readonly max = input<number | undefined>(undefined);
    readonly step = input(0.0001);
    readonly precision = input(4);
    readonly disabled = input(false);
    readonly valueChange = output<number>();

    protected readonly inputId = `adsmod-number-input-${numberInputIdCounter++}`;

    protected handleChange(event: Event): void {
        const rawValue = Number.parseFloat((event.target as HTMLInputElement).value);
        if (!Number.isNaN(rawValue)) {
            this.valueChange.emit(Number(rawValue.toFixed(this.precision())));
        }
    }
}