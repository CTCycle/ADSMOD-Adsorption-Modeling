import { Component, EventEmitter, Input, Output } from '@angular/core';
import type { ModelParameters } from '../../models/fitting.model';
import { NumberInputComponent } from '../../shared/components/number-input/number-input.component';

@Component({
    selector: 'adsmod-model-config-form',
    standalone: true,
    imports: [NumberInputComponent],
    template: `
        @if (parameterEntries.length === 0) {
            <div class="config-form-empty">
                <p>Configuration is not available for this model.</p>
            </div>
        } @else {
            <div class="model-config-form" [id]="'config-form-' + modelId">
                <div class="config-form-fields">
                    @for (entry of parameterEntries; track entry[0]) {
                        <div class="parameter-row">
                            <div class="parameter-label">{{ entry[0] }}</div>
                            <div class="parameter-inputs">
                                <adsmod-number-input
                                    label="min"
                                    [value]="currentValue(entry[0], 'min', entry[1][0])"
                                    [min]="0"
                                    [step]="0.0001"
                                    [precision]="4"
                                    (valueChange)="setParameter(entry[0], 'min', $event)"
                                />
                                <adsmod-number-input
                                    label="max"
                                    [value]="currentValue(entry[0], 'max', entry[1][1])"
                                    [min]="0"
                                    [step]="0.0001"
                                    [precision]="4"
                                    (valueChange)="setParameter(entry[0], 'max', $event)"
                                />
                            </div>
                        </div>
                    }
                </div>
            </div>
        }
    `,
})
export class ModelConfigFormComponent {
    @Input({ required: true }) modelId = '';
    @Input({ required: true }) parameterDefaults: Record<string, [number, number]> = {};
    @Input({ required: true }) value: ModelParameters = {};
    @Output() readonly valueChange = new EventEmitter<ModelParameters>();

    protected get parameterEntries(): [string, [number, number]][] {
        return Object.entries(this.parameterDefaults);
    }

    protected currentValue(paramName: string, boundType: 'min' | 'max', fallback: number): number {
        return this.value[paramName]?.[boundType] ?? fallback;
    }

    protected setParameter(paramName: string, boundType: 'min' | 'max', nextValue: number): void {
        this.valueChange.emit({
            ...this.value,
            [paramName]: {
                ...this.value[paramName],
                [boundType]: nextValue,
            },
        });
    }
}
