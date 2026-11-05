// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, input, output } from '@angular/core';
import type { ModelBounds, ModelCatalogParameter, ParameterBounds } from '../../../models/fitting.model';

/** Editable lower/upper search bounds per parameter, in canonical units (Pa, mol/kg). */
@Component({
    selector: 'adsmod-model-config-form',
    standalone: true,
    template: `
        <div class="bounds-form" [attr.aria-label]="label()" role="group">
            <div class="bounds-head" aria-hidden="true">
                <span>Parameter</span><span>Lower</span><span>Upper</span>
            </div>
            @for (parameter of parameters(); track parameter.name) {
                <div class="bounds-row" [class.changed]="isChanged(parameter)">
                    <div class="bounds-name">
                        <strong>{{ parameter.name }}</strong>
                        <span [title]="parameter.label">{{ parameter.label }}@if (parameter.unit && parameter.unit !== '1') { · {{ parameter.unit }}}</span>
                    </div>
                    @for (bound of boundKeys; track bound) {
                        <input
                            class="text-input bounds-input"
                            type="number"
                            step="any"
                            inputmode="decimal"
                            [disabled]="disabled()"
                            [attr.aria-label]="parameter.name + ' ' + bound + ' bound'"
                            [value]="value()[parameter.name]?.[bound] ?? parameter[bound]"
                            (change)="set(parameter, bound, $any($event.target).value)"
                        />
                    }
                </div>
            }
        </div>
    `,
    styles: [`
        .bounds-form { display: grid; gap: 0.35rem; }
        .bounds-head, .bounds-row { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr); gap: 0.4rem; align-items: center; }
        .bounds-head { padding-left: 0.4rem; color: var(--text-tertiary); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
        .bounds-row { padding: 0.15rem 0 0.15rem 0.4rem; border-left: 2px solid transparent; }
        .bounds-row.changed { border-left-color: var(--primary-600); }
        .bounds-name { display: grid; min-width: 0; line-height: 1.2; }
        .bounds-name strong { font-size: 0.82rem; }
        .bounds-name span { overflow: hidden; color: var(--text-secondary); font-size: 0.7rem; text-overflow: ellipsis; white-space: nowrap; }
        .bounds-input { width: 100%; min-width: 0; padding: 0.3rem 0.45rem; font-size: 0.8rem; font-variant-numeric: tabular-nums; }
    `],
})
export class ModelConfigFormComponent {
    readonly parameters = input.required<ModelCatalogParameter[]>();
    readonly value = input.required<ModelBounds>();
    /** Bounds the edits are compared against, to mark changed rows. */
    readonly reference = input<ModelBounds>({});
    readonly label = input('Parameter bounds');
    readonly disabled = input(false);
    readonly valueChange = output<ModelBounds>();
    protected readonly boundKeys = ['lower', 'upper'] as const;

    protected isChanged(parameter: ModelCatalogParameter): boolean {
        const current = this.value()[parameter.name];
        const reference = this.reference()[parameter.name];
        return Boolean(current && reference && (current.lower !== reference.lower || current.upper !== reference.upper));
    }

    protected set(parameter: ModelCatalogParameter, bound: keyof ParameterBounds, raw: string): void {
        const parsed = raw.trim() === '' ? Number.NaN : Number(raw);
        const current = this.value()[parameter.name] ?? { lower: parameter.lower, upper: parameter.upper };
        this.valueChange.emit({ ...this.value(), [parameter.name]: { ...current, [bound]: parsed } });
    }
}
