// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, input, model } from '@angular/core';

export interface SegmentedOption<T extends string> {
    value: T;
    label: string;
}

/** Compact two- or three-way switch used for acquisition modes. */
@Component({
    selector: 'adsmod-segmented-control',
    standalone: true,
    template: `
        <div class="segmented" role="group" [attr.aria-label]="label()">
            @for (option of options(); track option.value) {
                <button type="button" [attr.aria-pressed]="option.value === value()" (click)="value.set(option.value)">{{ option.label }}</button>
            }
        </div>
    `,
})
export class SegmentedControlComponent<T extends string = string> {
    readonly options = input.required<readonly SegmentedOption<T>[]>();
    readonly label = input.required<string>();
    readonly value = model.required<T>();
}
