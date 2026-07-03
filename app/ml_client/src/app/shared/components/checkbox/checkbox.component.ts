import { Component, input, output } from '@angular/core';

let checkboxIdCounter = 0;

@Component({
    selector: 'adsmod-checkbox',
    standalone: true,
    template: `
        <div class="checkbox-container">
            <input
                [id]="inputId()"
                type="checkbox"
                [checked]="checked()"
                [disabled]="disabled()"
                (change)="handleChange($event)"
            />
            <label [for]="inputId()" style="margin-bottom: 0; cursor: pointer;">
                {{ label() }}
            </label>
        </div>
    `,
})
export class CheckboxComponent {
    readonly label = input.required<string>();
    readonly checked = input(false);
    readonly disabled = input(false);
    readonly inputId = input(`adsmod-checkbox-${checkboxIdCounter++}`);
    readonly checkedChange = output<boolean>();

    protected handleChange(event: Event): void {
        this.checkedChange.emit((event.target as HTMLInputElement).checked);
    }
}
