import { Component, input, output } from '@angular/core';

let switchIdCounter = 0;

@Component({
    selector: 'adsmod-switch',
    standalone: true,
    template: `
        <div class="switch-container">
            <label class="switch">
                <input
                    [id]="inputId"
                    type="checkbox"
                    [checked]="checked()"
                    [disabled]="disabled()"
                    [attr.aria-label]="label() || ariaLabel()"
                    (change)="handleChange($event)"
                />
                <span class="slider"></span>
            </label>
            @if (label()) {
                <label [for]="inputId">{{ label() }}</label>
            }
        </div>
    `,
})
export class SwitchComponent {
    readonly checked = input(false);
    readonly disabled = input(false);
    readonly label = input('');
    readonly ariaLabel = input('Toggle option');
    readonly checkedChange = output<boolean>();

    protected readonly inputId = `adsmod-switch-${switchIdCounter++}`;

    protected handleChange(event: Event): void {
        this.checkedChange.emit((event.target as HTMLInputElement).checked);
    }
}
