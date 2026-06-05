import { Component, input, output } from '@angular/core';

@Component({
    selector: 'adsmod-switch',
    standalone: true,
    template: `
        <div class="switch-container">
            <label class="switch">
                <input
                    type="checkbox"
                    [checked]="checked()"
                    [disabled]="disabled()"
                    (change)="handleChange($event)"
                />
                <span class="slider"></span>
            </label>
            @if (label()) {
                <span>{{ label() }}</span>
            }
        </div>
    `,
})
export class SwitchComponent {
    readonly checked = input(false);
    readonly disabled = input(false);
    readonly label = input('');
    readonly checkedChange = output<boolean>();

    protected handleChange(event: Event): void {
        this.checkedChange.emit((event.target as HTMLInputElement).checked);
    }
}
