import { Component, EventEmitter, Input, Output } from '@angular/core';

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
                    [checked]="checked"
                    [disabled]="disabled"
                    [attr.aria-label]="label || ariaLabel"
                    (change)="handleChange($event)"
                />
                <span class="slider"></span>
            </label>
            @if (label) {
                <label [for]="inputId">{{ label }}</label>
            }
        </div>
    `,
})
export class SwitchComponent {
    @Input() checked = false;
    @Input() disabled = false;
    @Input() label = '';
    @Input() ariaLabel = 'Toggle option';
    @Output() readonly checkedChange = new EventEmitter<boolean>();

    protected readonly inputId = `adsmod-switch-${switchIdCounter++}`;

    protected handleChange(event: Event): void {
        const input = event.target as HTMLInputElement;
        this.checkedChange.emit(input.checked);
    }
}