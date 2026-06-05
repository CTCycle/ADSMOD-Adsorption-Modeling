import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
    selector: 'adsmod-switch',
    standalone: true,
    template: `
        <div class="switch-container">
            <label class="switch">
                <input type="checkbox" [checked]="checked" (change)="handleChange($event)" />
                <span class="slider"></span>
            </label>
            @if (label) {
                <span>{{ label }}</span>
            }
        </div>
    `,
})
export class SwitchComponent {
    @Input() checked = false;
    @Input() label = '';
    @Output() readonly checkedChange = new EventEmitter<boolean>();

    protected handleChange(event: Event): void {
        const input = event.target as HTMLInputElement;
        this.checkedChange.emit(input.checked);
    }
}
