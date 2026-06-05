import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
    selector: 'adsmod-checkbox',
    standalone: true,
    template: `
        <div class="checkbox-container">
            <input
                type="checkbox"
                [id]="checkboxId"
                [checked]="checked"
                (change)="handleChange($event)"
            />
            <label [for]="checkboxId" style="margin-bottom: 0; cursor: pointer;">
                {{ label }}
            </label>
        </div>
    `,
})
export class CheckboxComponent {
    @Input({ required: true }) label = '';
    @Input() checked = false;
    @Output() readonly checkedChange = new EventEmitter<boolean>();

    protected get checkboxId(): string {
        return `checkbox-${this.label}`;
    }

    protected handleChange(event: Event): void {
        const input = event.target as HTMLInputElement;
        this.checkedChange.emit(input.checked);
    }
}
