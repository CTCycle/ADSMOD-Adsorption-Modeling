import { Component, input, output } from '@angular/core';

@Component({
    selector: 'adsmod-training-table-actions',
    standalone: true,
    template: `
        <div class="split-table-actions-wrap">
            <button class="button secondary training-row-action" type="button" [title]="viewTitle()" [attr.aria-label]="viewTitle()" (click)="handleView($event)">
                View
            </button>
            <button class="button quiet danger training-row-action" type="button" [title]="deleteTitle()" [attr.aria-label]="deleteTitle()" (click)="handleDelete($event)">
                Delete
            </button>
        </div>
    `,
})
export class TrainingTableActionsComponent {
    readonly viewTitle = input.required<string>();
    readonly deleteTitle = input.required<string>();
    readonly view = output<void>();
    readonly delete = output<void>();

    protected handleView(event: Event): void {
        event.stopPropagation();
        this.view.emit();
    }

    protected handleDelete(event: Event): void {
        event.stopPropagation();
        this.delete.emit();
    }
}
