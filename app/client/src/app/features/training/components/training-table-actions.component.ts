import { Component, input, output } from '@angular/core';

@Component({
    selector: 'adsmod-training-table-actions',
    standalone: true,
    template: `
        <div class="split-table-actions-wrap">
            <button class="icon-action-button" type="button" [title]="viewTitle()" (click)="handleView($event)">
                <span aria-hidden="true">i</span>
            </button>
            <button class="icon-action-button" type="button" [title]="deleteTitle()" (click)="handleDelete($event)">
                <span aria-hidden="true">x</span>
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
