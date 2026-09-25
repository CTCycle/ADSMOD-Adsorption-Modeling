import { Component, ElementRef, effect, input, output, viewChild } from '@angular/core';

@Component({
    selector: 'adsmod-confirm-dialog',
    standalone: true,
    template: `
        @if (open()) {
            <div class="confirm-dialog-backdrop" (click)="closed.emit()">
                <section
                    #dialog
                    class="confirm-dialog"
                    role="alertdialog"
                    aria-modal="true"
                    aria-labelledby="confirm-dialog-title"
                    aria-describedby="confirm-dialog-message"
                    (click)="$event.stopPropagation()"
                    (keydown)="handleKeydown($event)"
                >
                    <h2 id="confirm-dialog-title">{{ title() }}</h2>
                    <p id="confirm-dialog-message">{{ message() }}</p>
                    <div class="confirm-dialog-actions">
                        <button #cancelButton class="button secondary" type="button" (click)="closed.emit()">Cancel</button>
                        <button class="confirm-destructive" type="button" (click)="confirmed.emit()">{{ confirmLabel() }}</button>
                    </div>
                </section>
            </div>
        }
    `,
})
export class ConfirmDialogComponent {
    readonly open = input(false);
    readonly title = input('Confirm action');
    readonly message = input('Are you sure you want to continue?');
    readonly confirmLabel = input('Confirm');
    readonly confirmed = output<void>();
    readonly closed = output<void>();

    private readonly cancelButton = viewChild<ElementRef<HTMLButtonElement>>('cancelButton');
    private returnFocusTarget: HTMLElement | null = null;

    constructor() {
        effect(() => {
            if (this.open()) {
                this.returnFocusTarget = document.activeElement instanceof HTMLElement
                    ? document.activeElement
                    : null;
                window.setTimeout(() => this.cancelButton()?.nativeElement.focus());
                return;
            }

            if (this.returnFocusTarget?.isConnected) {
                this.returnFocusTarget.focus();
            }
            this.returnFocusTarget = null;
        });
    }

    protected handleKeydown(event: KeyboardEvent): void {
        if (event.key === 'Escape') {
            event.preventDefault();
            this.closed.emit();
            return;
        }

        if (event.key !== 'Tab') {
            return;
        }

        const cancel = this.cancelButton()?.nativeElement;
        const confirm = event.currentTarget instanceof HTMLElement
            ? event.currentTarget.querySelector<HTMLButtonElement>('.confirm-destructive')
            : null;
        if (!cancel || !confirm) {
            event.preventDefault();
        } else if (event.shiftKey && document.activeElement === cancel) {
            event.preventDefault();
            confirm.focus();
        } else if (!event.shiftKey && document.activeElement === confirm) {
            event.preventDefault();
            cancel.focus();
        }
    }
}
