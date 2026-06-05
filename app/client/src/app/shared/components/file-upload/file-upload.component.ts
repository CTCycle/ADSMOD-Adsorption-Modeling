import { Component, EventEmitter, Input, Output } from '@angular/core';

let fileUploadIdCounter = 0;

@Component({
    selector: 'adsmod-file-upload',
    standalone: true,
    template: `
        <div class="file-upload">
            <input
                [id]="inputId"
                type="file"
                [accept]="accept"
                [disabled]="disabled"
                [attr.aria-label]="label"
                (change)="handleChange($event)"
            />
            <label class="file-upload-label" [for]="inputId">
                <span aria-hidden="true">[file]</span>
                <span>{{ label }}</span>
            </label>
        </div>
    `,
})
export class FileUploadComponent {
    @Input({ required: true }) label = '';
    @Input() accept = '';
    @Input() autoUpload = true;
    @Input() disabled = false;
    @Output() readonly fileSelected = new EventEmitter<File>();
    @Output() readonly fileUploaded = new EventEmitter<File>();

    protected readonly inputId = `adsmod-file-upload-${fileUploadIdCounter++}`;

    protected handleChange(event: Event): void {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        if (file) {
            this.fileSelected.emit(file);
            if (this.autoUpload) {
                this.fileUploaded.emit(file);
            }
        }
        input.value = '';
    }
}