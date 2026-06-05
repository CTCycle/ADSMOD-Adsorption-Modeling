import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
    selector: 'adsmod-file-upload',
    standalone: true,
    template: `
        <div class="file-upload">
            <input
                type="file"
                [accept]="accept"
                [disabled]="disabled"
                (change)="handleChange($event)"
            />
            <div class="file-upload-label">
                <span aria-hidden="true">[file]</span>
                <span>{{ label }}</span>
            </div>
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
