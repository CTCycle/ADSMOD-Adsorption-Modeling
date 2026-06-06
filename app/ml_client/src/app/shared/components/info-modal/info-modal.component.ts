import { Component, computed, input, output } from '@angular/core';
import type { InfoModalData, InfoModalValue } from '../../../models/json.model';

interface ModalEntry {
    key: string;
    value: InfoModalValue;
}

@Component({
    selector: 'adsmod-info-modal',
    standalone: true,
    template: `
        @if (isOpen() && data()) {
            <div class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="info-modal-title">
                <div class="info-modal-content">
                    <div class="info-modal-header">
                        <h4 id="info-modal-title">{{ title() }}</h4>
                        <button class="info-modal-close-btn" type="button" (click)="closed.emit()" aria-label="Close">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>

                    <div class="info-modal-body">
                        @for (entry of entries(); track entry.key) {
                            <div class="info-row" [class.full-width]="isFullWidth(entry)">
                                <div class="info-header-row">
                                    <div class="info-icon-wrapper" [innerHTML]="iconFor(entry.key)"></div>
                                    <span class="info-label">{{ entry.key.replaceAll('_', ' ') }}</span>
                                </div>
                                @if (isObjectValue(entry.value)) {
                                    <div class="info-object-container">
                                        <pre>{{ stringify(entry.value) }}</pre>
                                    </div>
                                } @else {
                                    <span class="info-value">{{ formatValue(entry.value) }}</span>
                                }
                            </div>
                        }
                    </div>

                    <div class="info-modal-footer">
                        <button class="primary" type="button" (click)="closed.emit()">Done</button>
                    </div>
                </div>
            </div>
        }
    `,
})
export class InfoModalComponent {
    readonly isOpen = input(false);
    readonly title = input('');
    readonly data = input<InfoModalData | null>(null);
    readonly closed = output<void>();

    protected readonly entries = computed<ModalEntry[]>(() =>
        Object.entries(this.data() ?? {})
            .filter(([, value]) => value !== null && value !== undefined)
            .map(([key, value]) => ({ key, value }))
    );

    protected isObjectValue(value: InfoModalValue): boolean {
        return typeof value === 'object' && value !== null;
    }

    protected isFullWidth(entry: ModalEntry): boolean {
        return this.isObjectValue(entry.value) || entry.key.length > 20;
    }

    protected stringify(value: InfoModalValue): string {
        return JSON.stringify(value, null, 2);
    }

    protected formatValue(value: InfoModalValue): string {
        if (typeof value === 'number' && !Number.isInteger(value)) {
            return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
        }
        return `${value}`;
    }

    protected iconFor(key: string): string {
        const lowerKey = key.toLowerCase();
        if (lowerKey.includes('label') || lowerKey.includes('name')) {
            return this.iconTag;
        }
        if (lowerKey.includes('created') || lowerKey.includes('date')) {
            return this.iconCalendar;
        }
        if (lowerKey.includes('samples') || lowerKey.includes('count')) {
            return this.iconDatabase;
        }
        if (lowerKey.includes('fraction') || lowerKey.includes('size')) {
            return this.iconActivity;
        }
        if (lowerKey.includes('max') || lowerKey.includes('min') || lowerKey.includes('length')) {
            return this.iconMaximize;
        }
        return this.iconDatabase;
    }

    private readonly iconTag = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>`;
    private readonly iconCalendar = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>`;
    private readonly iconDatabase = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>`;
    private readonly iconActivity = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>`;
    private readonly iconMaximize = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>`;
}
