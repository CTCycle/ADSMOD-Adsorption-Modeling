// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
    selector: 'adsmod-data-pagination',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="data-pagination">
            <span class="pagination-summary" aria-live="polite">{{ summary }}</span>
            <label class="pagination-size">
                Rows per page
                    <select
                        [disabled]="disabled"
                        autocomplete="off"
                        (change)="selectPageSize($event)"
                    >
                    @for (size of options; track size) {
                        <option [value]="size" [selected]="size === pageSize">{{ size }}</option>
                    }
                </select>
            </label>
            <div class="pagination-controls">
                <button
                    class="pagination-button"
                    type="button"
                    [disabled]="disabled || page <= 1"
                    (click)="goTo(page - 1)"
                >
                    Previous
                </button>
                <span class="pagination-page">Page {{ page }} of {{ pageCount }}</span>
                <button
                    class="pagination-button"
                    type="button"
                    [disabled]="disabled || page >= pageCount"
                    (click)="goTo(page + 1)"
                >
                    Next
                </button>
            </div>
        </div>
    `,
    styles: [
        `
            :host {
                display: block;
            }
            .data-pagination {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 10px 14px;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid var(--border-color, #e3e7ee);
                color: var(--text-secondary, #667085);
                font-size: 0.75rem;
            }
            .pagination-summary,
            .pagination-page {
                font-variant-numeric: tabular-nums;
            }
            .pagination-summary {
                min-width: 9rem;
            }
            .pagination-size {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                color: var(--text-secondary, #667085);
                font-size: 0.72rem;
                font-weight: 600;
            }
            .pagination-size select {
                min-height: 30px;
                height: 30px;
                padding: 0 8px;
                border: 1px solid var(--border-color, #d8dee8);
                border-radius: 6px;
                background: var(--surface);
                color: var(--text-primary, #18202c);
                font: inherit;
                font-size: 0.75rem;
            }
            .pagination-controls {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-left: auto;
            }
            .pagination-page {
                white-space: nowrap;
            }
            .pagination-button {
                min-height: 30px;
                height: 30px;
                padding: 0 10px;
                border: 1px solid var(--border-color, #d8dee8);
                border-radius: 6px;
                background: var(--surface);
                color: var(--text-primary, #18202c);
                font: inherit;
                font-size: 0.75rem;
                font-weight: 600;
                cursor: pointer;
            }
            .pagination-button:hover:not(:disabled) {
                background: var(--surface-subtle, #f7f9fc);
            }
            .pagination-button:focus-visible,
            .pagination-size select:focus-visible {
                outline: 3px solid rgba(50, 112, 218, 0.14);
                outline-offset: 1px;
                border-color: var(--primary-300);
            }
            .pagination-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            @media (max-width: 600px) {
                .pagination-controls {
                    width: 100%;
                    justify-content: flex-end;
                    margin-left: 0;
                }
            }
        `,
    ],
})
export class DataPaginationComponent {
    @Input() page = 1;
    @Input() pageSize = 10;
    @Input() total = 0;
    @Input() itemLabel = 'record';
    @Input() pageSizes: readonly number[] = [5, 10, 25];
    @Input() disabled = false;
    @Output() readonly pageChange = new EventEmitter<number>();
    @Output() readonly pageSizeChange = new EventEmitter<number>();

    protected get options(): number[] {
        const allowed = [...new Set(this.pageSizes)]
            .filter((size) => Number.isFinite(size) && size > 0 && size <= 25)
            .sort((left, right) => left - right);
        return allowed.length ? allowed : [5, 10, 25];
    }

    protected get pageCount(): number {
        return Math.max(1, Math.ceil(this.total / Math.max(1, this.pageSize)));
    }

    protected get firstItem(): number {
        if (!this.total) return 0;
        const page = Math.min(Math.max(1, this.page), this.pageCount);
        return (page - 1) * this.pageSize + 1;
    }

    protected get lastItem(): number {
        if (!this.total) return 0;
        const page = Math.min(Math.max(1, this.page), this.pageCount);
        return Math.min(this.total, page * this.pageSize);
    }

    protected get itemName(): string {
        return this.total === 1 ? this.itemLabel : `${this.itemLabel}s`;
    }

    protected get summary(): string {
        if (!this.total) return `0 ${this.itemLabel}s`;
        return `${this.firstItem}–${this.lastItem} of ${this.total} ${this.itemName}`;
    }

    protected goTo(page: number): void {
        if (this.disabled) return;
        const next = Math.min(Math.max(1, page), this.pageCount);
        if (next !== this.page) this.pageChange.emit(next);
    }

    protected selectPageSize(event: Event): void {
        const value = Number((event.target as HTMLSelectElement).value);
        if (Number.isFinite(value) && value !== this.pageSize) {
            this.pageSizeChange.emit(value);
        }
    }
}
