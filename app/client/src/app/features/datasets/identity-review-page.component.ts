// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { IdentityEntity, IdentityReview, IdentityReviewStatus } from '../../models/materials.model';
import { fetchIdentityReviews, mergeIdentityReview, rejectIdentityReview, runIdentityAudit } from '../../services/materials.service';
import { DataPaginationComponent } from '../../shared/components/data-pagination/data-pagination.component';

type KindFilter = 'all' | 'adsorbate' | 'adsorbent';

const REASONS: Record<string, string> = {
    ambiguous_alias: 'Name belongs to several records',
    ambiguous_external_match: 'PubChem returned several compounds',
    curated_conflict: 'Known synonyms exist as separate records',
    curated_synonym: 'Known synonyms exist as separate records',
    formula_not_unique: 'A formula does not identify one compound',
    fuzzy_name: 'Possible misspelling',
    identifier_conflict: 'Identifiers disagree',
    ion_or_oxidation_state: 'Charge state must be confirmed',
    separator_variant: 'Names differ only in punctuation',
};

const STATUS_FILTERS: { value: IdentityReviewStatus; label: string }[] = [
    { value: 'open', label: 'Open' },
    { value: 'merged', label: 'Merged' },
    { value: 'rejected', label: 'Kept separate' },
];

@Component({
    selector: 'adsmod-identity-review-page',
    standalone: true,
    imports: [DataPaginationComponent, RouterLink],
    template: `
        <div class="data-page identity-page">
            <header class="identity-header">
                <div>
                    <a routerLink="/datasets" class="identity-back">Datasets</a>
                    <h1>Identity review</h1>
                    <p>Records that may describe the same species or material. Nothing is merged until you approve it.</p>
                </div>
                <button class="button secondary" type="button" [disabled]="busy()" (click)="audit()">Find duplicates</button>
            </header>
            @if (notice()) {
                <div class="dataset-status success" role="status">{{ notice() }}</div>
            }
            @if (error()) {
                <div class="dataset-status error" role="alert">{{ error() }}</div>
            }
            <div class="identity-filters">
                <div class="identity-tabs" role="group" aria-label="Review status">
                    @for (item of statusFilters; track item.value) {
                        <button type="button" [class.active]="status() === item.value" [attr.aria-pressed]="status() === item.value" (click)="setStatus(item.value)">{{ item.label }}</button>
                    }
                </div>
                <label>
                    <span class="sr-only">Record type</span>
                    <select [value]="kind()" (change)="setKind($any($event.target).value)">
                        <option value="all">Materials and species</option>
                        <option value="adsorbent">Materials</option>
                        <option value="adsorbate">Species</option>
                    </select>
                </label>
            </div>
            @if (loading() && !items().length) {
                <p class="identity-message" role="status">Loading…</p>
            } @else if (!items().length) {
                <p class="identity-message">{{ status() === 'open' ? 'Nothing needs review.' : 'No decisions yet.' }}</p>
            } @else {
                <ul class="identity-list" [attr.aria-busy]="loading()">
                    @for (review of items(); track review.id) {
                        <li class="console-card identity-card">
                            <div class="identity-reason">
                                <strong>{{ reasonLabel(review.reason) }}</strong>
                                <span>{{ review.kind === 'adsorbent' ? 'Material' : 'Species' }}@if (review.score !== null && review.score < 1) { · {{ percent(review.score) }} similar }</span>
                            </div>
                            @if (review.entity && review.candidate) {
                                <div class="identity-pair">
                                    @for (entity of [review.entity, review.candidate]; track entity.id) {
                                        <div class="identity-entity">
                                            <span class="identity-name">{{ entity.name }}</span>
                                            <span class="identity-meta">{{ experiments(entity) }} · {{ entity.identifier ?? 'No identifier' }}</span>
                                            @if (review.status === 'open') {
                                                @if (pending()?.reviewId === review.id && pending()?.keepId === entity.id) {
                                                    <span class="identity-confirm">
                                                        Move {{ experiments(other(review, entity)) }} here and remove the other record?
                                                        <button class="button primary" type="button" [disabled]="busy()" (click)="merge(review, entity)">Merge</button>
                                                        <button class="button secondary" type="button" [disabled]="busy()" (click)="pending.set(null)">Cancel</button>
                                                    </span>
                                                } @else {
                                                    <button class="button secondary" type="button" [disabled]="busy()" (click)="pending.set({ reviewId: review.id, keepId: entity.id })">Keep this record</button>
                                                }
                                            }
                                        </div>
                                    }
                                </div>
                            } @else {
                                <p class="identity-name">{{ review.entity?.name ?? review.raw_name }}</p>
                            }
                            @if (review.status === 'open') {
                                <div class="identity-actions">
                                    <button class="button secondary" type="button" [disabled]="busy()" (click)="reject(review)">{{ review.candidate ? 'Keep separate' : 'Dismiss' }}</button>
                                </div>
                            }
                        </li>
                    }
                </ul>
            }
            @if (total()) {
                <adsmod-data-pagination
                    [page]="page()"
                    [pageSize]="pageSize()"
                    [pageSizes]="pageSizes"
                    [total]="total()"
                    itemLabel="review"
                    [disabled]="loading()"
                    (pageChange)="goTo($event)"
                    (pageSizeChange)="setPageSize($event)"
                />
            }
        </div>
    `,
    styles: [
        `
            .identity-page { display: flex; flex-direction: column; gap: var(--spacing-md); }
            .identity-header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: var(--spacing-md); }
            .identity-header h1 { margin: 0.15rem 0; font-size: 1.25rem; }
            .identity-header p { margin: 0; color: var(--text-secondary); font-size: 0.875rem; }
            .identity-back { color: var(--primary-600); font-size: 0.8125rem; }
            .identity-filters { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--spacing-sm); }
            .identity-tabs { display: inline-flex; border-bottom: 1px solid var(--border-color); }
            .identity-tabs button { min-height: 2.5rem; padding: 0 0.85rem; border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: transparent; color: var(--text-secondary); font: inherit; font-size: 0.875rem; font-weight: 600; cursor: pointer; }
            .identity-tabs button.active { border-bottom-color: var(--primary-600); color: var(--text-primary); }
            .identity-list { display: grid; gap: var(--spacing-sm); margin: 0; padding: 0; list-style: none; }
            .identity-card { display: grid; gap: var(--spacing-sm); padding: var(--spacing-md); }
            .identity-reason { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--spacing-xs); font-size: 0.875rem; }
            .identity-reason span { color: var(--text-secondary); font-size: 0.8125rem; }
            .identity-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--spacing-sm); }
            .identity-entity { display: flex; flex-direction: column; align-items: flex-start; gap: 0.35rem; min-width: 0; padding: var(--spacing-sm); border: 1px solid var(--border-color); border-radius: 0.5rem; }
            .identity-name { margin: 0; font-size: 0.9375rem; font-weight: 600; overflow-wrap: anywhere; }
            .identity-meta { max-width: 100%; color: var(--text-secondary); font-size: 0.8125rem; overflow-wrap: anywhere; }
            .identity-confirm { display: flex; flex-wrap: wrap; align-items: center; gap: var(--spacing-xs); font-size: 0.8125rem; }
            .identity-actions { display: flex; justify-content: flex-end; }
            .identity-message { color: var(--text-secondary); font-size: 0.875rem; }
            @media (max-width: 640px) {
                .identity-pair { grid-template-columns: minmax(0, 1fr); }
            }
        `,
    ],
})
export class IdentityReviewPageComponent {
    protected readonly statusFilters = STATUS_FILTERS;
    protected readonly pageSizes = [10, 25, 50] as const;
    protected readonly items = signal<IdentityReview[]>([]);
    protected readonly total = signal(0);
    protected readonly page = signal(1);
    protected readonly pageSize = signal<number>(10);
    protected readonly status = signal<IdentityReviewStatus>('open');
    protected readonly kind = signal<KindFilter>('all');
    protected readonly loading = signal(false);
    protected readonly busy = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly notice = signal('');
    protected readonly pending = signal<{ reviewId: number; keepId: number } | null>(null);

    private requestId = 0;

    constructor() {
        void this.load();
    }

    protected reasonLabel(reason: string): string {
        return REASONS[reason] ?? reason.replaceAll('_', ' ');
    }

    protected percent(score: number): string {
        return `${Math.round(score * 100)}%`;
    }

    protected experiments(entity: IdentityEntity): string {
        return `${entity.experiment_count} experiment${entity.experiment_count === 1 ? '' : 's'}`;
    }

    protected other(review: IdentityReview, entity: IdentityEntity): IdentityEntity {
        return (review.entity?.id === entity.id ? review.candidate : review.entity) ?? entity;
    }

    protected setStatus(status: IdentityReviewStatus): void {
        this.status.set(status);
        this.page.set(1);
        void this.load();
    }

    protected setKind(kind: KindFilter): void {
        this.kind.set(kind);
        this.page.set(1);
        void this.load();
    }

    protected goTo(page: number): void {
        this.page.set(page);
        void this.load();
    }

    protected setPageSize(size: number): void {
        this.pageSize.set(size);
        this.page.set(1);
        void this.load();
    }

    protected async audit(): Promise<void> {
        await this.act(async () => {
            const result = await runIdentityAudit();
            if (result.data) {
                const found = result.data.adsorbate + result.data.adsorbent;
                this.notice.set(found ? `${found} new item${found === 1 ? '' : 's'} to review.` : 'No new duplicates found.');
            }
            return result.error;
        });
    }

    protected async merge(review: IdentityReview, keep: IdentityEntity): Promise<void> {
        await this.act(async () => {
            const result = await mergeIdentityReview(review.id, keep.id);
            if (!result.error) this.notice.set(`Merged into ${keep.name}.`);
            return result.error;
        });
    }

    protected async reject(review: IdentityReview): Promise<void> {
        await this.act(async () => (await rejectIdentityReview(review.id)).error);
    }

    private async act(action: () => Promise<string | null>): Promise<void> {
        this.busy.set(true);
        this.notice.set('');
        const error = await action();
        this.busy.set(false);
        this.error.set(error);
        this.pending.set(null);
        await this.load();
    }

    private async load(): Promise<void> {
        const requestId = ++this.requestId;
        this.loading.set(true);
        const kind = this.kind();
        const result = await fetchIdentityReviews({
            page: this.page(),
            page_size: this.pageSize(),
            status: this.status(),
            kind: kind === 'all' ? undefined : kind,
        });
        if (requestId !== this.requestId) return;
        this.loading.set(false);
        if (result.error) this.error.set(result.error);
        this.items.set(result.data?.items ?? []);
        this.total.set(result.data?.pagination.total ?? 0);
        const lastPage = Math.max(1, Math.ceil(this.total() / this.pageSize()));
        if (this.page() > lastPage) {
            this.page.set(lastPage);
            void this.load();
        }
    }
}
