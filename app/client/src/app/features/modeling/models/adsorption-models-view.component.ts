// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ModelingStore, catalogBounds } from '../../../core/state/modeling.store';
import type { ModelBounds, ModelCatalogEntry } from '../../../models/fitting.model';
import { ConfirmDialogComponent } from '../../../shared/components/confirm-dialog/confirm-dialog.component';
import { ModelCardComponent } from './model-card.component';

interface CardFeedback {
    error: string | null;
    message: string | null;
}

/** Global adsorption model catalog: availability and persisted default search bounds. */
@Component({
    selector: 'adsmod-adsorption-models-view',
    standalone: true,
    imports: [ConfirmDialogComponent, ModelCardComponent],
    template: `
        <section class="catalog-view" aria-labelledby="catalog-title">
            <header class="catalog-toolbar">
                <div>
                    <h2 id="catalog-title">Adsorption models <span class="catalog-count">{{ enabledCount() }}/{{ store.catalog().length }} enabled</span></h2>
                    <p>Enable the models offered to new fitting sessions and set their default parameter bounds.</p>
                </div>
                <input type="search" class="catalog-search" aria-label="Search models" placeholder="Search models" [value]="search()" (input)="search.set($any($event.target).value)" />
            </header>

            @if (store.catalogError()) {
                <div class="inline-error" role="alert">
                    <span>Model catalog unavailable: {{ store.catalogError() }}</span>
                    <button class="button secondary" type="button" (click)="store.loadCatalog()">Retry</button>
                </div>
            } @else if (!store.catalogLoaded()) {
                <p class="catalog-empty" role="status">Loading models…</p>
            } @else if (!visible().length) {
                <p class="catalog-empty" role="status">No models match "{{ search() }}".</p>
            }

            <div class="catalog-grid">
                @for (model of visible(); track model.key) {
                    <adsmod-model-card
                        [model]="model"
                        [expanded]="expanded() === model.key"
                        [draft]="draftFor(model)"
                        [saved]="saved(model)"
                        [busy]="busy() === model.key"
                        [error]="feedback()[model.key]?.error ?? null"
                        [message]="feedback()[model.key]?.message ?? null"
                        (toggle)="expanded.set(expanded() === model.key ? null : model.key)"
                        (enabledChange)="setEnabled(model, $event)"
                        (draftChange)="setDraft(model.key, $event)"
                        (discard)="discard(model.key)"
                        (save)="save(model)"
                        (restore)="restoreTarget.set(model)"
                    />
                }
            </div>
        </section>

        <adsmod-confirm-dialog
            [open]="restoreTarget() !== null"
            title="Restore default bounds"
            [message]="'Replace the saved bounds for ' + (restoreTarget()?.name ?? '') + ' with the canonical defaults? Existing fitting sessions are not affected.'"
            confirmLabel="Restore defaults"
            (confirmed)="restore()"
            (closed)="restoreTarget.set(null)"
        />
    `,
    styles: [`
        :host { display: block; min-width: 0; }
        .catalog-view { display: grid; gap: var(--spacing-md); min-width: 0; }
        .catalog-toolbar { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: var(--spacing-sm) var(--spacing-md); }
        .catalog-toolbar h2 { margin: 0; font-size: 1.125rem; font-weight: 600; }
        .catalog-toolbar p { margin: 0.2rem 0 0; color: var(--text-secondary); font-size: 0.85rem; }
        .catalog-count { margin-left: 0.25rem; padding: 0 7px; border-radius: 999px; background: var(--slate-100); color: var(--text-secondary); font-size: 0.75rem; font-weight: 600; vertical-align: middle; }
        .catalog-search { width: min(100%, 16rem); }
        .catalog-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 17rem), 1fr)); gap: var(--spacing-sm); align-items: start; }
        .catalog-empty { margin: 0; color: var(--text-secondary); font-size: 0.85rem; }
        .inline-error { display: flex; flex-wrap: wrap; align-items: center; gap: var(--spacing-sm); padding: var(--spacing-sm); border: 1px solid var(--danger-border); border-radius: var(--radius-md); background: var(--danger-bg); color: var(--danger-text); font-size: 0.85rem; }
    `],
})
export class AdsorptionModelsViewComponent implements OnInit {
    protected readonly store = inject(ModelingStore);
    protected readonly search = signal('');
    protected readonly expanded = signal<string | null>(null);
    protected readonly drafts = signal<Record<string, ModelBounds>>({});
    protected readonly busy = signal<string | null>(null);
    protected readonly feedback = signal<Record<string, CardFeedback>>({});
    protected readonly restoreTarget = signal<ModelCatalogEntry | null>(null);

    protected readonly enabledCount = computed(() => this.store.enabledModels().length);
    protected readonly visible = computed(() => {
        const term = this.search().trim().toLowerCase();
        return this.store.catalog().filter((model) => !term || model.name.toLowerCase().includes(term) || model.key.includes(term));
    });

    ngOnInit(): void {
        void this.store.loadCatalog();
    }

    protected draftFor(model: ModelCatalogEntry): ModelBounds {
        const drafts: Partial<Record<string, ModelBounds>> = this.drafts();
        return drafts[model.key] ?? catalogBounds(model);
    }

    protected saved(model: ModelCatalogEntry): ModelBounds {
        return catalogBounds(model);
    }

    protected setDraft(key: string, bounds: ModelBounds): void {
        this.drafts.update((drafts) => ({ ...drafts, [key]: bounds }));
        this.setFeedback(key, null, null);
    }

    protected discard(key: string): void {
        this.drafts.update(({ [key]: _discarded, ...rest }) => rest);
        this.setFeedback(key, null, null);
    }

    protected async setEnabled(model: ModelCatalogEntry, enabled: boolean): Promise<void> {
        await this.run(model.key, () => this.store.setModelEnabled(model.key, enabled), enabled ? 'Enabled for new sessions.' : 'Disabled for new sessions.', false);
    }

    protected async save(model: ModelCatalogEntry): Promise<void> {
        const draft = this.drafts()[model.key];
        if (!draft) return;
        await this.run(model.key, () => this.store.saveModelDefaults(model.key, draft), 'Defaults saved.', true);
    }

    protected async restore(): Promise<void> {
        const model = this.restoreTarget();
        this.restoreTarget.set(null);
        if (model) await this.run(model.key, () => this.store.restoreModelDefaults(model.key), 'Canonical defaults restored.', true);
    }

    private async run(key: string, action: () => Promise<string | null>, success: string, clearDraft: boolean): Promise<void> {
        this.busy.set(key);
        const error = await action();
        this.busy.set(null);
        if (!error && clearDraft) this.discard(key);
        this.setFeedback(key, error, error ? null : success);
    }

    private setFeedback(key: string, error: string | null, message: string | null): void {
        this.feedback.update((feedback) => ({ ...feedback, [key]: { error, message } }));
    }
}
