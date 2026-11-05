// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, input, output } from '@angular/core';
import type { ModelBounds, ModelCatalogEntry } from '../../../models/fitting.model';
import { EquationRendererComponent } from '../../../shared/components/equation-renderer/equation-renderer.component';
import { SwitchComponent } from '../../../shared/components/switch/switch.component';
import { ModelConfigFormComponent } from './model-config-form.component';

/** Catalog card: availability switch, scientific summary, and the global default bounds editor. */
@Component({
    selector: 'adsmod-model-card',
    standalone: true,
    imports: [EquationRendererComponent, ModelConfigFormComponent, SwitchComponent],
    template: `
        <article class="catalog-card" [class.disabled]="!model().enabled" [class.expanded]="expanded()" [attr.data-model]="model().key">
            <header class="catalog-card-head">
                <adsmod-switch
                    [checked]="model().enabled"
                    [disabled]="busy()"
                    [ariaLabel]="(model().enabled ? 'Disable ' : 'Enable ') + model().name"
                    (checkedChange)="enabledChange.emit($event)"
                />
                <h3 [id]="titleId()">{{ model().name }}</h3>
                @if (customized()) {
                    <span class="catalog-badge" title="Saved defaults differ from the canonical bounds">Custom</span>
                }
                <button
                    class="button quiet icon-button catalog-expand"
                    type="button"
                    [attr.aria-expanded]="expanded()"
                    [attr.aria-controls]="contentId()"
                    [attr.aria-label]="(expanded() ? 'Hide ' : 'Edit ') + model().name + ' default bounds'"
                    [title]="expanded() ? 'Hide default bounds' : 'Edit default bounds'"
                    (click)="toggle.emit()"
                >
                    <svg class="catalog-chevron" [class.open]="expanded()" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="6 9 12 15 18 9" /></svg>
                </button>
            </header>
            <p class="catalog-description">{{ model().assumptions }}</p>
            <adsmod-equation-renderer class="catalog-equation" [latex]="model().equation_latex" />
            @if (!model().enabled) {
                <p class="catalog-note">Disabled · not offered for new fitting sessions.</p>
            }
            @if (!expanded() && error()) {
                <p class="catalog-error" role="alert">{{ error() }}</p>
            }
            @if (expanded()) {
                <section class="catalog-editor" [id]="contentId()" [attr.aria-labelledby]="titleId()">
                    <adsmod-model-config-form
                        [parameters]="model().parameters"
                        [value]="draft()"
                        [reference]="saved()"
                        [disabled]="busy()"
                        [label]="model().name + ' default bounds'"
                        (valueChange)="draftChange.emit($event)"
                    />
                    <p class="catalog-hint">Search bounds in Pa and mol/kg, used to initialize new fitting sessions. Existing sessions keep their own snapshot.</p>
                    @if (error()) {
                        <p class="catalog-error" role="alert">{{ error() }}</p>
                    } @else if (dirty()) {
                        <p class="catalog-dirty" role="status">Unsaved changes</p>
                    } @else if (message()) {
                        <p class="catalog-saved" role="status">{{ message() }}</p>
                    }
                    <div class="catalog-actions">
                        <button class="button secondary" type="button" [disabled]="busy() || !customized()" (click)="restore.emit()">Restore defaults</button>
                        @if (dirty()) {
                            <button class="button quiet" type="button" [disabled]="busy()" (click)="discard.emit()">Discard</button>
                        }
                        <button class="button primary" type="button" [disabled]="busy() || !dirty()" (click)="save.emit()">Save defaults</button>
                    </div>
                </section>
            }
        </article>
    `,
    styles: [`
        :host { display: block; min-width: 0; }
        .catalog-card {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            height: 100%;
            padding: 0.75rem 0.9rem 0.85rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            background: var(--surface);
            transition: border-color 150ms ease;
        }
        .catalog-card.expanded { border-color: var(--primary-600); }
        .catalog-card.disabled { background: var(--surface-subtle); }
        .catalog-card.disabled .catalog-description,
        .catalog-card.disabled .catalog-equation { opacity: 0.5; }
        .catalog-card-head { display: flex; align-items: center; gap: 0.55rem; min-width: 0; }
        .catalog-card-head h3 { flex: 1; min-width: 0; margin: 0; font-size: 0.95rem; font-weight: 700; line-height: 1.25; overflow-wrap: anywhere; }
        .catalog-badge { flex: none; padding: 1px 7px; border-radius: 999px; background: color-mix(in srgb, var(--primary-600) 12%, transparent); color: var(--primary-600); font-size: 0.65rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }
        .catalog-expand { flex: none; }
        .catalog-chevron { transition: transform 150ms ease; }
        .catalog-chevron.open { transform: rotate(180deg); }
        .catalog-description { margin: 0; color: var(--text-secondary); font-size: 0.8rem; line-height: 1.4; }
        .catalog-equation { display: flex; align-items: center; justify-content: center; min-height: 52px; margin-top: auto; overflow-x: auto; color: var(--text-primary); scrollbar-width: thin; }
        .catalog-note { margin: 0; color: var(--text-tertiary); font-size: 0.75rem; }
        .catalog-editor { display: grid; gap: 0.5rem; padding-top: 0.6rem; border-top: 1px solid var(--border-color); }
        .catalog-hint { margin: 0; color: var(--text-tertiary); font-size: 0.72rem; line-height: 1.35; }
        .catalog-error, .catalog-dirty, .catalog-saved { margin: 0; font-size: 0.75rem; font-weight: 600; }
        .catalog-error { color: var(--danger-text); }
        .catalog-dirty { color: var(--warning-text); }
        .catalog-saved { color: var(--success-text); }
        .catalog-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0.4rem; }
        @media (prefers-reduced-motion: reduce) { .catalog-chevron { transition: none; } }
    `],
})
export class ModelCardComponent {
    readonly model = input.required<ModelCatalogEntry>();
    readonly expanded = input(false);
    readonly draft = input.required<ModelBounds>();
    readonly saved = input.required<ModelBounds>();
    readonly busy = input(false);
    readonly error = input<string | null>(null);
    readonly message = input<string | null>(null);
    readonly toggle = output<void>();
    readonly enabledChange = output<boolean>();
    readonly draftChange = output<ModelBounds>();
    readonly save = output<void>();
    readonly discard = output<void>();
    readonly restore = output<void>();

    protected readonly titleId = computed(() => `model-title-${this.model().key}`);
    protected readonly contentId = computed(() => `model-editor-${this.model().key}`);
    protected readonly customized = computed(() => this.model().parameters.some((parameter) => parameter.customized));
    protected readonly dirty = computed(() => JSON.stringify(this.draft()) !== JSON.stringify(this.saved()));
}
