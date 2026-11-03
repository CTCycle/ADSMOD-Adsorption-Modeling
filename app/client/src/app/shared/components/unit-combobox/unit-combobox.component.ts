// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import {
    Component,
    computed,
    input,
    model,
    output,
    signal,
} from '@angular/core';
import type {
    CustomUnitDefinition,
    UnitOption,
    UnitQuantity,
} from '../../../models/dataset.model';
import { saveCustomUnit } from '../../../services/dataset.service';
import { findUnit, searchUnits } from '../../utils/unit-key';

export interface CustomUnitChoice {
    definition: CustomUnitDefinition;
    saved: UnitOption | null;
}

let nextId = 0;

@Component({
    selector: 'adsmod-unit-combobox',
    standalone: true,
    template: `
        <div class="unit-combobox">
            <input
                class="select-input unit-input"
                role="combobox"
                autocomplete="off"
                spellcheck="false"
                aria-autocomplete="list"
                [id]="inputId"
                [attr.aria-label]="label()"
                [attr.aria-expanded]="open()"
                [attr.aria-controls]="listId"
                [attr.aria-activedescendant]="activeId()"
                [attr.aria-invalid]="invalid() ? 'true' : null"
                [placeholder]="
                    detected() ? 'Detected: ' + detected() : placeholder()
                "
                [value]="text()"
                (focus)="openList()"
                (input)="onInput($event)"
                (keydown)="onKeydown($event)"
                (blur)="onBlur()"
            />
            @if (open()) {
                <ul class="unit-list" role="listbox" [id]="listId">
                    @for (group of groups(); track group.name) {
                        <li class="unit-group" role="presentation">
                            {{ group.name }}
                        </li>
                        @for (option of group.options; track option.symbol) {
                            <li
                                role="option"
                                class="unit-option"
                                [id]="optionId(option)"
                                [class.active]="option === highlighted()"
                                [attr.aria-selected]="option === highlighted()"
                                (mousedown)="$event.preventDefault()"
                                (click)="choose(option)"
                            >
                                <strong>{{ option.label }}</strong>
                                <span>{{ option.name }}</span>
                                @if (!option.has_conversion) {
                                    <em>no conversion</em>
                                }
                            </li>
                        }
                    } @empty {
                        <li class="unit-empty" role="presentation">
                            No catalog match. Press Enter to use
                            "{{ text() }}" as a custom unit.
                        </li>
                    }
                </ul>
            }
            @if (detected() && !text()) {
                <span class="unit-badge detected">Auto-detected</span>
            } @else if (text()) {
                <span class="unit-badge manual">Manual override</span>
                <button type="button" class="unit-reset" (click)="reset()">
                    Reset{{ detected() ? ' to ' + detected() : '' }}
                </button>
            }
            @if (selected(); as unit) {
                @if (unit.note) {
                    <small class="unit-note">{{ unit.note }}</small>
                }
                @if (!unit.has_conversion) {
                    <small class="unit-warning" role="alert">
                        "{{ unit.symbol }}" has no conversion definition, so
                        values cannot be converted and the import will be
                        blocked. Choose a catalog unit or define a factor.
                    </small>
                }
            }
            @if (pending(); as candidate) {
                <div
                    class="unit-confirm"
                    role="group"
                    [attr.aria-label]="'Unknown unit ' + candidate"
                >
                    <p>
                        <strong>"{{ candidate }}"</strong> is not in your
                        catalog. Would you like to save it for future use?
                    </p>
                    @if (quantity() !== 'temperature') {
                        <label>
                            <span>
                                Conversion factor to
                                {{ canonicalUnit() }} (optional)
                            </span>
                            <input
                                class="select-input"
                                inputmode="decimal"
                                placeholder="Leave empty for label only"
                                [value]="factorText()"
                                (input)="factorText.set(inputValue($event))"
                            />
                        </label>
                    }
                    <small>
                        Without a factor the unit is kept as a label and its
                        values are never converted; imports using it are
                        blocked until a factor is defined.
                    </small>
                    @if (confirmError()) {
                        <small class="unit-warning" role="alert">{{
                            confirmError()
                        }}</small>
                    }
                    <div class="unit-confirm-actions">
                        <button
                            type="button"
                            class="button primary"
                            [disabled]="saving()"
                            (click)="confirm(true)"
                        >
                            Save for future use
                        </button>
                        <button
                            type="button"
                            class="button secondary"
                            [disabled]="saving()"
                            (click)="confirm(false)"
                        >
                            Use for this import only
                        </button>
                        <button
                            type="button"
                            class="button quiet"
                            (click)="cancelPending()"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            }
        </div>
    `,
    styles: [
        `
            :host {
                display: block;
                min-width: 0;
            }
            .unit-combobox {
                position: relative;
                display: grid;
                gap: 0.3rem;
            }
            .unit-input {
                background-image: none;
                padding-right: 0.75rem;
                cursor: text;
            }
            .unit-list {
                position: absolute;
                top: 2.5rem;
                z-index: 20;
                width: 100%;
                min-width: 16rem;
                max-height: 16rem;
                margin: 0;
                padding: 0.25rem;
                overflow: auto;
                border: 1px solid var(--border-strong);
                border-radius: 10px;
                background: var(--surface);
                box-shadow: 0 12px 30px rgb(3 24 47 / 0.22);
                list-style: none;
            }
            .unit-group {
                padding: 0.4rem 0.5rem 0.2rem;
                color: var(--slate-500);
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .unit-option {
                display: flex;
                align-items: baseline;
                gap: 0.5rem;
                padding: 0.4rem 0.5rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.82rem;
            }
            .unit-option span {
                overflow: hidden;
                color: var(--slate-500);
                font-size: 0.74rem;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .unit-option em {
                margin-left: auto;
                color: #b45309;
                font-size: 0.7rem;
            }
            .unit-option.active,
            .unit-option:hover {
                background: var(--primary-50);
            }
            .unit-empty {
                padding: 0.6rem;
                color: var(--slate-500);
                font-size: 0.78rem;
            }
            .unit-badge {
                justify-self: start;
                padding: 0.1rem 0.45rem;
                border-radius: 999px;
                font-size: 0.68rem;
                font-weight: 700;
            }
            .unit-badge.detected {
                color: #0b6b3a;
                background: color-mix(in srgb, #16a34a 16%, transparent);
            }
            .unit-badge.manual {
                color: #92400e;
                background: color-mix(in srgb, #f59e0b 20%, transparent);
            }
            .unit-reset {
                justify-self: start;
                padding: 0;
                border: 0;
                color: var(--link);
                background: none;
                cursor: pointer;
                font-size: 0.74rem;
                text-decoration: underline;
            }
            .unit-note {
                color: var(--slate-500);
                font-weight: 400;
            }
            .unit-warning {
                color: #b91c1c;
                font-weight: 500;
            }
            .unit-confirm {
                display: grid;
                gap: 0.5rem;
                padding: 0.75rem;
                border: 1px solid #ca8a04;
                border-radius: 10px;
                background: var(--warning-bg);
            }
            .unit-confirm p {
                margin: 0;
                font-size: 0.82rem;
            }
            .unit-confirm label {
                display: grid;
                gap: 0.25rem;
            }
            .unit-confirm .select-input {
                background-image: none;
                padding-right: 0.75rem;
                cursor: text;
            }
            .unit-confirm-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
        `,
    ],
})
export class UnitComboboxComponent {
    readonly quantity = input.required<UnitQuantity>();
    readonly options = input.required<readonly UnitOption[]>();
    readonly label = input.required<string>();
    readonly detected = input<string | null>(null);
    readonly placeholder = input('Search or type a unit');
    readonly invalid = input(false);
    /** The selected unit text; empty means "use the detected unit". */
    readonly value = model('');
    readonly customUnit = output<CustomUnitChoice>();

    protected readonly inputId = `unit-combobox-${nextId++}`;
    protected readonly listId = `${this.inputId}-list`;
    protected readonly query = signal<string | null>(null);
    protected readonly open = signal(false);
    protected readonly highlighted = signal<UnitOption | null>(null);
    protected readonly pending = signal<string | null>(null);
    protected readonly factorText = signal('');
    protected readonly confirmError = signal('');
    protected readonly saving = signal(false);

    protected readonly text = computed(() => this.query() ?? this.value());
    protected readonly selected = computed(() =>
        findUnit(this.options(), this.value()),
    );
    protected readonly canonicalUnit = computed(() =>
        this.quantity() === 'pressure' ? 'Pa' : 'mol/kg',
    );
    protected readonly filtered = computed(() =>
        searchUnits(this.options(), this.query() ?? ''),
    );
    protected readonly groups = computed(() => {
        const groups = new Map<string, UnitOption[]>();
        for (const option of this.filtered()) {
            groups.set(option.group, [
                ...(groups.get(option.group) ?? []),
                option,
            ]);
        }
        return [...groups].map(([name, options]) => ({ name, options }));
    });
    protected readonly activeId = computed(() => {
        const option = this.highlighted();
        return this.open() && option ? this.optionId(option) : null;
    });

    protected optionId(option: UnitOption): string {
        return `${this.inputId}-${option.id ?? option.symbol}`.replace(
            /[^A-Za-z0-9_-]/g,
            '_',
        );
    }

    protected inputValue(event: Event): string {
        return (event.target as HTMLInputElement).value;
    }

    protected openList(): void {
        this.open.set(true);
    }

    protected onInput(event: Event): void {
        this.query.set(this.inputValue(event));
        this.open.set(true);
        this.highlighted.set(this.filtered()[0] ?? null);
    }

    protected onKeydown(event: KeyboardEvent): void {
        const items = this.filtered();
        const index = items.findIndex((item) => item === this.highlighted());
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            this.open.set(true);
            if (items.length === 0) {
                return;
            }
            const step = event.key === 'ArrowDown' ? 1 : -1;
            this.highlighted.set(
                items[(index + step + items.length) % items.length],
            );
        } else if (event.key === 'Enter') {
            event.preventDefault();
            const highlighted = this.highlighted();
            if (this.query() !== null && highlighted && this.open()) {
                this.choose(highlighted);
            } else {
                this.commitText();
            }
        } else if (event.key === 'Escape' && this.open()) {
            event.stopPropagation();
            this.close();
        }
    }

    protected onBlur(): void {
        window.setTimeout(() => {
            if (this.pending() === null && this.query() !== null) {
                this.commitText();
            }
        });
    }

    protected choose(option: UnitOption): void {
        this.pending.set(null);
        this.value.set(option.label);
        this.close();
    }

    protected reset(): void {
        this.pending.set(null);
        this.value.set('');
        this.close();
    }

    private close(): void {
        this.query.set(null);
        this.open.set(false);
        this.highlighted.set(null);
    }

    private commitText(): void {
        const typed = (this.query() ?? this.value()).trim();
        this.close();
        if (!typed) {
            this.value.set('');
            return;
        }
        const known = findUnit(this.options(), typed);
        if (known) {
            this.choose(known);
            return;
        }
        this.factorText.set('');
        this.confirmError.set('');
        this.pending.set(typed);
    }

    protected cancelPending(): void {
        this.pending.set(null);
    }

    protected async confirm(persist: boolean): Promise<void> {
        const symbol = this.pending();
        if (!symbol) {
            return;
        }
        const factorText = this.factorText().trim();
        const factor = factorText ? Number(factorText) : null;
        if (factor !== null && (!Number.isFinite(factor) || factor <= 0)) {
            this.confirmError.set(
                'The conversion factor must be a positive number.',
            );
            return;
        }
        const definition: CustomUnitDefinition = {
            quantity: this.quantity(),
            symbol,
            factor_to_canonical:
                this.quantity() === 'temperature' ? null : factor,
        };
        let saved: UnitOption | null = null;
        if (persist) {
            this.saving.set(true);
            const result = await saveCustomUnit(definition);
            this.saving.set(false);
            if (result.error || !result.data) {
                this.confirmError.set(
                    result.error || 'The unit could not be saved.',
                );
                return;
            }
            saved = result.data.unit;
        }
        this.pending.set(null);
        this.value.set(saved?.label ?? symbol);
        this.customUnit.emit({ definition, saved });
    }
}
