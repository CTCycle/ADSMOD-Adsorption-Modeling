// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, inject } from '@angular/core';
import { ThemePreference, ThemeService } from '../../../core/theme/theme.service';

interface ThemeOption {
    value: ThemePreference;
    label: string;
}

@Component({
    selector: 'adsmod-theme-switch',
    standalone: true,
    template: `
        <div class="theme-switch" role="radiogroup" aria-label="Colour theme" (keydown)="handleKeydown($event)">
            @for (option of options; track option.value) {
                <button
                    class="theme-switch-option"
                    type="button"
                    role="radio"
                    [attr.aria-checked]="theme.preference() === option.value"
                    [attr.aria-label]="option.label"
                    [attr.tabindex]="theme.preference() === option.value ? 0 : -1"
                    [title]="option.label"
                    (click)="theme.setPreference(option.value)"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        @switch (option.value) {
                            @case ('light') {
                                <circle cx="12" cy="12" r="4" />
                                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
                            }
                            @case ('system') {
                                <rect x="3" y="4" width="18" height="12" rx="2" />
                                <path d="M8 20h8M12 16v4" />
                            }
                            @case ('dark') {
                                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
                            }
                        }
                    </svg>
                </button>
            }
        </div>
    `,
})
export class ThemeSwitchComponent {
    protected readonly theme = inject(ThemeService);
    protected readonly options: readonly ThemeOption[] = [
        { value: 'light', label: 'Light theme' },
        { value: 'system', label: 'Match system theme' },
        { value: 'dark', label: 'Dark theme' },
    ];

    /** Radio-group keyboard model: arrows move the selection, which follows focus. */
    protected handleKeydown(event: KeyboardEvent): void {
        const step = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1 : event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 0;
        if (step === 0) {
            return;
        }
        event.preventDefault();
        const current = this.options.findIndex((option) => option.value === this.theme.preference());
        const next = this.options[(current + step + this.options.length) % this.options.length];
        this.theme.setPreference(next.value);
        const buttons = (event.currentTarget as HTMLElement).querySelectorAll<HTMLButtonElement>('button');
        queueMicrotask(() => buttons[this.options.indexOf(next)]?.focus());
    }
}
