// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ModelBounds, ModelCatalogEntry } from '../../../models/fitting.model';
import { ModelCardComponent } from './model-card.component';
import { ModelConfigFormComponent } from './model-config-form.component';

// Model card behavior

function entry(customized = false): ModelCatalogEntry {
    return {
        key: 'langmuir',
        name: 'Langmuir',
        equation_latex: 'q = q_m K p',
        assumptions: 'Monolayer adsorption on a homogeneous surface.',
        pressure_requirement: 'absolute',
        requires_temperature: false,
        reference: 'Langmuir 1918',
        enabled: true,
        parameters: [
            { name: 'qmax', label: 'Saturation uptake', lower: 0.1, upper: 10, canonical_lower: 0.1, canonical_upper: 10, unit: 'mol/kg', customized },
        ],
    };
}

const SAVED: ModelBounds = { qmax: { lower: 0.1, upper: 10 } };

function setup(overrides: { customized?: boolean; draft?: ModelBounds; expanded?: boolean } = {}) {
    const fixture = TestBed.createComponent(ModelCardComponent);
    fixture.componentRef.setInput('model', entry(overrides.customized));
    fixture.componentRef.setInput('draft', overrides.draft ?? SAVED);
    fixture.componentRef.setInput('saved', SAVED);
    fixture.componentRef.setInput('expanded', overrides.expanded ?? false);
    fixture.detectChanges();
    return { fixture, root: fixture.nativeElement as HTMLElement };
}

function button(root: HTMLElement, label: string): HTMLButtonElement {
    const found = Array.from(root.querySelectorAll('button')).find((item) => item.textContent?.trim() === label);
    if (!found) throw new Error(`Button "${label}" not found`);
    return found;
}

describe('ModelCardComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({ imports: [ModelCardComponent] }).compileComponents();
    });

    it('renders the model name and equation and emits when the switch changes', () => {
        const { fixture, root } = setup();
        const enabledSpy = vi.spyOn(fixture.componentInstance.enabledChange, 'emit');

        expect(root.querySelector('h3')?.textContent).toContain('Langmuir');
        expect(root.querySelector('adsmod-equation-renderer')).not.toBeNull();

        root.querySelector<HTMLElement>('adsmod-switch input')?.click();
        expect(enabledSpy).toHaveBeenCalledWith(false);
    });

    it('emits toggle from the expand button and only shows the editor when expanded', () => {
        const { fixture, root } = setup();
        const toggleSpy = vi.spyOn(fixture.componentInstance.toggle, 'emit');

        expect(root.querySelector('adsmod-model-config-form')).toBeNull();
        root.querySelector<HTMLButtonElement>('.catalog-expand')?.click();
        expect(toggleSpy).toHaveBeenCalledOnce();

        fixture.componentRef.setInput('expanded', true);
        fixture.detectChanges();
        expect(root.querySelector('adsmod-model-config-form')).not.toBeNull();
    });

    it('enables Save only when the draft differs from the saved bounds', () => {
        const { fixture, root } = setup({ expanded: true });
        expect(button(root, 'Save defaults').disabled).toBe(true);

        fixture.componentRef.setInput('draft', { qmax: { lower: 0.2, upper: 10 } });
        fixture.detectChanges();
        expect(button(root, 'Save defaults').disabled).toBe(false);
        expect(button(root, 'Discard')).toBeTruthy();
    });

    it('enables Restore defaults only when a parameter is customized', () => {
        const { fixture, root } = setup({ expanded: true });
        expect(button(root, 'Restore defaults').disabled).toBe(true);

        fixture.componentRef.setInput('model', entry(true));
        fixture.detectChanges();
        expect(button(root, 'Restore defaults').disabled).toBe(false);
    });
});

// Model configuration form behavior

describe('ModelConfigFormComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({ imports: [ModelConfigFormComponent] }).compileComponents();
    });

    it('emits the full bounds map with the edited bound parsed as a number', () => {
        const fixture = TestBed.createComponent(ModelConfigFormComponent);
        const emitSpy = vi.spyOn(fixture.componentInstance.valueChange, 'emit');
        fixture.componentRef.setInput('parameters', entry().parameters);
        fixture.componentRef.setInput('value', SAVED);
        fixture.detectChanges();

        const input = (fixture.nativeElement as HTMLElement).querySelector<HTMLInputElement>('input[aria-label="qmax upper bound"]');
        if (!input) throw new Error('input missing');
        input.value = '25';
        input.dispatchEvent(new Event('change'));

        expect(emitSpy).toHaveBeenCalledWith({ qmax: { lower: 0.1, upper: 25 } });
    });
});
