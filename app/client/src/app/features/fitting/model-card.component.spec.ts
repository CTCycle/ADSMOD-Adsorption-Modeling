// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelCardComponent } from './model-card.component';
import { ModelConfigFormComponent } from './model-config-form.component';

// Model card behavior

const MODEL = { id: 'langmuir', name: 'Langmuir', shortDescription: 'test', equationLatex: 'q=Kp', parameterDefaults: { k: [0, 1] as [number, number] } };

describe('ModelCardComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [ModelCardComponent],
        }).compileComponents();
    });

    it('toggles from keyboard Enter and Space on the header button role', () => {
        const fixture = TestBed.createComponent(ModelCardComponent);
        const toggleSpy = vi.spyOn(fixture.componentInstance.toggle, 'emit');
        fixture.componentRef.setInput('model', MODEL);
        fixture.componentRef.setInput('currentConfig', { k: { min: 0.1, max: 0.5 }, qsat: { min: 1, max: 5 } });
        fixture.componentRef.setInput('isEnabled', true);
        fixture.detectChanges();

        const header = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.model-card-header');
        expect(header?.getAttribute('role')).toBe('button');
        expect(header?.getAttribute('aria-controls')).toBe('model-content-langmuir');

        header?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
        header?.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));

        expect(toggleSpy).toHaveBeenNthCalledWith(1, MODEL.id);
        expect(toggleSpy).toHaveBeenNthCalledWith(2, MODEL.id);
    });

    it('does not toggle when the card is disabled', () => {
        const fixture = TestBed.createComponent(ModelCardComponent);
        const toggleSpy = vi.spyOn(fixture.componentInstance.toggle, 'emit');
        fixture.componentRef.setInput('model', MODEL);
        fixture.componentRef.setInput('currentConfig', { k: { min: 0.1, max: 0.5 }, qsat: { min: 1, max: 5 } });
        fixture.componentRef.setInput('isEnabled', false);
        fixture.detectChanges();

        const header = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.model-card-header');
        header?.dispatchEvent(new Event('click'));
        header?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

        expect(toggleSpy).not.toHaveBeenCalled();
    });
});

// Model configuration form behavior

describe('ModelConfigFormComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [ModelConfigFormComponent],
        }).compileComponents();
    });

    it('shows an empty state when no parameter defaults are available', () => {
        const fixture = TestBed.createComponent(ModelConfigFormComponent);
        fixture.componentRef.setInput('modelId', 'empty-model');
        fixture.componentRef.setInput('parameterDefaults', {});
        fixture.componentRef.setInput('value', {});
        fixture.detectChanges();

        expect((fixture.nativeElement as HTMLElement).textContent).toContain('Configuration is not available for this model.');
    });

    it('emits updated bounds when a nested number input changes', () => {
        const fixture = TestBed.createComponent(ModelConfigFormComponent);
        const emitSpy = vi.spyOn(fixture.componentInstance.valueChange, 'emit');
        fixture.componentRef.setInput('modelId', 'langmuir');
        fixture.componentRef.setInput('parameterDefaults', { k: [0.1, 1.5] });
        fixture.componentRef.setInput('value', { k: { min: 0.1, max: 1.5 } });
        fixture.detectChanges();

        const numberInput = fixture.debugElement.children[0].children[0].children[0].children[1].children[0].componentInstance as {
            valueChange: { emit: (value: number) => void };
        };
        numberInput.valueChange.emit(0.25);

        expect(emitSpy).toHaveBeenCalledWith({
            k: { min: 0.25, max: 1.5 },
        });
    });
});
