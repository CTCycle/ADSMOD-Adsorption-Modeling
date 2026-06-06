import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ADSORPTION_MODELS } from '../../core/constants/adsorption-models';
import { ModelCardComponent } from './model-card.component';

describe('ModelCardComponent', () => {
    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [ModelCardComponent],
        }).compileComponents();
    });

    it('toggles from keyboard Enter and Space on the header button role', () => {
        const fixture = TestBed.createComponent(ModelCardComponent);
        const toggleSpy = vi.spyOn(fixture.componentInstance.toggle, 'emit');
        fixture.componentRef.setInput('model', ADSORPTION_MODELS[0]);
        fixture.componentRef.setInput('currentConfig', { k: { min: 0.1, max: 0.5 }, qsat: { min: 1, max: 5 } });
        fixture.componentRef.setInput('isEnabled', true);
        fixture.detectChanges();

        const header = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.model-card-header');
        expect(header?.getAttribute('role')).toBe('button');
        expect(header?.getAttribute('aria-controls')).toBe('model-content-langmuir');

        header?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
        header?.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));

        expect(toggleSpy).toHaveBeenNthCalledWith(1, ADSORPTION_MODELS[0].id);
        expect(toggleSpy).toHaveBeenNthCalledWith(2, ADSORPTION_MODELS[0].id);
    });

    it('does not toggle when the card is disabled', () => {
        const fixture = TestBed.createComponent(ModelCardComponent);
        const toggleSpy = vi.spyOn(fixture.componentInstance.toggle, 'emit');
        fixture.componentRef.setInput('model', ADSORPTION_MODELS[0]);
        fixture.componentRef.setInput('currentConfig', { k: { min: 0.1, max: 0.5 }, qsat: { min: 1, max: 5 } });
        fixture.componentRef.setInput('isEnabled', false);
        fixture.detectChanges();

        const header = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.model-card-header');
        header?.dispatchEvent(new Event('click'));
        header?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

        expect(toggleSpy).not.toHaveBeenCalled();
    });
});
