import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelConfigFormComponent } from './model-config-form.component';

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
