// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed, type ComponentFixture } from '@angular/core/testing';
import type { UnitOption } from '../../../models/dataset.model';
import { findUnit, searchUnits, unitKey } from '../../utils/unit-key';
import {
    UnitComboboxComponent,
    type CustomUnitChoice,
} from './unit-combobox.component';

function option(symbol: string, overrides: Partial<UnitOption> = {}): UnitOption {
    return {
        quantity: 'uptake',
        symbol,
        label: symbol,
        name: symbol,
        group: 'Molar uptake',
        note: '',
        aliases: [],
        built_in: true,
        has_conversion: true,
        factor_to_canonical: null,
        id: null,
        ...overrides,
    };
}

const OPTIONS = [
    option('mmol/g', { aliases: ['mmol g-1'] }),
    option('mmol/kg'),
    option('cm³(STP)/g', { symbol: 'cm3(STP)/g', name: 'cubic centimetre (STP) per gram' }),
];

describe('unit key matching', () => {
    it('ignores case and spacing but keeps distinct units apart', () => {
        expect(unitKey(' MMOL / G ')).toBe(unitKey('mmol/g'));
        expect(unitKey('mmol/g')).not.toBe(unitKey('mmol/kg'));
        expect(findUnit(OPTIONS, 'MMOL G-1')?.symbol).toBe('mmol/g');
        expect(findUnit(OPTIONS, 'cm3 (stp)/g')?.symbol).toBe('cm3(STP)/g');
        expect(findUnit(OPTIONS, 'mmol/cc')).toBeNull();
    });

    it('searches by symbol, alias and name', () => {
        expect(searchUnits(OPTIONS, 'kg').map((item) => item.symbol)).toEqual(['mmol/kg']);
        expect(searchUnits(OPTIONS, 'cubic').map((item) => item.symbol)).toEqual(['cm3(STP)/g']);
    });
});

describe('UnitComboboxComponent', () => {
    let fixture: ComponentFixture<UnitComboboxComponent>;
    let choices: CustomUnitChoice[];

    function type(text: string): HTMLInputElement {
        const input = fixture.nativeElement.querySelector('input') as HTMLInputElement;
        input.value = text;
        input.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        return input;
    }

    beforeEach(async () => {
        await TestBed.configureTestingModule({ imports: [UnitComboboxComponent] }).compileComponents();
        fixture = TestBed.createComponent(UnitComboboxComponent);
        fixture.componentRef.setInput('quantity', 'uptake');
        fixture.componentRef.setInput('label', 'Uptake unit');
        fixture.componentRef.setInput('options', OPTIONS);
        choices = [];
        fixture.componentInstance.customUnit.subscribe((choice) => choices.push(choice));
        fixture.detectChanges();
    });

    it('selects a catalog unit without asking to save it', () => {
        const input = type('MMOL/KG');
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        fixture.detectChanges();

        expect(fixture.componentInstance.value()).toBe('mmol/kg');
        expect(fixture.nativeElement.querySelector('.unit-confirm')).toBeNull();
    });

    it('asks before accepting an unknown unit and keeps a session-only unit out of the catalog', async () => {
        const input = type('mmol/cc');
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        // No match, so Enter commits the typed text instead of a highlighted option.
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('.unit-confirm')?.textContent).toContain(
            'is not in your catalog',
        );

        const sessionOnly = Array.from<HTMLButtonElement>(
            fixture.nativeElement.querySelectorAll('.unit-confirm button'),
        ).find((button) => button.textContent?.includes('this import only'));
        sessionOnly?.click();
        await fixture.whenStable();
        fixture.detectChanges();

        expect(choices).toEqual([
            {
                definition: { quantity: 'uptake', symbol: 'mmol/cc', factor_to_canonical: null },
                saved: null,
            },
        ]);
        expect(fixture.componentInstance.value()).toBe('mmol/cc');
    });

    it('rejects a non-positive conversion factor instead of defining the unit', async () => {
        const input = type('mmol/cc');
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        fixture.detectChanges();
        const factor = fixture.nativeElement.querySelector('.unit-confirm input') as HTMLInputElement;
        factor.value = '-2';
        factor.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        (fixture.nativeElement.querySelector('.unit-confirm button') as HTMLButtonElement).click();
        await fixture.whenStable();
        fixture.detectChanges();

        expect(fixture.nativeElement.querySelector('.unit-confirm')?.textContent).toContain(
            'must be a positive number',
        );
        expect(choices).toEqual([]);
    });
});
