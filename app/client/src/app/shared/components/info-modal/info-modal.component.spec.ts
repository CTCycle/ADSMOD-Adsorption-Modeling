import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { InfoModalComponent } from './info-modal.component';

describe('InfoModalComponent', () => {
    it('renders only defined data entries', async () => {
        await TestBed.configureTestingModule({
            imports: [InfoModalComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(InfoModalComponent);
        fixture.componentRef.setInput('isOpen', true);
        fixture.componentRef.setInput('title', 'Dataset Metadata');
        fixture.componentRef.setInput('data', {
            Name: 'Dataset A',
            Count: 12,
            Empty: null,
        });
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.textContent).toContain('Dataset Metadata');
        expect(root.textContent).toContain('Dataset A');
        expect(root.textContent).toContain('12');
        expect(root.textContent).not.toContain('Empty');
    });

    it('emits close when the done button is clicked', async () => {
        await TestBed.configureTestingModule({
            imports: [InfoModalComponent],
        }).compileComponents();

        const fixture = TestBed.createComponent(InfoModalComponent);
        const emitSpy = vi.spyOn(fixture.componentInstance.closed, 'emit');
        fixture.componentRef.setInput('isOpen', true);
        fixture.componentRef.setInput('title', 'Checkpoint Details');
        fixture.componentRef.setInput('data', { Name: 'cp-1' });
        fixture.detectChanges();

        (fixture.nativeElement as HTMLElement).querySelector<HTMLButtonElement>('.info-modal-footer button')?.click();
        expect(emitSpy).toHaveBeenCalled();
    });
});
