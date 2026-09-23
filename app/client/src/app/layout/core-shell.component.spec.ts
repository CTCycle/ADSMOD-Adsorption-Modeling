import { TestBed } from '@angular/core/testing';
import type { ComponentFixture } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreShellComponent } from './core-shell.component';

describe('CoreShellComponent', () => {
    const fetchMock = vi.fn();
    const fixtures: ComponentFixture<CoreShellComponent>[] = [];

    function setBackendResponses(machineLearning: boolean | null, readiness: boolean): void {
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/system/capabilities')) {
                if (machineLearning === null) {
                    throw new Error('backend unavailable');
                }
                return {
                    ok: true,
                    json: async () => ({
                        version: '3.0.0',
                        features: {
                            datasets: true,
                            nist: true,
                            fitting: true,
                            machine_learning: machineLearning,
                            training: machineLearning,
                            checkpoints: machineLearning,
                        },
                    }),
                };
            }
            if (url.endsWith('/health/ready')) {
                if (!readiness) {
                    throw new Error('backend unavailable');
                }
                return {
                    ok: true,
                    json: async () => ({ service: 'backend', version: '3.0.0', state: 'ready' }),
                };
            }
            return { ok: false, status: 404, json: async () => ({}) };
        });
    }

    beforeEach(async () => {
        TestBed.resetTestingModule();
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
        await TestBed.configureTestingModule({
            imports: [CoreShellComponent],
            providers: [provideRouter([])],
        }).compileComponents();
    });

    afterEach(() => {
        for (const fixture of fixtures.splice(0)) {
            fixture.destroy();
        }
        vi.unstubAllGlobals();
    });

    it('recovers backend status from Offline to Online and gates Training by the active profile', async () => {
        setBackendResponses(false, true);
        const fixture = TestBed.createComponent(CoreShellComponent);
        fixtures.push(fixture);
        fixture.detectChanges();
        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const statusBar = root.querySelector<HTMLElement>('.console-status-bar');
        expect(statusBar?.textContent).toContain('BackendOnline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(false);
        expect(root.querySelector('a[routerLink="/training"]')).toBeNull();

        setBackendResponses(null, false);

        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        expect(statusBar?.textContent).toContain('BackendOffline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(true);
        expect(root.querySelector('a[routerLink="/training"]')).toBeNull();

        setBackendResponses(true, true);

        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        expect(statusBar?.textContent).toContain('BackendOnline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(false);
        expect(root.querySelector('a[routerLink="/training"]')).not.toBeNull();
    });

    it('keeps Docs and Settings visible but disabled with accessible unavailable labels', () => {
        const fixture = TestBed.createComponent(CoreShellComponent);
        fixtures.push(fixture);
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const docs = root.querySelector<HTMLButtonElement>('.console-footer-link[aria-label^="Docs"]');
        const settings = root.querySelector<HTMLButtonElement>('.console-footer-link[aria-label^="Settings"]');

        expect(docs?.textContent).toContain('Docs');
        expect(docs?.disabled).toBe(true);
        expect(docs?.getAttribute('aria-label')).toContain('not available yet');
        expect(docs?.title).toContain('not available yet');
        expect(settings?.textContent).toContain('Settings');
        expect(settings?.disabled).toBe(true);
        expect(settings?.getAttribute('aria-label')).toContain('not available yet');
        expect(settings?.title).toContain('not available yet');
    });

    it('moves Help focus into the dialog, contains keyboard focus, and restores it after each close path', async () => {
        setBackendResponses(false, true);
        const fixture = TestBed.createComponent(CoreShellComponent);
        fixtures.push(fixture);
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const trigger = root.querySelector<HTMLButtonElement>('[aria-label="Help"]');
        expect(trigger).not.toBeNull();
        if (!trigger) {
            return;
        }

        const openHelp = async (): Promise<HTMLElement> => {
            trigger.click();
            fixture.detectChanges();
            await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
            fixture.detectChanges();
            const dialog = root.querySelector<HTMLElement>('.help-modal');
            if (!dialog) {
                throw new Error('Help dialog did not open');
            }
            return dialog;
        };
        const focusReturnedToTrigger = async (): Promise<void> => {
            await Promise.resolve();
            fixture.detectChanges();
            expect(document.activeElement).toBe(trigger);
        };

        const dialog = await openHelp();
        const closeButton = dialog.querySelector<HTMLButtonElement>('[aria-label="Close help"]');
        const doneButton = dialog.querySelector<HTMLButtonElement>('.help-modal-footer button');
        expect(closeButton).not.toBeNull();
        expect(doneButton).not.toBeNull();
        expect(document.activeElement).toBe(closeButton);
        if (!closeButton || !doneButton) {
            return;
        }

        const shiftTab = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true, cancelable: true });
        closeButton.dispatchEvent(shiftTab);
        expect(shiftTab.defaultPrevented).toBe(true);
        expect(document.activeElement).toBe(doneButton);

        const tab = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
        doneButton.dispatchEvent(tab);
        expect(tab.defaultPrevented).toBe(true);
        expect(document.activeElement).toBe(closeButton);

        closeButton.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        fixture.detectChanges();
        expect(root.querySelector('.help-modal')).toBeNull();
        await focusReturnedToTrigger();

        const doneDialog = await openHelp();
        doneDialog.querySelector<HTMLButtonElement>('.help-modal-footer button')?.click();
        fixture.detectChanges();
        expect(root.querySelector('.help-modal')).toBeNull();
        await focusReturnedToTrigger();

        await openHelp();
        root.querySelector<HTMLElement>('.help-modal-backdrop')?.click();
        fixture.detectChanges();
        expect(root.querySelector('.help-modal')).toBeNull();
        await focusReturnedToTrigger();
    });
});
