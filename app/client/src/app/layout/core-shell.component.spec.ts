import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreShellComponent } from './core-shell.component';

describe('CoreShellComponent', () => {
    const fetchMock = vi.fn();

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
        vi.unstubAllGlobals();
    });

    it('renders health-status transitions for core and ML services', async () => {
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/health')) {
                return { ok: true };
            }
            return { ok: false };
        });

        const fixture = TestBed.createComponent(CoreShellComponent);
        fixture.detectChanges();
        await Promise.resolve();
        await Promise.resolve();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        let statusBar = root.querySelector<HTMLElement>('.console-status-bar');
        expect(statusBar?.textContent).toContain('Core ServiceOnline');
        expect(statusBar?.textContent).toContain('ML ServiceUnavailable');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(false);
        expect(root.querySelector('.service-dot.ml')?.classList.contains('offline')).toBe(true);

        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/health')) {
                throw new Error('core unavailable');
            }
            return { ok: true };
        });

        await (fixture.componentInstance as unknown as { refreshServiceStatus: () => Promise<void> }).refreshServiceStatus();
        fixture.detectChanges();

        statusBar = root.querySelector<HTMLElement>('.console-status-bar');
        expect(statusBar?.textContent).toContain('Core ServiceOffline');
        expect(statusBar?.textContent).toContain('ML ServiceOnline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(true);
        expect(root.querySelector('.service-dot.ml')?.classList.contains('offline')).toBe(false);
    });
});
