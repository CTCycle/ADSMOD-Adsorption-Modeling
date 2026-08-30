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
            if (url.endsWith('/system/capabilities')) {
                return {
                    ok: true,
                    json: async () => ({
                        configured_mode: 'core',
                        version: '3.0.0',
                        features: { datasets: true, nist: true, fitting: true, training: false, checkpoints: false },
                        services: { ml: { configured: false, health: 'unknown', readiness: 'unavailable' } },
                    }),
                };
            }
            if (url.endsWith('/health/ready')) {
                return { ok: true, json: async () => ({ service: 'core', version: '3.0.0', state: 'ready' }) };
            }
            if (url.endsWith('/ml-health/ready')) {
                return { ok: false, status: 503, json: async () => ({ detail: 'ML not configured' }) };
            }
            return { ok: false };
        });

        const fixture = TestBed.createComponent(CoreShellComponent);
        fixture.detectChanges();
        await (fixture.componentInstance as unknown as { refreshServiceStatus: () => Promise<void> }).refreshServiceStatus();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        let statusBar = root.querySelector<HTMLElement>('.console-status-bar');
        expect(statusBar?.textContent).toContain('Core ServiceOnline');
        expect(statusBar?.textContent).toContain('ML ServiceUnavailable');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(false);
        expect(root.querySelector('.service-dot.ml')?.classList.contains('offline')).toBe(true);

        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/system/capabilities')) {
                return {
                    ok: true,
                    json: async () => ({
                        configured_mode: 'core-ml',
                        version: '3.0.0',
                        features: { datasets: true, nist: true, fitting: true, training: true, checkpoints: true },
                        services: { ml: { configured: true, health: 'healthy', readiness: 'ready' } },
                    }),
                };
            }
            if (url.endsWith('/health/ready')) {
                throw new Error('core unavailable');
            }
            if (url.endsWith('/ml-health/ready')) {
                return { ok: true, json: async () => ({ service: 'ml', version: '3.0.0', state: 'ready' }) };
            }
            return { ok: false };
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
