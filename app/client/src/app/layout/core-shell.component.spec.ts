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

    it('renders unified backend status and exposes training only when ML is available', async () => {
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/system/capabilities')) {
                return {
                    ok: true,
                    json: async () => ({
                        version: '3.0.0',
                        features: {
                            datasets: true,
                            nist: true,
                            fitting: true,
                            machine_learning: false,
                            training: false,
                            checkpoints: false,
                        },
                    }),
                };
            }
            if (url.endsWith('/health/ready')) {
                return {
                    ok: true,
                    json: async () => ({ service: 'backend', version: '3.0.0', state: 'ready' }),
                };
            }
            return { ok: false, status: 404 };
        });

        const fixture = TestBed.createComponent(CoreShellComponent);
        fixture.detectChanges();
        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        let statusBar = root.querySelector<HTMLElement>('.console-status-bar');
        expect(statusBar?.textContent).toContain('BackendOnline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(false);
        expect(root.querySelector('a[routerLink="/training"]')).toBeNull();

        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/system/capabilities')) {
                return {
                    ok: true,
                    json: async () => ({
                        version: '3.0.0',
                        features: {
                            datasets: true,
                            nist: true,
                            fitting: true,
                            machine_learning: true,
                            training: true,
                            checkpoints: true,
                        },
                    }),
                };
            }
            if (url.endsWith('/health/ready')) {
                throw new Error('backend unavailable');
            }
            return { ok: false, status: 404 };
        });

        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        statusBar = root.querySelector<HTMLElement>('.console-status-bar');
        expect(statusBar?.textContent).toContain('BackendOffline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(true);
        expect(root.querySelector('a[routerLink="/training"]')).not.toBeNull();
    });
});
