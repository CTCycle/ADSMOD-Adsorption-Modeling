// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it, afterEach, beforeEach, vi } from 'vitest';
import { routes } from './app.routes';
import { CoreShellComponent } from './layout/core-shell.component';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

// Route table behavior

describe('application routes', () => {
    it('keeps implemented shell destinations reachable and recovers unknown routes to datasets', () => {
        const childRoutes = routes[0].children ?? [];
        const paths = childRoutes.map((route) => route.path);

        expect(routes[0]?.component).toBe(CoreShellComponent);
        expect(paths).toEqual(
            expect.arrayContaining([
                'datasets',
                'public-data',
                'public-data/:view',
                'dashboards',
                'fitting',
                'training',
                'training/:view',
            ])
        );
        expect(paths).not.toContain('public-materials');
        const dashboards = childRoutes.find((route) => route.path === 'dashboards');
        expect(dashboards?.children?.find((route) => route.path === '')?.redirectTo).toBe('overview');
        expect(dashboards?.children?.map((route) => route.path)).toEqual(['', 'overview', 'explore']);
        expect(childRoutes.find((route) => route.path === 'public-data')?.redirectTo).toBe('public-data/overview');
        expect(childRoutes.find((route) => route.path === '')?.redirectTo).toBe('datasets');
        expect(routes.find((route) => route.path === '**')?.redirectTo).toBe('datasets');
    });

    it('guards the training entry route instead of combining redirectTo with canActivate', () => {
        const trainingRoute = (routes[0].children ?? []).find((route) => route.path === 'training');

        expect(trainingRoute?.redirectTo).toBeUndefined();
        expect(trainingRoute?.canActivate).toHaveLength(1);
        expect(trainingRoute?.loadComponent).toBeTypeOf('function');

        const trainingViewRoute = (routes[0].children ?? []).find((route) => route.path === 'training/:view');
        expect(trainingViewRoute?.canActivate).toHaveLength(1);
        expect(trainingViewRoute?.loadComponent).toBeTypeOf('function');
    });
});

// Shell lifecycle and capability navigation

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

    it('links Docs to the documentation page, removes the header help button, and exposes the Settings route', () => {
        const fixture = TestBed.createComponent(CoreShellComponent);
        fixtures.push(fixture);
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const docs = root.querySelector<HTMLAnchorElement>('.console-footer-link[aria-label="Docs"]');
        const settings = root.querySelector<HTMLAnchorElement>('.console-footer-link[routerLink="/settings"]');

        expect(docs?.textContent).toContain('Docs');
        expect(docs?.getAttribute('href')).toBe('/help#dataset-management');
        expect(docs?.title).toBe('Documentation, tutorials, and tips');
        expect(root.querySelector('[aria-label="Help"]')).toBeNull();
        expect(settings?.textContent).toContain('Settings');
        expect(settings?.getAttribute('href')).toBe('/settings');
        expect(settings?.getAttribute('aria-label')).toBe('Settings');
        expect(settings?.title).toBe('Application settings');
    });
});
