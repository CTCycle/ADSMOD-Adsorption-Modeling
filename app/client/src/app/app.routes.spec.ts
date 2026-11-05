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
                'datasets/:id',
                'sources',
                'sources/:view',
                'dashboards',
                'modeling',
            ])
        );
        expect(paths).not.toContain('fitting');
        expect(paths).not.toContain('training');
        expect(paths).not.toContain('public-data');
        expect(paths).not.toContain('public-data/:view');
        const dashboards = childRoutes.find((route) => route.path === 'dashboards');
        expect(dashboards?.children?.find((route) => route.path === '')?.redirectTo).toBe('overview');
        expect(dashboards?.children?.map((route) => route.path)).toEqual(['', 'overview', 'explore']);
        expect(childRoutes.find((route) => route.path === 'sources')?.redirectTo).toBe('sources/catalog');
        expect(childRoutes.find((route) => route.path === 'datasets/:id')?.canDeactivate).toHaveLength(1);
        expect(childRoutes.find((route) => route.path === '')?.redirectTo).toBe('datasets');
        expect(routes.find((route) => route.path === '**')?.redirectTo).toBe('datasets');
    });

    it('nests the Modeling workspace views and guards the deep-learning entry routes', () => {
        const modeling = (routes[0].children ?? []).find((route) => route.path === 'modeling');
        const children = modeling?.children ?? [];

        expect(children.map((route) => route.path)).toEqual([
            '',
            'fitting',
            'fitting/new',
            'fitting/:sessionId',
            'models',
            'deep-learning',
            'deep-learning/:view',
        ]);
        expect(children.find((route) => route.path === '')?.redirectTo).toBe('fitting');

        const entry = children.find((route) => route.path === 'deep-learning');
        expect(entry?.redirectTo).toBeUndefined();
        expect(entry?.canActivate).toHaveLength(1);
        expect(entry?.loadComponent).toBeTypeOf('function');
        expect(children.find((route) => route.path === 'deep-learning/:view')?.canActivate).toHaveLength(1);
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

    it('recovers backend status from Offline to Online', async () => {
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

        setBackendResponses(null, false);

        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        expect(statusBar?.textContent).toContain('BackendOffline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(true);

        setBackendResponses(true, true);

        await (fixture.componentInstance as unknown as { refreshBackendStatus: () => Promise<void> }).refreshBackendStatus();
        fixture.detectChanges();

        expect(statusBar?.textContent).toContain('BackendOnline');
        expect(root.querySelector('.service-dot.core')?.classList.contains('offline')).toBe(false);
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
