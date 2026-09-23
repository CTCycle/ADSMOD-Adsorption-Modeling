import { describe, expect, it } from 'vitest';
import { routes } from './app.routes';
import { CoreShellComponent } from './layout/core-shell.component';

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
        expect(childRoutes.find((route) => route.path === 'datasets')?.loadComponent).toBeTypeOf('function');
        expect(childRoutes.find((route) => route.path === 'public-data/:view')?.loadComponent).toBeTypeOf('function');
        expect(childRoutes.find((route) => route.path === 'dashboards')?.loadComponent).toBeTypeOf('function');
        expect(childRoutes.find((route) => route.path === 'fitting')?.loadComponent).toBeTypeOf('function');
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
