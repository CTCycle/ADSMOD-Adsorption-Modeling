import { describe, expect, it } from 'vitest';
import { routes } from './app.routes';

describe('application routes', () => {
    it('registers the split data destinations and keeps datasets as the default', () => {
        const childRoutes = routes[0].children ?? [];
        const paths = childRoutes.map((route) => route.path);

        expect(paths).toEqual(expect.arrayContaining(['datasets', 'public-data', 'public-materials']));
        expect(childRoutes.find((route) => route.path === '')?.redirectTo).toBe('datasets');
        expect(routes.find((route) => route.path === '**')?.redirectTo).toBe('datasets');
        expect(paths).not.toContain('source');
    });

    it('guards the training entry route instead of combining redirectTo with canActivate', () => {
        const trainingRoute = (routes[0].children ?? []).find((route) => route.path === 'training');

        expect(trainingRoute?.redirectTo).toBeUndefined();
        expect(trainingRoute?.canActivate).toHaveLength(1);
        expect(trainingRoute?.loadComponent).toBeTypeOf('function');
    });
});
