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
});
