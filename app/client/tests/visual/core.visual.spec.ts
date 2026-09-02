import { expect, test } from '@playwright/test';

const prepareVisualPage = async (page: import('@playwright/test').Page) => {
    await page.addInitScript(() => {
        window.localStorage.clear();
    });
};

const disableMotion = async (page: import('@playwright/test').Page) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.addStyleTag({
        content: `
            *,
            *::before,
            *::after {
                animation-duration: 0s !important;
                animation-delay: 0s !important;
                transition-duration: 0s !important;
                transition-delay: 0s !important;
                caret-color: transparent !important;
            }
        `,
    });
};

const mockCoreApi = async (page: import('@playwright/test').Page) => {
    await page.route('**/health/**', async (route) => {
        const url = new URL(route.request().url());
        if (url.pathname === '/health/ready') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ service: 'backend', version: '3.0.0', state: 'ready' }),
            });
            return;
        }
        await route.fulfill({ status: 404, body: 'Not found' });
    });

    await page.route('**/api/**', async (route) => {
        const request = route.request();
        const url = new URL(request.url());

        if (request.method() === 'GET' && url.pathname === '/api/v1/system/capabilities') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
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
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/v1/system/configuration') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    status: 'success',
                    supported_optimizers: ['trf', 'dogbox'],
                    default_optimizer: 'trf',
                    default_max_evaluations: 1000,
                    max_evaluations_bounds: { minimum: 1, maximum: 10000 },
                    weighting_options: ['uniform', 'relative'],
                    default_weighting: 'uniform',
                    display_units: { pressure: ['bar'], uptake: ['mmol/g'], default_pressure: 'bar', default_uptake: 'mmol/g' },
                    parameter_defaults: {},
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/v1/datasets') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    datasets: [
                        {
                            id: 1,
                            name: 'uploaded_demo',
                            source: 'uploaded',
                            created_at: '2026-05-28T09:15:00Z',
                            experiment_count: 2,
                            observation_count: 24,
                            tags: ['demo'],
                            description: 'Uploaded test dataset',
                        },
                        {
                            id: 2,
                            name: 'NIST ISODB',
                            source: 'nist',
                            created_at: '2026-05-29T09:15:00Z',
                            experiment_count: 214,
                            observation_count: 812,
                            tags: [],
                            description: 'Public NIST collection',
                        },
                    ],
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/v1/fitting/models') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    status: 'success',
                    pressure_unit: 'bar',
                    uptake_unit: 'mmol/g',
                    models: [
                        {
                            key: 'langmuir',
                            name: 'Langmuir',
                            equation_latex: 'q = q_{sat} K p / (1 + K p)',
                            assumptions: 'Single-site adsorption at equilibrium.',
                            parameters: [
                                { name: 'q_sat', label: 'q sat', lower: 0, upper: 100, initial: 1, unit: 'mmol/g' },
                                { name: 'K', label: 'K', lower: 0, upper: 100, initial: 1, unit: '1/bar' },
                            ],
                        },
                    ],
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/v1/nist/categories/status') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    status: 'success',
                    categories: [
                        {
                            category: 'experiments',
                            local_count: 214,
                            available_count: 812,
                            last_update: '2026-05-28T09:15:00Z',
                            server_ok: true,
                            server_checked_at: '2026-05-28T09:18:00Z',
                            supports_enrichment: false,
                        },
                        {
                            category: 'guest',
                            local_count: 389,
                            available_count: 389,
                            last_update: '2026-05-29T12:30:00Z',
                            server_ok: true,
                            server_checked_at: '2026-05-29T12:33:00Z',
                            supports_enrichment: true,
                        },
                        {
                            category: 'host',
                            local_count: 144,
                            available_count: 201,
                            last_update: '2026-05-30T15:40:00Z',
                            server_ok: false,
                            server_checked_at: '2026-05-30T15:45:00Z',
                            supports_enrichment: true,
                        },
                    ],
                }),
            });
            return;
        }

        await route.fulfill({
            status: 404,
            contentType: 'application/json',
            body: JSON.stringify({ detail: `Unhandled visual mock for ${request.method()} ${url.pathname}` }),
        });
    });
};

test.describe('core visual regression', () => {
    test.beforeEach(async ({ page }) => {
        await mockCoreApi(page);
        await prepareVisualPage(page);
        await page.goto('/datasets');
        await disableMotion(page);
    });

    test('custom datasets empty state remains visually stable', async ({ page }) => {
        await expect(page.locator('main').getByRole('heading', { name: 'Custom Datasets', exact: true })).toBeVisible();
        await expect(page.getByText('NIST ISODB')).not.toBeVisible();
        await expect(page).toHaveScreenshot('core-custom-datasets-empty-page.png', { fullPage: true });
    });

    test('custom datasets pending upload state remains visually stable', async ({ page }) => {
        await page.setInputFiles('input[type="file"]', {
            name: 'sample.csv',
            mimeType: 'text/csv',
            buffer: Buffer.from('pressure,uptake\n1,2\n3,4\n'),
        });
        await expect(page.getByRole('dialog')).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Understand sample.csv' })).toBeVisible();
        await expect(page).toHaveScreenshot('core-custom-datasets-pending-page.png', { fullPage: true });
    });

    test('public adsorption data page remains visually stable', async ({ page }) => {
        await page.goto('/public-data');
        await expect(page.getByRole('heading', { name: 'Public Adsorption Data' })).toBeVisible();
        await expect(page.getByText('Adsorption experiments', { exact: true })).toBeVisible();
        await expect(page.getByText('Adsorbate species', { exact: true })).not.toBeVisible();
        await expect(page).toHaveScreenshot('core-public-data-page.png', { fullPage: true });
    });

    test('public materials page separates adsorbates and adsorbents', async ({ page }) => {
        await page.goto('/public-materials');
        await expect(page.getByRole('heading', { name: 'Public Materials & Adsorbates' })).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Adsorbates', exact: true })).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Adsorbent Materials', exact: true })).toBeVisible();
        await expect(page).toHaveScreenshot('core-public-materials-page.png', { fullPage: true });
    });
});
