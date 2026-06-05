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
    await page.route('**/api/**', async (route) => {
        const request = route.request();
        const url = new URL(request.url());

        if (request.method() === 'GET' && url.pathname === '/api/datasets/names') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    names: ['baseline_screening_run', 'zeolite_batch_august'],
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/nist/categories/status') {
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
            body: JSON.stringify({
                detail: `Unhandled visual mock for ${request.method()} ${url.pathname}`,
            }),
        });
    });
};

test.describe('core visual regression', () => {
    test.beforeEach(async ({ page }) => {
        await mockCoreApi(page);
        await prepareVisualPage(page);
        await page.goto('/source');
        await disableMotion(page);
    });

    test('source empty state remains visually stable', async ({ page }) => {
        await expect(page.locator('.section-title').filter({ hasText: 'Load Experimental Data' })).toBeVisible();
        await expect(page).toHaveScreenshot('core-source-empty-page.png', { fullPage: true });
    });

    test('source pending upload state remains visually stable', async ({ page }) => {
        await page.setInputFiles('input[type="file"]', {
            name: 'sample.csv',
            mimeType: 'text/csv',
            buffer: Buffer.from('pressure,uptake\n1,2\n3,4\n'),
        });
        await expect(page.locator('.dataset-upload-button')).toBeEnabled();
        await expect(page).toHaveScreenshot('core-source-pending-page.png', { fullPage: true });
    });

    test('fitting ready state remains visually stable', async ({ page }) => {
        await page.goto('/fitting');
        await expect(page.getByRole('heading', { name: 'Select Adsorption Models' })).toBeVisible();
        await expect(page).toHaveScreenshot('core-fitting-ready-page.png', { fullPage: true });
    });

    test('fitting expanded card remains visually stable', async ({ page }) => {
        await page.goto('/fitting');
        await expect(page.getByRole('heading', { name: 'Select Adsorption Models' })).toBeVisible();
        await page.locator('.model-card-header').first().click();
        await expect(page).toHaveScreenshot('core-fitting-expanded-page.png', { fullPage: true });
    });

    test('fitting disabled card remains visually stable', async ({ page }) => {
        await page.goto('/fitting');
        await expect(page.getByRole('heading', { name: 'Select Adsorption Models' })).toBeVisible();
        await page.locator('.switch').first().click();
        await expect(page.locator('.model-grid-card').first()).toHaveClass(/disabled/);
        await expect(page).toHaveScreenshot('core-fitting-disabled-page.png', { fullPage: true });
    });
});
