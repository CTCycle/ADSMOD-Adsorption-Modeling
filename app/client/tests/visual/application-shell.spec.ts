// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { expect, test, type Page, type TestInfo } from '@playwright/test';

const datasetRecord = {
    id: 17,
    name: 'Example workspace dataset',
    source: 'uploaded',
    created_at: '2026-09-24T12:00:00Z',
    experiment_count: 1,
    observation_count: 4,
    tags: ['team'],
    description: 'A small imported dataset for the shell interaction check.',
};

async function installShellApiMocks(page: Page, machineLearning = true): Promise<void> {
    await page.route('**/health/ready', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ service: 'backend', version: '3.0.0', state: 'starting' }),
        });
    });
    await page.route('**/api/v1/system/capabilities**', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
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
        });
    });
}

async function readShellGeometry(page: Page) {
    return page.evaluate(() => {
        const rect = (selector: string) => {
            const element = document.querySelector<HTMLElement>(selector);
            if (!element) {
                throw new Error(`Missing shell element: ${selector}`);
            }
            const box = element.getBoundingClientRect();
            return { x: box.x, y: box.y, width: box.width, height: box.height, bottom: box.bottom };
        };
        const main = document.querySelector<HTMLElement>('.console-main');
        if (!main) {
            throw new Error('Missing main content region');
        }
        const style = getComputedStyle(main);
        const mainBox = main.getBoundingClientRect();
        return {
            viewportWidth: document.documentElement.clientWidth,
            documentWidth: document.documentElement.scrollWidth,
            headerHeightToken: Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--console-header-height')),
            sidebar: rect('.console-sidebar'),
            header: rect('.console-header'),
            main: rect('.console-main'),
            help: rect('.console-header .header-icon-button'),
            contentOrigin: {
                x: mainBox.x + Number.parseFloat(style.paddingLeft),
                y: mainBox.y + Number.parseFloat(style.paddingTop),
            },
            status: rect('.console-status-bar'),
        };
    });
}

test('primary routes share one stable shell at each supported viewport', async ({ page }, testInfo: TestInfo) => {
    await installShellApiMocks(page);
    const routes = [
        { path: '/datasets', title: 'Custom Datasets', key: 'datasets' },
        { path: '/public-data/overview', title: 'Public Data', key: 'public-data' },
        { path: '/dashboards', title: 'Dashboards', key: 'dashboards' },
        { path: '/fitting', title: 'Fitting', key: 'fitting' },
        { path: '/training/processing', title: 'Training', key: 'training' },
    ];
    let referenceGeometry: Awaited<ReturnType<typeof readShellGeometry>> | null = null;

    for (const route of routes) {
        await page.goto(route.path);
        await expect(page.getByRole('heading', { name: route.title, exact: true })).toBeVisible();
        await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveCount(1);
        const geometry = await readShellGeometry(page);
        expect(geometry.documentWidth).toBeLessThanOrEqual(geometry.viewportWidth + 1);
        expect(geometry.header.height).toBe(geometry.headerHeightToken);
        expect(geometry.help.y).toBeGreaterThanOrEqual(geometry.header.y);
        expect(geometry.help.bottom).toBeLessThanOrEqual(geometry.header.bottom);
        expect(geometry.main.x).toBe(geometry.header.x);
        expect(geometry.status.height).toBe(34);
        expect(geometry.status.bottom).toBe(page.viewportSize()!.height);
        if (referenceGeometry) {
            expect(geometry).toEqual(referenceGeometry);
        } else {
            referenceGeometry = geometry;
        }

        if (route.key === 'training') {
            await expect(page.locator('.training-view-tab[aria-current="page"]')).toHaveCount(1);
            const tabRows = await page.locator('.training-view-tab').evaluateAll((tabs) =>
                Array.from(new Set(tabs.map((tab) => Math.round(tab.getBoundingClientRect().y))))
            );
            expect(tabRows).toHaveLength(1);
        }

        if (route.key === 'datasets') {
            await expect(page.locator('input[type="file"]')).toHaveAttribute('aria-label', 'Choose a dataset file');
        }

        const captureDesktop = testInfo.project.name === 'viewport-1440x920';
        const captureMobile = testInfo.project.name === 'mobile-600x900' && ['datasets', 'training'].includes(route.key);
        const captureKnownOverflowViewport = testInfo.project.name === 'known-overflow-1280x720' && route.key === 'public-data';
        if (captureDesktop || captureMobile || captureKnownOverflowViewport) {
            await page.screenshot({ path: testInfo.outputPath(`${route.key}.png`) });
        }
    }

    if (testInfo.project.name === 'mobile-600x900') {
        for (const viewport of [
            { width: 390, height: 844 },
            { width: 195, height: 422 },
        ]) {
            await page.setViewportSize(viewport);
            await page.goto('/public-data/materials');
            await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveText(/Public Data/);
            const geometry = await readShellGeometry(page);
            expect(geometry.documentWidth).toBeLessThanOrEqual(viewport.width + 1);
            expect(geometry.help.x + geometry.help.width).toBeLessThanOrEqual(viewport.width + 1);

            if (viewport.width === 390) {
                await page.screenshot({ path: testInfo.outputPath('public-materials-mobile-390x844.png') });
            }
        }
    }
});

test('training routes remain gated when the optional service is unavailable', async ({ page }) => {
    await installShellApiMocks(page, false);
    await page.goto('/training/processing');
    await expect(page).toHaveURL(/\/datasets$/);
    await expect(page.getByRole('heading', { name: 'Custom Datasets', exact: true })).toBeVisible();
});

test('dataset tables stay contained and deletion requires an explicit confirmation', async ({ page }, testInfo: TestInfo) => {
    await installShellApiMocks(page, false);
    let dataset: typeof datasetRecord | null = { ...datasetRecord };
    let deleteRequests = 0;
    await page.route(/\/api\/v1\/datasets(?:\/[^?]*)?(?:\?.*)?$/, async (route) => {
        const request = route.request();
        const pathname = new URL(request.url()).pathname;
        if (request.method() === 'GET' && pathname.endsWith('/datasets')) {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ datasets: dataset ? [dataset] : [] }),
            });
            return;
        }
        if (request.method() === 'GET' && pathname.endsWith('/17/experiments')) {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ experiments: [{
                    id: 31,
                    dataset_id: 17,
                    external_key: 'experiment-with-a-long-stable-key-31',
                    name: 'Carbon dioxide adsorption experiment',
                    adsorbent: 'Hierarchical activated carbon framework',
                    adsorbates: ['carbon dioxide'],
                    temperature_k: 298.15,
                    pressure_basis: 'absolute',
                    observation_count: 2,
                    fitting_eligible: true,
                    ineligibility_reason: null,
                }] }),
            });
            return;
        }
        if (request.method() === 'GET' && pathname.endsWith('/17/experiments/31/observations')) {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    status: 'success',
                    dataset_id: 17,
                    isotherm_id: 31,
                    offset: 0,
                    limit: 10,
                    total: 2,
                    rows: [
                        { id: 1, source_row: 2, pressure_original: 1, pressure_original_unit: 'bar', pressure_canonical: 100000, pressure_canonical_unit: 'Pa', uptake_original: 0.1, uptake_original_unit: 'mmol/g', uptake_mol_kg: 0.1 },
                        { id: 2, source_row: 3, pressure_original: 2, pressure_original_unit: 'bar', pressure_canonical: 200000, pressure_canonical_unit: 'Pa', uptake_original: 0.2, uptake_original_unit: 'mmol/g', uptake_mol_kg: 0.2 },
                    ],
                }),
            });
            return;
        }
        if (request.method() === 'PATCH' && pathname.endsWith('/17/rename')) {
            const body = request.postDataJSON() as { new_name: string };
            dataset = { ...datasetRecord, name: body.new_name };
            await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ dataset }) });
            return;
        }
        if (request.method() === 'DELETE' && pathname.endsWith('/17')) {
            deleteRequests += 1;
            dataset = null;
            await route.fulfill({ status: 204, body: '' });
            return;
        }
        await route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
    });

    await page.goto('/datasets');
    await expect(page.getByRole('button', { name: datasetRecord.name, exact: true })).toBeVisible();
    const datasetTableGeometry = await page.locator('.dataset-table-frame').evaluate((frame) => {
        const card = frame.closest<HTMLElement>('.workspace-datasets-card');
        const table = frame.querySelector('table');
        if (!card || !table) throw new Error('Dataset table geometry is unavailable');
        return {
            frameLeft: frame.getBoundingClientRect().left,
            frameRight: frame.getBoundingClientRect().right,
            cardLeft: card.getBoundingClientRect().left,
            cardRight: card.getBoundingClientRect().right,
            frameWidth: frame.clientWidth,
            frameScrollWidth: frame.scrollWidth,
            tableWidth: table.getBoundingClientRect().width,
            viewportWidth: document.documentElement.clientWidth,
            documentWidth: document.documentElement.scrollWidth,
        };
    });
    expect(datasetTableGeometry.frameLeft).toBeGreaterThanOrEqual(datasetTableGeometry.cardLeft);
    expect(datasetTableGeometry.frameRight).toBeLessThanOrEqual(datasetTableGeometry.cardRight + 1);
    expect(datasetTableGeometry.documentWidth).toBeLessThanOrEqual(datasetTableGeometry.viewportWidth + 1);
    if (datasetTableGeometry.frameWidth >= 900) {
        expect(datasetTableGeometry.tableWidth).toBeGreaterThanOrEqual(datasetTableGeometry.frameWidth - 1);
    } else {
        expect(datasetTableGeometry.frameScrollWidth).toBeGreaterThan(datasetTableGeometry.frameWidth);
    }
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('datasets-populated-desktop.png') });
    }
    await page.getByRole('button', { name: `Rename ${datasetRecord.name}`, exact: true }).click();
    await page.getByLabel('Dataset name').fill('Renamed workspace dataset');
    await page.getByRole('button', { name: 'Save name' }).click();
    await expect(page.getByRole('button', { name: 'Renamed workspace dataset', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Renamed workspace dataset', exact: true }).click();
    await expect(page.locator('.dataset-experiments-frame')).toBeVisible();
    await expect(page.locator('.dataset-observations-frame')).toBeVisible();
    const inspectorTableGeometry = await page.locator('.dataset-inspector').evaluate((inspector) => {
        const frames = Array.from(inspector.querySelectorAll<HTMLElement>('.data-table-frame'));
        const inspectorRect = inspector.getBoundingClientRect();
        return {
            viewportWidth: document.documentElement.clientWidth,
            documentWidth: document.documentElement.scrollWidth,
            frames: frames.map((frame) => ({
                left: frame.getBoundingClientRect().left,
                right: frame.getBoundingClientRect().right,
                clientWidth: frame.clientWidth,
                scrollWidth: frame.scrollWidth,
                tableWidth: frame.querySelector('table')?.getBoundingClientRect().width ?? 0,
            })),
            inspectorLeft: inspectorRect.left,
            inspectorRight: inspectorRect.right,
        };
    });
    expect(inspectorTableGeometry.documentWidth).toBeLessThanOrEqual(inspectorTableGeometry.viewportWidth + 1);
    expect(inspectorTableGeometry.frames).toHaveLength(2);
    for (const frame of inspectorTableGeometry.frames) {
        expect(frame.left).toBeGreaterThanOrEqual(inspectorTableGeometry.inspectorLeft);
        expect(frame.right).toBeLessThanOrEqual(inspectorTableGeometry.inspectorRight + 1);
        if (frame.clientWidth < frame.tableWidth) {
            expect(frame.scrollWidth).toBeGreaterThan(frame.clientWidth);
        }
    }
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('dataset-inspector-populated-desktop.png') });
    }

    await page.getByRole('button', { name: 'Delete Renamed workspace dataset', exact: true }).click();
    const confirmation = page.getByRole('alertdialog');
    await expect(confirmation).toContainText('Renamed workspace dataset');
    await expect(confirmation.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(confirmation.getByRole('button', { name: 'Delete dataset', exact: true })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(confirmation.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(confirmation).toHaveCount(0);
    expect(deleteRequests).toBe(0);

    await page.getByRole('button', { name: 'Delete Renamed workspace dataset', exact: true }).click();
    await expect(confirmation.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(confirmation).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Delete Renamed workspace dataset', exact: true })).toBeFocused();
    await page.getByRole('button', { name: 'Delete Renamed workspace dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Delete dataset', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Renamed workspace dataset', exact: true })).toHaveCount(0);
    expect(deleteRequests).toBe(1);
});

test('dataset import keeps keyboard focus inside the wizard and restores the trigger', async ({ page }, testInfo: TestInfo) => {
    await installShellApiMocks(page, false);
    await page.route('**/api/v1/datasets', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ datasets: [] }) });
    });
    await page.route('**/api/v1/datasets/supported-units', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ allowed_extensions: ['.csv', '.xls', '.xlsx'] }),
        });
    });
    await page.route('**/api/v1/datasets/import/preview', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                status: 'success',
                filename: 'sample.csv',
                source_sha256: 'test-sha256',
                row_count: 2,
                column_count: 3,
                detected_structure: 'atomic',
                structure_confidence: 0.95,
                columns: [
                    { name: 'pressure', inferred_type: 'number', proposed_role: 'pressure', detected_unit: 'bar' },
                    { name: 'uptake', inferred_type: 'number', proposed_role: 'uptake', detected_unit: 'mmol/g' },
                    { name: 'temperature', inferred_type: 'number', proposed_role: 'temperature', detected_unit: 'K' },
                ],
                preview_rows: [{ pressure: 1, uptake: 0.1, temperature: 298 }],
                proposed_grouping_columns: [],
                proposed_pressure_basis: 'absolute',
                issues: [],
                guidance: [],
            }),
        });
    });

    await page.goto('/datasets');
    const chooserPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Add dataset', exact: true }).click();
    const chooser = await chooserPromise;
    await chooser.setFiles({
        name: 'sample.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from('pressure,uptake,temperature\n1,0.1,298\n'),
    });

    const dialog = page.getByRole('dialog', { name: 'Understand sample.csv' });
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('.import-preview-table')).toBeVisible();
    const previewGeometry = await dialog.locator('.import-preview-table').evaluate((table) => {
        const frame = table.parentElement;
        const body = table.closest('.import-wizard-body');
        if (!frame || !body) throw new Error('Import preview geometry is unavailable');
        return {
            frameLeft: frame.getBoundingClientRect().left,
            frameRight: frame.getBoundingClientRect().right,
            bodyLeft: body.getBoundingClientRect().left,
            bodyRight: body.getBoundingClientRect().right,
            viewportWidth: document.documentElement.clientWidth,
            documentWidth: document.documentElement.scrollWidth,
        };
    });
    expect(previewGeometry.frameLeft).toBeGreaterThanOrEqual(previewGeometry.bodyLeft);
    expect(previewGeometry.frameRight).toBeLessThanOrEqual(previewGeometry.bodyRight + 1);
    expect(previewGeometry.documentWidth).toBeLessThanOrEqual(previewGeometry.viewportWidth + 1);
    const closeButton = dialog.getByRole('button', { name: 'Close import wizard', exact: true });
    await expect(closeButton).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(dialog.getByRole('button', { name: 'Review mapping', exact: true })).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(closeButton).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Add dataset', exact: true })).toBeFocused();

    if (testInfo.project.name !== 'mobile-600x900') {
        return;
    }

    await page.setViewportSize({ width: 390, height: 844 });
    const mobileChooserPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Add dataset', exact: true }).click();
    const mobileChooser = await mobileChooserPromise;
    await mobileChooser.setFiles({
        name: 'sample.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from('pressure,uptake,temperature\n1,0.1,298\n'),
    });

    const mobileDialog = page.getByRole('dialog', { name: 'Understand sample.csv' });
    await mobileDialog.getByRole('button', { name: 'Review mapping', exact: true }).click();

    const mappingGeometry = await mobileDialog.evaluate((element) => {
        const settings = element.querySelector<HTMLElement>('.import-settings-grid');
        const table = element.querySelector<HTMLElement>('.import-mapping-table');
        const tableWrap = table?.parentElement;
        const body = element.querySelector<HTMLElement>('.import-wizard-body');
        return {
            settingsColumns: settings ? getComputedStyle(settings).gridTemplateColumns.trim().split(/\s+/).length : 0,
            documentWidth: document.documentElement.scrollWidth,
            viewportWidth: document.documentElement.clientWidth,
            tableWrapWidth: tableWrap?.getBoundingClientRect().width ?? 0,
            tableWrapScrollWidth: tableWrap?.scrollWidth ?? 0,
            tableWrapClientWidth: tableWrap?.clientWidth ?? 0,
            bodyWidth: body?.getBoundingClientRect().width ?? 0,
        };
    });
    expect(mappingGeometry.settingsColumns).toBe(1);
    expect(mappingGeometry.documentWidth).toBeLessThanOrEqual(mappingGeometry.viewportWidth + 1);
    expect(mappingGeometry.tableWrapWidth).toBeLessThanOrEqual(mappingGeometry.bodyWidth + 1);
    expect(mappingGeometry.tableWrapScrollWidth).toBeGreaterThan(mappingGeometry.tableWrapClientWidth);

    await page.setViewportSize({ width: 195, height: 422 });
    await expect(mobileDialog.getByRole('button', { name: 'Validate and preview series', exact: true })).toBeVisible();
    const narrowDocumentWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(narrowDocumentWidth).toBeLessThanOrEqual(196);
});
