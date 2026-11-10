// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { expect, test, type Page, type TestInfo } from '@playwright/test';

const datasetRecord = {
    id: 17,
    name: 'Example workspace dataset',
    source: 'uploaded',
    created_at: '2026-09-24T12:00:00Z',
    last_update: '2026-09-25T08:30:00Z',
    experiment_count: 1,
    observation_count: 2,
    edited_experiment_count: 0,
    material_count: 1,
    species_count: 1,
    material_names: ['Hierarchical activated carbon framework'],
    species_names: ['carbon dioxide'],
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
            themeSwitch: rect('.console-header adsmod-theme-switch'),
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
        { path: '/home', title: 'Home', key: 'home' },
        { path: '/datasets', title: 'Datasets', key: 'datasets' },
        { path: '/sources/catalog', title: 'Sources', key: 'sources' },
        { path: '/dashboards', title: 'Dashboards', key: 'dashboards' },
        { path: '/modeling/fitting', title: 'Modeling', key: 'fitting' },
        { path: '/modeling/deep-learning/processing', title: 'Modeling', key: 'training' },
    ];
    let referenceGeometry: Awaited<ReturnType<typeof readShellGeometry>> | null = null;

    for (const route of routes) {
        await page.goto(route.path);
        await expect(page.getByRole('heading', { name: route.title, exact: true })).toBeVisible();
        await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveCount(1);
        const geometry = await readShellGeometry(page);
        expect(geometry.documentWidth).toBeLessThanOrEqual(geometry.viewportWidth + 1);
        expect(geometry.header.height).toBe(geometry.headerHeightToken);
        expect(geometry.themeSwitch.y).toBeGreaterThanOrEqual(geometry.header.y);
        expect(geometry.themeSwitch.bottom).toBeLessThanOrEqual(geometry.header.bottom);
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
        const captureKnownOverflowViewport = testInfo.project.name === 'known-overflow-1280x720' && route.key === 'sources';
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
            await page.goto('/sources/structures');
            await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveText(/Sources/);
            const geometry = await readShellGeometry(page);
            expect(geometry.documentWidth).toBeLessThanOrEqual(viewport.width + 1);
            expect(geometry.themeSwitch.x + geometry.themeSwitch.width).toBeLessThanOrEqual(viewport.width + 1);

            if (viewport.width === 390) {
                await page.screenshot({ path: testInfo.outputPath('sources-structures-mobile-390x844.png') });
            }
        }
    }
});

test('Jobs popover keeps about ten rows visible and scrolls the rest inside the list', async ({ page }) => {
    await installShellApiMocks(page);
    const jobs = Array.from({ length: 30 }, (_, index) => ({
        id: `job-${index}`, kind: 'fitting', name: `Fitting session ${index}`, source: null, status: 'completed', progress: 100,
        created_at: new Date(Date.UTC(2026, 10, 5, 12, 0) - index * 60_000).toISOString(), finished_at: null,
        saved_count: 4, skipped_count: 0, failed_count: 0, error: null, run: null,
    }));
    await page.route('**/api/v1/jobs**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ jobs, active_count: 0 }) });
    });
    await page.goto('/home');
    await page.getByRole('button', { name: /Jobs/ }).click();

    const list = page.locator('.jobs-list');
    await expect(list.locator('.job-row')).toHaveCount(30);
    const metrics = await list.evaluate((element) => {
        const row = element.querySelector('.job-row')!.getBoundingClientRect().height;
        const panel = element.closest('.jobs-panel')!.getBoundingClientRect();
        return { visible: element.clientHeight / row, scrollable: element.scrollHeight > element.clientHeight, panelTop: panel.top, panelBottom: panel.bottom };
    });
    expect(metrics.scrollable).toBe(true);
    expect(metrics.visible).toBeLessThanOrEqual(10.5);
    expect(metrics.panelTop).toBeGreaterThanOrEqual(0);
    expect(metrics.panelBottom).toBeLessThanOrEqual(page.viewportSize()!.height);
});

test('training routes remain gated when the optional service is unavailable', async ({ page }) => {
    await installShellApiMocks(page, false);
    await page.goto('/modeling/deep-learning/processing');
    await expect(page).toHaveURL(/\/datasets$/);
    await expect(page.getByRole('heading', { name: 'Datasets', exact: true })).toBeVisible();
});

const experimentRecord = {
    id: 31,
    dataset_id: 17,
    external_key: 'experiment-with-a-long-stable-key-31',
    name: 'Carbon dioxide adsorption experiment',
    adsorbent_id: 4,
    adsorbent: 'Hierarchical activated carbon framework',
    adsorbates: ['carbon dioxide'],
    components: [{ position: 1, adsorbate_id: 7, name: 'carbon dioxide' }],
    temperature_k: 298.15,
    temperature_original: 298.15,
    temperature_original_unit: 'K',
    pressure_basis: 'absolute',
    saturation_pressure_pa: null,
    duplicate_policy: 'reject',
    observation_count: 2,
    pressure_min: 100000,
    pressure_max: 200000,
    pressure_canonical_unit: 'Pa',
    pressure_original_unit: 'bar',
    uptake_original_unit: 'mmol/g',
    edited_at: null,
    fitting_eligible: true,
    ineligibility_reason: null,
};

test('datasets, materials, and species stay contained; edits save and deletion is confirmed', async ({ page }, testInfo: TestInfo) => {
    await installShellApiMocks(page, false);
    let dataset: typeof datasetRecord | null = { ...datasetRecord };
    let uptake = 0.1;
    let deleteRequests = 0;
    let patchBody: Record<string, unknown> | null = null;
    const json = (body: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    await page.route(/\/api\/v1\/(materials|species)(?:\?.*)?$/, async (route) => {
        const isMaterial = new URL(route.request().url()).pathname.endsWith('/materials');
        const item = isMaterial
            ? { id: 4, name: 'Hierarchical activated carbon framework', formula: null, molar_mass_g_mol: null, smiles: null, structure_count: 0, isotherm_count: dataset ? 1 : 0, external_identifiers: [] }
            : { id: 7, name: 'carbon dioxide', preferred_name: null, formula: 'CO2', molecular_weight: 44.01, inchi: null, inchi_key: null, connectivity_smiles: null, smiles: null, pubchem_cid: null, synonyms: [], properties: [], external_identifiers: [], structure_2d_url: null, conformer_3d_url: null, isotherm_count: dataset ? 1 : 0, retrieved_at: null };
        await route.fulfill(json({ items: [item], pagination: { page: 1, page_size: 25, total: 1 } }));
    });
    await page.route(/\/api\/v1\/datasets(?:\/[^?]*)?(?:\?.*)?$/, async (route) => {
        const request = route.request();
        const pathname = new URL(request.url()).pathname;
        if (request.method() === 'GET' && pathname.endsWith('/datasets')) {
            await route.fulfill(json({ datasets: dataset ? [dataset] : [] }));
        } else if (request.method() === 'GET' && pathname.endsWith('/datasets/supported-units')) {
            await route.fulfill(json({ pressure: [], uptake: [], temperature: [], allowed_extensions: ['.csv'] }));
        } else if (request.method() === 'GET' && pathname.endsWith('/datasets/17')) {
            await route.fulfill(json({ status: 'success', dataset, provenance: {}, latest_import: null }));
        } else if (request.method() === 'GET' && pathname.endsWith('/17/delete-impact')) {
            await route.fulfill(json({ dataset_id: 17, experiment_count: 1, observation_count: 2, fitting_run_count: 3 }));
        } else if (request.method() === 'GET' && pathname.endsWith('/17/experiments')) {
            await route.fulfill(json({ status: 'success', dataset_id: 17, offset: 0, limit: 50, total: 1, experiments: [experimentRecord] }));
        } else if (request.method() === 'GET' && pathname.endsWith('/17/experiments/31/observations')) {
            await route.fulfill(json({
                status: 'success', dataset_id: 17, isotherm_id: 31, offset: 0, limit: 100, total: 2,
                rows: [
                    { id: 1, adsorbate_id: 7, gas_mole_fraction: null, uptake_stddev_mol_kg: null, source_row: 2, pressure_original: 1, pressure_original_unit: 'bar', pressure_canonical: 100000, pressure_canonical_unit: 'Pa', uptake_original: uptake, uptake_original_unit: 'mmol/g', uptake_mol_kg: uptake },
                    { id: 2, adsorbate_id: 7, gas_mole_fraction: null, uptake_stddev_mol_kg: null, source_row: 3, pressure_original: 2, pressure_original_unit: 'bar', pressure_canonical: 200000, pressure_canonical_unit: 'Pa', uptake_original: 0.2, uptake_original_unit: 'mmol/g', uptake_mol_kg: 0.2 },
                ],
            }));
        } else if (request.method() === 'PATCH' && pathname.endsWith('/datasets/17')) {
            patchBody = request.postDataJSON() as Record<string, unknown>;
            uptake = 0.15;
            dataset = { ...datasetRecord, last_update: '2026-09-26T09:00:00Z', edited_experiment_count: 1 };
            await route.fulfill(json({ status: 'success', dataset }));
        } else if (request.method() === 'DELETE' && pathname.endsWith('/datasets/17')) {
            deleteRequests += 1;
            dataset = null;
            await route.fulfill({ status: 204, body: '' });
        } else {
            await route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
        }
    });

    await page.goto('/datasets');
    await expect(page.getByRole('link', { name: datasetRecord.name, exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: /Materials/ })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByRole('button', { name: 'Show details for Hierarchical activated carbon framework' })).toBeVisible();
    const listGeometry = await page.evaluate(() => {
        const primary = document.querySelector('.datasets-primary')!.getBoundingClientRect();
        const secondary = document.querySelector('.datasets-secondary')!.getBoundingClientRect();
        return {
            viewportWidth: document.documentElement.clientWidth,
            documentWidth: document.documentElement.scrollWidth,
            sideBySide: secondary.left >= primary.right,
            primaryWidth: primary.width,
            secondaryWidth: secondary.width,
        };
    });
    expect(listGeometry.documentWidth).toBeLessThanOrEqual(listGeometry.viewportWidth + 1);
    if (listGeometry.viewportWidth >= 1200) {
        expect(listGeometry.sideBySide).toBe(true);
        expect(listGeometry.primaryWidth).toBeGreaterThan(listGeometry.secondaryWidth * 1.5);
    } else {
        expect(listGeometry.sideBySide).toBe(false);
    }
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('datasets-populated-desktop.png') });
    }

    await page.getByRole('link', { name: `Explore ${datasetRecord.name}` }).click();
    await expect(page).toHaveURL(/\/datasets\/17\?experiment=31$/);
    await expect(page.getByRole('heading', { name: datasetRecord.name, level: 2 })).toBeVisible();
    const worksheet = page.getByRole('table', { name: /^Measurements of/ });
    await expect(worksheet.locator('tbody tr')).toHaveCount(2);
    await expect(worksheet.locator('input')).toHaveCount(0);

    await page.getByRole('button', { name: 'Edit dataset' }).click();
    await expect(page.getByText('Editing', { exact: true })).toBeVisible();
    const cell = page.getByLabel('Uptake for measurement 1');
    await cell.fill('0.15');
    await cell.press('Enter');
    await expect(page.getByLabel('Uptake for measurement 2')).toBeFocused();
    await expect(page.getByText(/1 unsaved change\b/)).toBeVisible();
    const detailsGeometry = await page.evaluate(() => ({
        viewportWidth: document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
    }));
    expect(detailsGeometry.documentWidth).toBeLessThanOrEqual(detailsGeometry.viewportWidth + 1);
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('dataset-details-editing-desktop.png') });
    }
    await page.getByRole('button', { name: 'Save changes' }).click();
    await expect(page.getByText('Changes saved.', { exact: true })).toBeVisible();
    expect(patchBody).toEqual({ observations: [{ id: 1, uptake_original: 0.15 }] });
    await expect(worksheet.locator('input')).toHaveCount(0);
    await expect(worksheet.locator('tbody tr').first()).toContainText('0.15');

    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    const confirmation = page.getByRole('alertdialog');
    await expect(confirmation).toContainText('1 experiments, 2 measurements and 3 saved fitting runs');
    await expect(confirmation.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(confirmation.getByRole('button', { name: 'Delete dataset', exact: true })).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(confirmation).toHaveCount(0);
    expect(deleteRequests).toBe(0);

    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    await page.getByRole('button', { name: 'Delete dataset', exact: true }).click();
    await expect(page).toHaveURL(/\/datasets$/);
    await expect(page.getByRole('link', { name: datasetRecord.name, exact: true })).toHaveCount(0);
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
            body: JSON.stringify({ pressure: [], uptake: [], temperature: [], allowed_extensions: ['.csv', '.xls', '.xlsx'] }),
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
    await page.getByRole('button', { name: 'Upload dataset', exact: true }).click();
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
    await expect(page.getByRole('button', { name: 'Upload dataset', exact: true })).toBeFocused();

    if (testInfo.project.name !== 'mobile-600x900') {
        return;
    }

    await page.setViewportSize({ width: 390, height: 844 });
    const mobileChooserPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Upload dataset', exact: true }).click();
    const mobileChooser = await mobileChooserPromise;
    await mobileChooser.setFiles({
        name: 'sample.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from('pressure,uptake,temperature\n1,0.1,298\n'),
    });

    const mobileDialog = page.getByRole('dialog', { name: 'Understand sample.csv' });
    await mobileDialog.getByRole('button', { name: 'Review mapping', exact: true }).click();

    const mappingGeometry = await mobileDialog.evaluate((element) => {
        const settings = element.querySelector<HTMLElement>('.settings-cards');
        const table = element.querySelector<HTMLElement>('.mapping-table');
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
    expect(mappingGeometry.tableWrapScrollWidth).toBeLessThanOrEqual(mappingGeometry.tableWrapClientWidth + 1);

    await page.setViewportSize({ width: 195, height: 422 });
    await expect(mobileDialog.getByRole('button', { name: 'Validate and preview series', exact: true })).toBeVisible();
    const narrowDocumentWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(narrowDocumentWidth).toBeLessThanOrEqual(196);
});
