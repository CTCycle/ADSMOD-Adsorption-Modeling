// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { expect, test, type Page, type TestInfo } from '@playwright/test';

const longMaterial = 'Hierarchical activated carbon framework with an intentionally long scientific sample designation 2026-09-A';
const longExternalId = 'NIST-ISODB-EXPERIMENT-IDENTIFIER-WITH-LONG-PROVENANCE-SUFFIX-0000000000001';

async function installApiMocks(page: Page): Promise<void> {
    await page.route('**/health/ready', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ service: 'backend', version: '3.0.0', state: 'ready' }) });
    });
    await page.route('**/api/v1/system/capabilities**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ version: '3.0.0', features: { datasets: true, nist: true, fitting: true, machine_learning: false, training: false, checkpoints: false } }) });
    });
    await page.route('**/api/v1/public-data/sources', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sources: [
            { key: 'nist', name: 'NIST/ARPA-E Adsorption Database', description: 'Adsorption experiments, guest species, and host materials.', capabilities: ['adsorption', 'materials', 'chemicals', 'references'], health_status: 'available', health_detail: null, health_fresh: true, last_health_checked_at: '2026-09-02T12:00:00Z', last_successful_sync_at: null, homepage_url: 'https://adsorption.nist.gov/', license_name: null, license_url: null, terms_url: 'https://adsorption.nist.gov/', record_count: 3210 },
            { key: 'pubchem', name: 'PubChem', description: 'Chemical identities, descriptors, synonyms, and molecular structures.', capabilities: ['chemicals', 'structures', 'references'], health_status: 'available', health_detail: null, health_fresh: true, last_health_checked_at: '2026-09-02T12:00:00Z', last_successful_sync_at: null, homepage_url: 'https://pubchem.ncbi.nlm.nih.gov/', license_name: 'Public domain / source-attributed records', license_url: 'https://pubchem.ncbi.nlm.nih.gov/docs/downloads', terms_url: 'https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access', record_count: 142 },
            { key: 'cod', name: 'Crystallography Open Database', description: 'Open crystal structures, CIF records, and publication metadata.', capabilities: ['structures', 'references'], health_status: 'unavailable', health_detail: 'Validation fixture: provider temporarily unavailable.', health_fresh: true, last_health_checked_at: '2026-09-02T12:00:00Z', last_successful_sync_at: null, homepage_url: 'https://www.crystallography.net/cod/', license_name: 'CC0 1.0', license_url: 'https://creativecommons.org/publicdomain/zero/1.0/', terms_url: 'https://wiki.crystallography.net/howtoobtaincod/', record_count: 38 },
        ] }) });
    });
    await page.route('**/api/v1/public-data/sync-runs**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ runs: [], total: 0 }) });
    });
    await page.route('**/api/v1/public-data/overview', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ adsorption_records: 101, materials: 50, chemicals: 80, structures: 38, active_sync_runs: 0, failed_sync_runs: 0, latest_successful_sync_at: null }) });
    });
    await page.route('**/api/v1/public-data/adsorption/1', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 1, external_id: longExternalId, source: 'nist', source_url: 'https://adsorption.nist.gov/', material_id: 4, material: longMaterial, adsorbates: [{ id: 7, name: 'carbon dioxide' }], temperature_k: 298.15, pressure_min_pa: 1000, pressure_max_pa: 1000000, uptake_min_mol_kg: 0.02, uptake_max_mol_kg: 8.4, point_count: 4, doi: '10.1000/example', reference_title: null, retrieved_at: '2026-09-02T12:00:00Z', pressure_basis: 'absolute', conditions: { method: 'volumetric' }, provenance: { normalized: true }, external_identifiers: [{ source: 'nist', external_id: longExternalId, source_url: 'https://adsorption.nist.gov/', retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], measurements: [
            { sequence_index: 0, adsorbate: 'carbon dioxide', pressure_original: 1, pressure_original_unit: 'kPa', pressure_pa: 1000, uptake_original: 0.02, uptake_original_unit: 'mol/kg', uptake_mol_kg: 0.02 },
            { sequence_index: 1, adsorbate: 'carbon dioxide', pressure_original: 10, pressure_original_unit: 'kPa', pressure_pa: 10000, uptake_original: 0.6, uptake_original_unit: 'mol/kg', uptake_mol_kg: 0.6 },
            { sequence_index: 2, adsorbate: 'carbon dioxide', pressure_original: 100, pressure_original_unit: 'kPa', pressure_pa: 100000, uptake_original: 3.2, uptake_original_unit: 'mol/kg', uptake_mol_kg: 3.2 },
            { sequence_index: 3, adsorbate: 'carbon dioxide', pressure_original: 1000, pressure_original_unit: 'kPa', pressure_pa: 1000000, uptake_original: 8.4, uptake_original_unit: 'mol/kg', uptake_mol_kg: 8.4 },
        ] }) });
    });
    await page.route(/\/api\/v1\/public-data\/adsorption(?:\?.*)?$/, async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 1, external_id: longExternalId, source: 'nist', source_url: 'https://adsorption.nist.gov/', material_id: 4, material: longMaterial, adsorbates: [{ id: 7, name: 'carbon dioxide' }], temperature_k: 298.15, pressure_min_pa: 1000, pressure_max_pa: 1000000, uptake_min_mol_kg: 0.02, uptake_max_mol_kg: 8.4, point_count: 120, doi: '10.1000/example', reference_title: null, retrieved_at: '2026-09-02T12:00:00Z' }], pagination: { page: 1, page_size: 25, total: 101 } }) });
    });
    await page.route('**/api/v1/public-data/materials**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 4, name: longMaterial, formula: 'C120H24O8', molar_mass_g_mol: 1593.4, structure_count: 1, isotherm_count: 1, external_identifiers: [{ source: 'nist', external_id: 'material-123', source_url: null, retrieved_at: '2026-09-02T12:00:00Z', source_version: null }] }], pagination: { page: 1, page_size: 25, total: 50 } }) });
    });
    await page.route('**/api/v1/public-data/chemicals/7', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 7, name: 'carbon dioxide', preferred_name: 'carbon dioxide', formula: 'CO2', molecular_weight: 44.009, inchi: 'InChI=1S/CO2/c2-1-3', inchi_key: 'CURLTUGMZLYLDI-UHFFFAOYSA-N', connectivity_smiles: 'C(=O)=O', smiles: 'C(=O)=O', pubchem_cid: '280', synonyms: Array.from({ length: 20 }, (_, index) => `long scientific synonym ${index + 1}`), properties: [{ key: 'tpsa_angstrom2', value_number: 34.1, value_text: null, unit: 'Å²', source: 'pubchem' }], external_identifiers: [{ source: 'pubchem', external_id: '280', source_url: 'https://pubchem.ncbi.nlm.nih.gov/compound/280', retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], structure_2d_url: null, conformer_3d_url: 'https://example.test/co2.sdf', isotherm_count: 1, retrieved_at: '2026-09-02T12:00:00Z' }) });
    });
    await page.route(/\/api\/v1\/public-data\/chemicals(?:\?.*)?$/, async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 7, name: 'carbon dioxide', preferred_name: 'carbon dioxide', formula: 'CO2', molecular_weight: 44.009, inchi: 'InChI=1S/CO2/c2-1-3', inchi_key: 'CURLTUGMZLYLDI-UHFFFAOYSA-N', connectivity_smiles: 'C(=O)=O', smiles: 'C(=O)=O', pubchem_cid: '280', synonyms: [], properties: [], external_identifiers: [{ source: 'pubchem', external_id: '280', source_url: 'https://pubchem.ncbi.nlm.nih.gov/compound/280', retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], structure_2d_url: null, conformer_3d_url: 'https://example.test/co2.sdf', isotherm_count: 1, retrieved_at: '2026-09-02T12:00:00Z' }], pagination: { page: 1, page_size: 25, total: 80 } }) });
    });
    await page.route('**/api/v1/public-data/structures**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 9, source: 'cod', external_id: '9012345', source_url: 'https://www.crystallography.net/cod/9012345.html', material_id: null, material_name: null, name: longMaterial, formula: 'C120H24O8', format: 'cif', content_sha256: 'a'.repeat(64), space_group: 'P 21/c', space_group_number: 14, cell_a_angstrom: 12.3, cell_b_angstrom: 18.4, cell_c_angstrom: 25.1, cell_alpha_deg: 90, cell_beta_deg: 102.3, cell_gamma_deg: 90, cell_volume_angstrom3: 5500.2, has_coordinates: true, atom_count: 144, doi: '10.1000/example', reference_title: null, retrieved_at: '2026-09-02T12:00:00Z', atoms: [] }], pagination: { page: 1, page_size: 25, total: 38 } }) });
    });
    await page.route('**/api/v1/public-data/sources/nist/collections', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ collections: [
            { category: 'experiments', local_count: 100, available_count: 1000, indexed_at: '2026-10-19T10:00:00Z' },
            { category: 'guest', local_count: 20, available_count: null, indexed_at: null },
            { category: 'host', local_count: 30, available_count: 120, indexed_at: '2026-10-19T10:00:00Z' },
        ] }) });
    });
    await page.route('**/api/v1/public-data/materials/4', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 4, name: longMaterial, formula: 'C120H24O8', molar_mass_g_mol: 1593.4, structure_count: 1, isotherm_count: 1, external_identifiers: [{ source: 'nist', external_id: 'material-123', source_url: null, retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], structures: [{ id: 9, source: 'cod', external_id: '9012345', source_url: null, material_id: 4, material_name: longMaterial, name: null, formula: 'C120H24O8', format: 'cif', content_sha256: 'a'.repeat(64), space_group: 'P 21/c', space_group_number: 14, cell_a_angstrom: null, cell_b_angstrom: null, cell_c_angstrom: null, cell_alpha_deg: null, cell_beta_deg: null, cell_gamma_deg: null, cell_volume_angstrom3: null, has_coordinates: true, atom_count: 144, doi: null, reference_title: null, retrieved_at: null, atoms: [] }] }) });
    });
}

async function assertContainedLayout(page: Page): Promise<void> {
    const metrics = await page.evaluate(() => ({
        viewport: document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
        bodyWidth: document.body.scrollWidth,
    }));
    expect(metrics.documentWidth).toBeLessThanOrEqual(metrics.viewport + 1);
    expect(metrics.bodyWidth).toBeLessThanOrEqual(metrics.viewport + 1);
    await expect(page.locator('.public-workspace')).toBeVisible();
    const frames = await page.locator('.data-table-frame').evaluateAll((elements) => elements.map((frame) => {
        const parent = frame.closest('.public-section, .detail-panel') ?? frame.parentElement;
        if (!parent) throw new Error('Public data table has no layout parent');
        const frameRect = frame.getBoundingClientRect();
        const parentRect = parent.getBoundingClientRect();
        return {
            frameLeft: frameRect.left,
            frameRight: frameRect.right,
            parentLeft: parentRect.left,
            parentRight: parentRect.right,
            clientWidth: frame.clientWidth,
            scrollWidth: frame.scrollWidth,
            overflowX: getComputedStyle(frame).overflowX,
        };
    }));
    for (const frame of frames) {
        expect(frame.frameLeft).toBeGreaterThanOrEqual(frame.parentLeft - 1);
        expect(frame.frameRight).toBeLessThanOrEqual(frame.parentRight + 1);
        if (frame.scrollWidth > frame.clientWidth) {
            expect(['auto', 'scroll']).toContain(frame.overflowX);
        }
    }
}

test('public data workspace remains contained across dense scientific views', async ({ page }, testInfo: TestInfo) => {
    await installApiMocks(page);
    let useOverviewScreenshotFixture = true;
    await page.route('**/api/v1/public-data/sources', async (route) => {
        if (!useOverviewScreenshotFixture) {
            await route.fallback();
            return;
        }
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sources: [
            { key: 'cod', name: 'Crystallography Open Database', description: 'Open crystallographic structures with unit-cell metadata, atomic coordinates, CIF files, and publication references.', capabilities: ['structures', 'references'], health_status: 'unknown', health_detail: null, health_fresh: false, last_health_checked_at: null, last_successful_sync_at: null, homepage_url: 'https://www.crystallography.net/cod/', license_name: 'CC0 1.0', license_url: 'https://creativecommons.org/publicdomain/zero/1.0/', terms_url: 'https://wiki.crystallography.net/howtoobtaincod/', record_count: 0 },
            { key: 'nist', name: 'NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials', description: 'Public adsorption experiments plus guest-species and host-material records from the ADSMOD NIST integration.', capabilities: ['adsorption', 'materials', 'chemicals', 'references'], health_status: 'unknown', health_detail: null, health_fresh: false, last_health_checked_at: null, last_successful_sync_at: null, homepage_url: 'https://adsorption.nist.gov/', license_name: null, license_url: null, terms_url: 'https://adsorption.nist.gov/', record_count: 0 },
            { key: 'pubchem', name: 'PubChem', description: 'NIH public chemical information, identifiers, molecular descriptors, and 2D/3D structure records.', capabilities: ['chemicals', 'structures', 'references'], health_status: 'unknown', health_detail: null, health_fresh: false, last_health_checked_at: null, last_successful_sync_at: null, homepage_url: 'https://pubchem.ncbi.nlm.nih.gov/', license_name: 'Public domain / source-attributed records', license_url: 'https://pubchem.ncbi.nlm.nih.gov/docs/downloads', terms_url: 'https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access', record_count: 0 },
        ] }) });
    });
    for (const resource of ['adsorption', 'materials', 'chemicals', 'structures']) {
        await page.route(`**/api/v1/public-data/${resource}**`, async (route) => {
            if (!useOverviewScreenshotFixture) {
                await route.fallback();
                return;
            }
            await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [], pagination: { page: 1, page_size: 1, total: 0 } }) });
        });
    }
    await page.goto('/public-data/overview');
    await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveText(/Public Data/);
    await expect(page.getByRole('heading', { name: 'Integrated sources' })).toBeVisible();
    await expect(page.locator('.public-workspace .section-heading').last()).toHaveCSS('flex-direction', 'row');
    await expect(page.locator('.public-workspace .section-heading').last()).toHaveCSS('text-align', 'left');
    await assertContainedLayout(page);
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('public-data-overview-after.jpg') });
    }
    useOverviewScreenshotFixture = false;

    for (const view of ['acquire', 'adsorption', 'materials', 'chemicals', 'structures', 'sources']) {
        await page.goto(`/public-data/${view}`);
        await expect(page.locator('.public-workspace')).toBeVisible();
        await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveText(/Public Data/);
        await assertContainedLayout(page);
    }

    await page.getByRole('link', { name: 'Fitting', exact: true }).hover();
    await expect(page.locator('.console-nav-item[aria-current="page"]')).toHaveText(/Public Data/);
    await expect(page.getByRole('link', { name: 'Fitting', exact: true })).not.toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('link', { name: 'Fitting', exact: true })).toHaveCSS('box-shadow', 'none');
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('public-nav-unselected-hover.png') });
    }

    await page.goto('/public-data/materials');
    const controlHeight = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--control-height').trim());
    await expect(page.locator('.filter-bar input, .filter-bar select').first()).toHaveCSS('height', controlHeight);

    await page.goto('/public-data/adsorption');
    await page.getByRole('button', { name: 'Inspect' }).click();
    await expect(page.getByRole('img', { name: 'Pressure versus uptake isotherm plot' })).toBeVisible();
    await assertContainedLayout(page);

    await page.goto('/public-data/chemicals');
    await page.getByRole('button', { name: 'Inspect' }).click();
    await expect(page.getByRole('heading', { name: 'carbon dioxide' })).toBeVisible();
    await assertContainedLayout(page);
});

test('Overview renders before one local summary response and skips provider health probes', async ({ page }) => {
    await installApiMocks(page);

    const sourceRequests: string[] = [];
    const providerRequests: string[] = [];
    let overviewRequests = 0;
    let releaseOverview!: () => void;
    let markOverviewStarted!: () => void;
    const overviewGate = new Promise<void>((resolve) => { releaseOverview = resolve; });
    const overviewStarted = new Promise<void>((resolve) => { markOverviewStarted = resolve; });

    await page.route('**/api/v1/public-data/sources', async (route) => {
        sourceRequests.push(route.request().url());
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sources: [] }) });
    });
    await page.route('**/api/v1/public-data/overview', async (route) => {
        overviewRequests += 1;
        markOverviewStarted();
        await overviewGate;
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ adsorption_records: 40, materials: 12, chemicals: 8, structures: 3, active_sync_runs: 0, failed_sync_runs: 0, latest_successful_sync_at: null }) });
    });
    page.on('request', (request) => {
        if (/adsorption\.nist\.gov|pubchem\.ncbi\.nlm\.nih\.gov|crystallography\.net/i.test(request.url())) {
            providerRequests.push(request.url());
        }
    });

    await page.goto('/public-data/overview');
    await overviewStarted;
    await expect(page.getByRole('heading', { name: 'Integrated sources' })).toBeVisible();
    await expect(page.locator('.overview-totals strong').first()).toHaveText('—');
    expect(overviewRequests).toBe(1);
    expect(sourceRequests).toHaveLength(1);
    expect(new URL(sourceRequests[0]).pathname).toBe('/api/v1/public-data/sources');
    expect(providerRequests).toEqual([]);

    releaseOverview();
    await expect(page.locator('.overview-totals strong').nth(0)).toHaveText('40');
    await expect(page.locator('.overview-totals strong').nth(1)).toHaveText('12');
    await expect(page.locator('.overview-totals strong').nth(2)).toHaveText('8');
    await expect(page.locator('.overview-totals strong').nth(3)).toHaveText('3');
});

test('Sources shows cached status and refreshes it through a background health run', async ({ page }) => {
    await installApiMocks(page);

    let releaseHealth!: () => void;
    let markHealthStarted!: () => void;
    let healthResolved = false;
    const healthGate = new Promise<void>((resolve) => { releaseHealth = resolve; });
    const healthStarted = new Promise<void>((resolve) => { markHealthStarted = resolve; });

    await page.route('**/api/v1/public-data/sources', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sources: [
            { key: 'nist', name: 'NIST test provider', description: 'Local provider registration.', capabilities: ['adsorption'], health_status: healthResolved ? 'available' : 'unknown', health_detail: null, health_fresh: healthResolved, homepage_url: 'https://adsorption.nist.gov/', license_name: null, license_url: null, terms_url: null, record_count: 4, last_health_checked_at: healthResolved ? '2026-10-19T10:00:00Z' : null, last_successful_sync_at: null },
        ] }) });
    });
    await page.route('**/api/v1/public-data/sources/nist/health-check', async (route) => {
        await route.fulfill({ status: 202, contentType: 'application/json', body: JSON.stringify({ id: 'health-1', source_key: 'nist', operation: 'health_check', parameters: {}, status: 'queued', progress: 0, processed_count: 0, imported_count: 0, skipped_count: 0, failed_count: 0, checkpoint: {}, cancel_requested: false, error_detail: null, created_at: '2026-10-19T10:00:00Z', started_at: null, finished_at: null }) });
    });
    await page.route('**/api/v1/public-data/sync-runs/health-1', async (route) => {
        markHealthStarted();
        await healthGate;
        healthResolved = true;
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 'health-1', source_key: 'nist', operation: 'health_check', parameters: {}, status: 'completed', progress: 100, processed_count: 1, imported_count: 0, skipped_count: 0, failed_count: 0, checkpoint: {}, cancel_requested: false, error_detail: null, created_at: '2026-10-19T10:00:00Z', started_at: '2026-10-19T10:00:00Z', finished_at: '2026-10-19T10:00:01Z' }) });
    });
    await page.route('**/api/v1/public-data/sync-runs?limit=20&offset=0', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ runs: [], total: 0 }) });
    });

    await page.goto('/public-data/sources');
    await expect(page.getByText('NIST test provider')).toBeVisible();
    await expect(page.getByText('4 stored source records')).toBeVisible();
    await expect(page.getByText('Not checked')).toBeVisible();

    await page.getByRole('button', { name: 'Refresh status' }).click();
    await healthStarted;
    await expect(page.getByText('Not checked')).toBeVisible();
    releaseHealth();
    await expect(page.getByText('available', { exact: true })).toBeVisible();
});

test('NIST collection fetching in Acquire tracks durable progress and cooperative cancellation', async ({ page }, testInfo: TestInfo) => {
    await installApiMocks(page);

    let started = false;
    let cancelRequested = false;
    let requestBody: Record<string, unknown> | null = null;
    const nistRun = (status: string) => ({
        id: 'nist-run-1', source_key: 'nist', operation: 'nist_category',
        parameters: { category: 'experiments', fraction: 0.1 }, status,
        progress: 30, processed_count: 15, imported_count: 12, skipped_count: 2, failed_count: 1,
        checkpoint: { pending_count: 5 }, cancel_requested: cancelRequested, error_detail: null,
        created_at: '2026-10-19T10:00:00Z', started_at: '2026-10-19T10:00:01Z',
        finished_at: status === 'cancelled' ? '2026-10-19T10:00:02Z' : null,
    });
    await page.route('**/api/v1/public-data/sync-runs', async (route) => {
        if (route.request().method() !== 'POST') {
            await route.fallback();
            return;
        }
        requestBody = route.request().postDataJSON() as Record<string, unknown>;
        started = true;
        await route.fulfill({ status: 202, contentType: 'application/json', body: JSON.stringify(nistRun('queued')) });
    });
    await page.route(/\/api\/v1\/public-data\/sync-runs\?/, async (route) => {
        const scope = new URL(route.request().url()).searchParams.get('scope');
        const runs = scope === 'active'
            ? (started && !cancelRequested ? [nistRun('running')] : [])
            : (cancelRequested ? [nistRun('cancelled')] : []);
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ runs, total: runs.length }) });
    });
    await page.route('**/api/v1/public-data/sync-runs/nist-run-1/cancel', async (route) => {
        cancelRequested = true;
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(nistRun('cancelled')) });
    });
    await page.route('**/api/v1/public-data/sync-runs/nist-run-1', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(nistRun(cancelRequested ? 'cancelled' : 'running')) });
    });
    await page.route('**/api/v1/public-data/sync-runs/nist-run-1/failures**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
            total: 1,
            failures: [{ id: 1, source_key: 'nist', external_id: 'record-unsupported-1', reason: 'Unsupported uptake unit: foo.', retry_eligible_at: null, created_at: '2026-10-19T10:00:02Z' }],
        }) });
    });

    await page.addInitScript(() => window.localStorage.setItem('adsmod.nist.fraction.experiments', '0.1'));
    await page.goto('/public-data/acquire');
    const isotherms = page.locator('.collection-row', { hasText: 'Adsorption isotherms' });
    await expect(isotherms).toContainText(/100 \/ 1[,.\s]?000/);
    await assertContainedLayout(page);
    const assertNistTableGeometry = async () => {
        const geometry = await page.locator('.collection-table-frame').evaluate((frame) => {
            const table = frame.querySelector<HTMLTableElement>('.collection-table');
            if (!table) throw new Error('NIST collection table is unavailable');
            const headers = Array.from(table.querySelectorAll('thead th'));
            const cells = Array.from(table.querySelectorAll('tbody tr:first-child td'));
            return {
                viewportWidth: document.documentElement.clientWidth,
                documentWidth: document.documentElement.scrollWidth,
                frameWidth: frame.clientWidth,
                frameScrollWidth: frame.scrollWidth,
                tableWidth: table.getBoundingClientRect().width,
                columns: headers.map((header, index) => ({
                    headerLeft: header.getBoundingClientRect().left,
                    headerWidth: header.getBoundingClientRect().width,
                    cellLeft: cells[index]?.getBoundingClientRect().left ?? -1,
                    cellWidth: cells[index]?.getBoundingClientRect().width ?? -1,
                })),
            };
        });
        expect(geometry.documentWidth).toBeLessThanOrEqual(geometry.viewportWidth + 1);
        expect(geometry.columns).toHaveLength(4);
        for (const column of geometry.columns) {
            expect(Math.abs(column.headerLeft - column.cellLeft)).toBeLessThanOrEqual(1);
            expect(Math.abs(column.headerWidth - column.cellWidth)).toBeLessThanOrEqual(1);
        }
        if (geometry.frameWidth >= 780) {
            expect(geometry.tableWidth).toBeGreaterThanOrEqual(geometry.frameWidth - 1);
        } else {
            expect(geometry.frameScrollWidth).toBeGreaterThan(geometry.frameWidth);
        }
        return geometry;
    };
    await assertNistTableGeometry();
    if (testInfo.project.name === 'viewport-1440x920') {
        await page.screenshot({ path: testInfo.outputPath('nist-collections-populated-desktop.png') });
    }
    if (testInfo.project.name === 'mobile-600x900') {
        await page.setViewportSize({ width: 390, height: 844 });
        const narrowGeometry = await assertNistTableGeometry();
        expect(narrowGeometry.frameWidth).toBeLessThan(narrowGeometry.frameScrollWidth);
        await page.screenshot({ path: testInfo.outputPath('nist-collections-narrow-390.png') });
    }
    await isotherms.getByRole('button', { name: 'Fetch' }).click();
    await expect.poll(() => requestBody).toEqual({
        source_key: 'nist', operation: 'nist_category', category: 'experiments', fraction: 0.1,
    });
    const cancelButton = page.locator('#acquire-runs').getByRole('button', { name: 'Cancel' });
    await expect(cancelButton).toBeVisible();
    await cancelButton.click();
    await expect(page.locator('#acquire-runs')).toContainText('cancelled');
    await page.getByRole('button', { name: 'Details', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Acquisition details' })).toContainText('Unsupported uptake unit: foo.');
});

test('COD search in Acquire imports a bounded selection and links to the stored structures', async ({ page }) => {
    await installApiMocks(page);
    let requestBody: Record<string, unknown> | null = null;
    await page.route('**/api/v1/public-data/structures/search**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [
            { cod_id: '9012345', name: 'Carbon dioxide phase A', formula: 'CO2', space_group: 'P 21/c', space_group_number: 14, cell_volume_angstrom3: 200.0, doi: '10.1000/a', year: 2006, has_coordinates: true, source_url: 'https://www.crystallography.net/cod/9012345.html', cif_url: 'https://www.crystallography.net/cod/9012345.cif' },
            { cod_id: '9012346', name: 'Carbon dioxide phase B', formula: 'CO2', space_group: 'P 1', space_group_number: 1, cell_volume_angstrom3: 210.0, doi: null, year: null, has_coordinates: true, source_url: 'https://www.crystallography.net/cod/9012346.html', cif_url: 'https://www.crystallography.net/cod/9012346.cif' },
        ] }) });
    });
    const completedRun = () => ({
        id: 'cod-batch-1', source_key: 'cod', operation: 'cod_import', parameters: requestBody,
        status: 'completed', progress: 100, processed_count: 2, imported_count: 2,
        skipped_count: 0, failed_count: 0, checkpoint: { entity_ids: [1, 2], pending_count: 0 },
        cancel_requested: false, error_detail: null, created_at: '2026-10-19T10:00:00Z',
        started_at: '2026-10-19T10:00:01Z', finished_at: '2026-10-19T10:00:02Z',
    });
    await page.route('**/api/v1/public-data/sync-runs', async (route) => {
        if (route.request().method() !== 'POST') {
            await route.fallback();
            return;
        }
        requestBody = route.request().postDataJSON() as Record<string, unknown>;
        await route.fulfill({ status: 202, contentType: 'application/json', body: JSON.stringify(completedRun()) });
    });

    await page.goto('/public-data/acquire?panel=cod');
    await page.locator('#acquire-cod-query').fill('carbon dioxide');
    await page.locator('#acquire-cod').getByRole('button', { name: 'Search' }).click();
    await expect(page.getByText('Carbon dioxide phase A')).toBeVisible();
    await assertContainedLayout(page);
    await page.getByRole('checkbox', { name: 'Select all results' }).check();
    await page.getByRole('button', { name: 'Import selected (2)' }).click();

    await expect.poll(() => requestBody).toEqual({
        source_key: 'cod',
        operation: 'cod_import',
        cod_ids: ['9012345', '9012346'],
    });
    await expect(page.locator('#acquire-runs').getByRole('link', { name: /View structures/ })).toHaveAttribute('href', '/public-data/structures?source=cod');
    await assertContainedLayout(page);
});

test('stored records cross-link isotherms, materials, chemicals, and structures', async ({ page }) => {
    await installApiMocks(page);

    await page.goto('/public-data/adsorption?id=1');
    await expect(page.getByRole('img', { name: 'Pressure versus uptake isotherm plot' })).toBeVisible();
    await page.getByRole('complementary', { name: 'Adsorption record details' }).getByRole('link', { name: /^Material:/ }).click();
    await expect(page).toHaveURL(/\/public-data\/materials\?id=4$/);
    const material = page.getByRole('complementary', { name: 'Material details' });
    await expect(material.getByRole('link', { name: 'View 1 isotherms' })).toHaveAttribute('href', '/public-data/adsorption?material_id=4');
    await expect(material.getByRole('link', { name: 'Find COD structures' })).toHaveAttribute('href', /panel=cod.*formula=C120H24O8/);
    await material.getByRole('link', { name: 'Open' }).first().click();
    await expect(page).toHaveURL(/\/public-data\/structures\?id=9$/);
    await expect(page.getByRole('complementary', { name: 'Structure details' })).toBeVisible();
    await assertContainedLayout(page);
});

test('late Overview responses do not overwrite the selected view', async ({ page }) => {
    await installApiMocks(page);

    let releaseOverview!: () => void;
    let markOverviewStarted!: () => void;
    const overviewGate = new Promise<void>((resolve) => { releaseOverview = resolve; });
    const overviewStarted = new Promise<void>((resolve) => { markOverviewStarted = resolve; });
    await page.route('**/api/v1/public-data/overview', async (route) => {
        markOverviewStarted();
        await overviewGate;
        await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Delayed Overview failure' }) });
    });

    await page.goto('/public-data/overview');
    await overviewStarted;
    await page.getByRole('link', { name: 'Materials', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Materials' })).toBeVisible();
    releaseOverview();
    await expect(page.getByRole('alert')).toHaveCount(0);
});
