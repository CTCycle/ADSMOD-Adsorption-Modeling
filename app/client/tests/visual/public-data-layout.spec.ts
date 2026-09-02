import { expect, test, type Page } from '@playwright/test';

const longMaterial = 'Hierarchical activated carbon framework with an intentionally long scientific sample designation 2026-09-A';
const longExternalId = 'NIST-ISODB-EXPERIMENT-IDENTIFIER-WITH-LONG-PROVENANCE-SUFFIX-0000000000001';

async function installApiMocks(page: Page): Promise<void> {
    await page.route('**/health/ready', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ service: 'backend', version: '3.0.0', state: 'ready' }) });
    });
    await page.route('**/api/v1/system/capabilities**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ version: '3.0.0', features: { datasets: true, nist: true, fitting: true, machine_learning: false, training: false, checkpoints: false } }) });
    });
    await page.route('**/api/v1/public-data/sources**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sources: [
            { key: 'nist', name: 'NIST/ARPA-E Adsorption Database', description: 'Adsorption experiments, guest species, and host materials.', capabilities: ['adsorption', 'materials', 'chemicals', 'references'], status: 'available', status_detail: null, homepage_url: 'https://adsorption.nist.gov/', license_name: null, license_url: null, terms_url: 'https://adsorption.nist.gov/', record_count: 3210, last_checked_at: '2026-09-02T12:00:00Z' },
            { key: 'pubchem', name: 'PubChem', description: 'Chemical identities, descriptors, synonyms, and molecular structures.', capabilities: ['chemicals', 'structures', 'references'], status: 'available', status_detail: null, homepage_url: 'https://pubchem.ncbi.nlm.nih.gov/', license_name: 'Public domain / source-attributed records', license_url: 'https://pubchem.ncbi.nlm.nih.gov/docs/downloads', terms_url: 'https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access', record_count: 142, last_checked_at: '2026-09-02T12:00:00Z' },
            { key: 'cod', name: 'Crystallography Open Database', description: 'Open crystal structures, CIF records, and publication metadata.', capabilities: ['materials', 'structures', 'references'], status: 'unavailable', status_detail: 'Validation fixture: provider temporarily unavailable.', homepage_url: 'https://www.crystallography.net/cod/', license_name: 'CC0 1.0', license_url: 'https://creativecommons.org/publicdomain/zero/1.0/', terms_url: 'https://wiki.crystallography.net/howtoobtaincod/', record_count: 38, last_checked_at: '2026-09-02T12:00:00Z' },
        ] }) });
    });
    await page.route('**/api/v1/public-data/adsorption/1', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 1, external_id: longExternalId, source: 'nist', source_url: 'https://adsorption.nist.gov/', material: longMaterial, adsorbates: ['carbon dioxide'], temperature_k: 298.15, pressure_min_pa: 1000, pressure_max_pa: 1000000, uptake_min_mol_kg: 0.02, uptake_max_mol_kg: 8.4, point_count: 4, reference: '10.1000/example', retrieved_at: '2026-09-02T12:00:00Z', pressure_basis: 'absolute', conditions: { method: 'volumetric' }, provenance: { normalized: true }, external_identifiers: [{ source: 'nist', external_id: longExternalId, source_url: 'https://adsorption.nist.gov/', retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], measurements: [
            { sequence_index: 0, adsorbate: 'carbon dioxide', pressure_original: 1, pressure_original_unit: 'kPa', pressure_pa: 1000, uptake_original: 0.02, uptake_original_unit: 'mol/kg', uptake_mol_kg: 0.02 },
            { sequence_index: 1, adsorbate: 'carbon dioxide', pressure_original: 10, pressure_original_unit: 'kPa', pressure_pa: 10000, uptake_original: 0.6, uptake_original_unit: 'mol/kg', uptake_mol_kg: 0.6 },
            { sequence_index: 2, adsorbate: 'carbon dioxide', pressure_original: 100, pressure_original_unit: 'kPa', pressure_pa: 100000, uptake_original: 3.2, uptake_original_unit: 'mol/kg', uptake_mol_kg: 3.2 },
            { sequence_index: 3, adsorbate: 'carbon dioxide', pressure_original: 1000, pressure_original_unit: 'kPa', pressure_pa: 1000000, uptake_original: 8.4, uptake_original_unit: 'mol/kg', uptake_mol_kg: 8.4 },
        ] }) });
    });
    await page.route(/\/api\/v1\/public-data\/adsorption(?:\?.*)?$/, async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 1, external_id: longExternalId, source: 'nist', source_url: 'https://adsorption.nist.gov/', material: longMaterial, adsorbates: ['carbon dioxide'], temperature_k: 298.15, pressure_min_pa: 1000, pressure_max_pa: 1000000, uptake_min_mol_kg: 0.02, uptake_max_mol_kg: 8.4, point_count: 120, reference: '10.1000/example', retrieved_at: '2026-09-02T12:00:00Z' }], pagination: { page: 1, page_size: 25, total: 101 } }) });
    });
    await page.route('**/api/v1/public-data/materials**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 4, name: longMaterial, formula: 'C120H24O8', molar_mass_g_mol: 1593.4, structure_count: 2, external_identifiers: [{ source: 'nist', external_id: 'material-123', source_url: null, retrieved_at: '2026-09-02T12:00:00Z', source_version: null }] }], pagination: { page: 1, page_size: 25, total: 50 } }) });
    });
    await page.route('**/api/v1/public-data/chemicals/7', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 7, name: 'carbon dioxide', preferred_name: 'carbon dioxide', formula: 'CO2', molecular_weight: 44.009, inchi: 'InChI=1S/CO2/c2-1-3', inchi_key: 'CURLTUGMZLYLDI-UHFFFAOYSA-N', connectivity_smiles: 'C(=O)=O', smiles: 'C(=O)=O', pubchem_cid: '280', synonyms: Array.from({ length: 20 }, (_, index) => `long scientific synonym ${index + 1}`), properties: [{ key: 'tpsa_angstrom2', value_number: 34.1, value_text: null, unit: 'Å²', source: 'pubchem' }], external_identifiers: [{ source: 'pubchem', external_id: '280', source_url: 'https://pubchem.ncbi.nlm.nih.gov/compound/280', retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], structure_2d_url: null, conformer_3d_url: 'https://example.test/co2.sdf', retrieved_at: '2026-09-02T12:00:00Z' }) });
    });
    await page.route(/\/api\/v1\/public-data\/chemicals(?:\?.*)?$/, async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 7, name: 'carbon dioxide', preferred_name: 'carbon dioxide', formula: 'CO2', molecular_weight: 44.009, inchi: 'InChI=1S/CO2/c2-1-3', inchi_key: 'CURLTUGMZLYLDI-UHFFFAOYSA-N', connectivity_smiles: 'C(=O)=O', smiles: 'C(=O)=O', pubchem_cid: '280', synonyms: [], properties: [], external_identifiers: [{ source: 'pubchem', external_id: '280', source_url: 'https://pubchem.ncbi.nlm.nih.gov/compound/280', retrieved_at: '2026-09-02T12:00:00Z', source_version: null }], structure_2d_url: null, conformer_3d_url: 'https://example.test/co2.sdf', retrieved_at: '2026-09-02T12:00:00Z' }], pagination: { page: 1, page_size: 25, total: 80 } }) });
    });
    await page.route('**/api/v1/public-data/structures**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [{ id: 9, source: 'cod', external_id: '9012345', source_url: 'https://www.crystallography.net/cod/9012345.html', material_id: null, material_name: null, name: longMaterial, formula: 'C120H24O8', format: 'cif', content_sha256: 'a'.repeat(64), space_group: 'P 21/c', space_group_number: 14, cell_a_angstrom: 12.3, cell_b_angstrom: 18.4, cell_c_angstrom: 25.1, cell_alpha_deg: 90, cell_beta_deg: 102.3, cell_gamma_deg: 90, cell_volume_angstrom3: 5500.2, has_coordinates: true, atom_count: 144, doi: '10.1000/example', retrieved_at: '2026-09-02T12:00:00Z', atoms: [] }], pagination: { page: 1, page_size: 25, total: 38 } }) });
    });
    await page.route('**/api/v1/nist/categories**', async (route) => {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ category: 'experiments', local_count: 100, available_count: 1000, last_update: null, server_ok: true, server_checked_at: null, supports_enrichment: false }, { category: 'guest', local_count: 20, available_count: 100, last_update: null, server_ok: true, server_checked_at: null, supports_enrichment: true }, { category: 'host', local_count: 30, available_count: 120, last_update: null, server_ok: true, server_checked_at: null, supports_enrichment: true }]) });
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
}

test('public data workspace remains contained across dense scientific views', async ({ page }) => {
    await installApiMocks(page);
    await page.goto('/public-data/overview');
    await expect(page.getByRole('heading', { name: 'Integrated sources' })).toBeVisible();
    await assertContainedLayout(page);

    for (const view of ['adsorption', 'materials', 'chemicals', 'structures', 'sources']) {
        await page.goto(`/public-data/${view}`);
        await expect(page.locator('.public-workspace')).toBeVisible();
        await assertContainedLayout(page);
    }

    await page.goto('/public-data/adsorption');
    await page.getByRole('button', { name: 'Inspect' }).click();
    await expect(page.getByRole('img', { name: 'Pressure versus uptake isotherm plot' })).toBeVisible();
    await assertContainedLayout(page);

    await page.goto('/public-data/chemicals');
    await page.getByRole('button', { name: 'Inspect' }).click();
    await expect(page.getByRole('heading', { name: 'carbon dioxide' })).toBeVisible();
    await assertContainedLayout(page);
});
