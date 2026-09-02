export type PublicDataView = 'overview' | 'adsorption' | 'materials' | 'chemicals' | 'structures' | 'sources';
export type SourceStatus = 'available' | 'degraded' | 'unavailable' | 'unknown';

export interface Pagination {
    page: number;
    page_size: number;
    total: number;
}

export interface ExternalIdentifierView {
    source: string;
    external_id: string;
    source_url: string | null;
    retrieved_at: string | null;
    source_version: string | null;
}

export interface PublicSourceSummary {
    key: string;
    name: string;
    description: string;
    capabilities: string[];
    status: SourceStatus;
    status_detail: string | null;
    homepage_url: string;
    license_name: string | null;
    license_url: string | null;
    terms_url: string | null;
    record_count: number;
    last_checked_at: string | null;
}

export interface PublicSourceListResponse {
    sources: PublicSourceSummary[];
}

export interface AdsorptionRecordView {
    id: number;
    external_id: string;
    source: string;
    source_url: string | null;
    material: string;
    adsorbates: string[];
    temperature_k: number;
    pressure_min_pa: number | null;
    pressure_max_pa: number | null;
    uptake_min_mol_kg: number | null;
    uptake_max_mol_kg: number | null;
    point_count: number;
    reference: string | null;
    retrieved_at: string | null;
}

export interface MeasurementView {
    sequence_index: number;
    adsorbate: string;
    pressure_original: number;
    pressure_original_unit: string;
    pressure_pa: number;
    uptake_original: number;
    uptake_original_unit: string;
    uptake_mol_kg: number;
}

export interface AdsorptionDetailResponse extends AdsorptionRecordView {
    pressure_basis: string;
    conditions: Record<string, unknown>;
    provenance: Record<string, unknown>;
    measurements: MeasurementView[];
    external_identifiers: ExternalIdentifierView[];
}

export interface AdsorptionPageResponse {
    items: AdsorptionRecordView[];
    pagination: Pagination;
}

export interface MaterialRecordView {
    id: number;
    name: string;
    formula: string | null;
    molar_mass_g_mol: number | null;
    structure_count: number;
    external_identifiers: ExternalIdentifierView[];
}

export interface MaterialPageResponse {
    items: MaterialRecordView[];
    pagination: Pagination;
}

export interface ChemicalPropertyView {
    key: string;
    value_number: number | null;
    value_text: string | null;
    unit: string | null;
    source: string;
}

export interface ChemicalRecordView {
    id: number;
    name: string;
    preferred_name: string | null;
    formula: string | null;
    molecular_weight: number | null;
    inchi: string | null;
    inchi_key: string | null;
    connectivity_smiles: string | null;
    smiles: string | null;
    pubchem_cid: string | null;
    synonyms: string[];
    properties: ChemicalPropertyView[];
    external_identifiers: ExternalIdentifierView[];
    structure_2d_url: string | null;
    conformer_3d_url: string | null;
    retrieved_at: string | null;
}

export interface ChemicalPageResponse {
    items: ChemicalRecordView[];
    pagination: Pagination;
}

export interface CODSearchResult {
    cod_id: string;
    name: string | null;
    formula: string | null;
    space_group: string | null;
    space_group_number: number | null;
    cell_a_angstrom: number | null;
    cell_b_angstrom: number | null;
    cell_c_angstrom: number | null;
    cell_alpha_deg: number | null;
    cell_beta_deg: number | null;
    cell_gamma_deg: number | null;
    cell_volume_angstrom3: number | null;
    doi: string | null;
    year: number | null;
    has_coordinates: boolean;
    source_url: string;
    cif_url: string;
}

export interface CODSearchResponse {
    items: CODSearchResult[];
}

export interface StructureAtomView {
    sequence_index: number;
    label: string;
    element: string;
    fractional_x: number;
    fractional_y: number;
    fractional_z: number;
    occupancy: number | null;
}

export interface StructureRecordView {
    id: number;
    source: string;
    external_id: string;
    source_url: string | null;
    material_id: number | null;
    material_name: string | null;
    name: string | null;
    formula: string | null;
    format: string;
    content_sha256: string;
    space_group: string | null;
    space_group_number: number | null;
    cell_a_angstrom: number | null;
    cell_b_angstrom: number | null;
    cell_c_angstrom: number | null;
    cell_alpha_deg: number | null;
    cell_beta_deg: number | null;
    cell_gamma_deg: number | null;
    cell_volume_angstrom3: number | null;
    has_coordinates: boolean;
    atom_count: number;
    doi: string | null;
    retrieved_at: string | null;
    atoms: StructureAtomView[];
}

export interface StructurePageResponse {
    items: StructureRecordView[];
    pagination: Pagination;
}
