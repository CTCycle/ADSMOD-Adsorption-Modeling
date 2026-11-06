// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export type SourcesView = 'catalog' | 'acquire' | 'structures';
export type SourceStatus = 'available' | 'degraded' | 'unavailable' | 'unknown';
export type PublicDataSyncStatus = 'queued' | 'running' | 'completed' | 'partially_completed' | 'failed' | 'cancelled' | 'interrupted';
export type PublicDataSyncOperation = 'health_check' | 'nist_index' | 'nist_category' | 'pubchem_resolve' | 'pubchem_enrich' | 'cod_import';
export type NISTCategory = 'experiments' | 'guest' | 'host';

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
    homepage_url: string;
    license_name: string | null;
    license_url: string | null;
    terms_url: string | null;
    record_count: number;
    health_status: SourceStatus;
    health_detail: string | null;
    last_health_checked_at: string | null;
    health_fresh: boolean;
    last_successful_sync_at: string | null;
}

export interface PublicSourceListResponse {
    sources: PublicSourceSummary[];
}

export interface PublicDataOverviewResponse {
    adsorption_records: number;
    materials: number;
    chemicals: number;
    structures: number;
    species_total: number;
    species_enriched: number;
    materials_total: number;
    materials_with_structure: number;
    active_sync_runs: number;
    failed_sync_runs: number;
    latest_successful_sync_at: string | null;
}

export interface PublicDataSyncRequest {
    source_key: 'nist' | 'pubchem' | 'cod';
    operation: Exclude<PublicDataSyncOperation, 'health_check'>;
    category?: NISTCategory;
    fraction?: number;
    query?: string;
    adsorbate_id?: number;
    cod_id?: string;
    cod_ids?: string[];
    adsorbent_id?: number;
    identifiers?: string[];
    overwrite_local_edits?: boolean;
    /** pubchem_enrich only: download missing 3D conformers instead of resolving identities. */
    conformers?: boolean;
}

export interface NISTSearchItem {
    filename: string;
    doi: string | null;
    adsorbent_name: string | null;
    adsorbent_hashkey: string | null;
    adsorbate_names: string[];
    adsorbate_inchi_keys: string[];
    temperature_k: number | null;
    local: boolean;
}

export interface NISTSearchResponse {
    items: NISTSearchItem[];
    total: number;
}

export interface ChemicalLookupItem {
    cid: string;
    name: string;
    formula: string | null;
    molecular_weight: number | null;
    inchi_key: string | null;
    smiles: string | null;
    source_url: string;
    structure_2d_url: string | null;
    local_adsorbate_id: number | null;
}

export interface ChemicalLookupResponse {
    items: ChemicalLookupItem[];
}

export interface NISTEditedExperimentsResponse {
    count: number;
}

export interface NISTCollectionView {
    category: NISTCategory;
    local_count: number;
    available_count: number | null;
    indexed_at: string | null;
}

export interface NISTCollectionListResponse {
    collections: NISTCollectionView[];
}

export interface PublicDataSyncRun {
    id: string;
    source_key: string;
    operation: PublicDataSyncOperation;
    parameters: Record<string, unknown>;
    status: PublicDataSyncStatus;
    progress: number;
    processed_count: number;
    imported_count: number;
    skipped_count: number;
    failed_count: number;
    checkpoint: Record<string, unknown>;
    cancel_requested: boolean;
    error_detail: string | null;
    created_at: string;
    started_at: string | null;
    finished_at: string | null;
}

export interface PublicDataSyncRunListResponse {
    runs: PublicDataSyncRun[];
    total: number;
}

export interface PublicDataSyncFailure {
    id: number;
    source_key: string;
    external_id: string | null;
    reason: string;
    retry_eligible_at: string | null;
    created_at: string;
}

export interface PublicDataSyncFailureListResponse {
    failures: PublicDataSyncFailure[];
    total: number;
}

export interface EntityReference {
    id: number;
    name: string;
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
    reference_title: string | null;
    retrieved_at: string | null;
    atoms: StructureAtomView[];
}

export interface StructurePageResponse {
    items: StructureRecordView[];
    pagination: Pagination;
}
