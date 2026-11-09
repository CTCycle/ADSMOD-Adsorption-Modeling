// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { ExternalIdentifierView, Pagination, StructureRecordView } from './public-data.model';

export type IdentityStatus = 'resolved' | 'provisional' | 'generic';

export interface DatasetUsage {
    id: number;
    name: string;
    experiment_count: number;
}

export interface MaterialRecord {
    id: number;
    name: string;
    identity_status: IdentityStatus;
    formula: string | null;
    molar_mass_g_mol: number | null;
    smiles: string | null;
    structure_count: number;
    isotherm_count: number;
    external_identifiers: ExternalIdentifierView[];
}

export interface MaterialDetail extends MaterialRecord {
    structures: StructureRecordView[];
    datasets: DatasetUsage[];
}

export interface MaterialPage {
    items: MaterialRecord[];
    pagination: Pagination;
}

export interface SpeciesProperty {
    key: string;
    value_number: number | null;
    value_text: string | null;
    unit: string | null;
    source: string;
}

export interface SpeciesRecord {
    id: number;
    name: string;
    identity_status: IdentityStatus;
    preferred_name: string | null;
    formula: string | null;
    molecular_weight: number | null;
    inchi: string | null;
    inchi_key: string | null;
    connectivity_smiles: string | null;
    smiles: string | null;
    pubchem_cid: string | null;
    synonyms: string[];
    properties: SpeciesProperty[];
    external_identifiers: ExternalIdentifierView[];
    structure_2d_url: string | null;
    conformer_3d_url: string | null;
    conformer_3d_cached: boolean;
    isotherm_count: number;
    retrieved_at: string | null;
}

export interface SpeciesDetail extends SpeciesRecord {
    datasets: DatasetUsage[];
}

export interface SpeciesPage {
    items: SpeciesRecord[];
    pagination: Pagination;
}

export type IdentityReviewStatus = 'open' | 'merged' | 'rejected';

export interface IdentityEntity {
    id: number;
    name: string;
    identity_status: IdentityStatus;
    identifier: string | null;
    experiment_count: number;
}

export interface IdentityReview {
    id: number;
    kind: 'adsorbate' | 'adsorbent';
    raw_name: string;
    reason: string;
    score: number | null;
    evidence: Record<string, unknown>;
    status: IdentityReviewStatus;
    entity: IdentityEntity | null;
    candidate: IdentityEntity | null;
    created_at: string;
    decided_at: string | null;
}

export interface IdentityReviewPage {
    items: IdentityReview[];
    pagination: Pagination;
}

export interface IdentityAuditResult {
    adsorbate: number;
    adsorbent: number;
}
