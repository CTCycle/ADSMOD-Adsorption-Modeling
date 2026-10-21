// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { EntityReference, Pagination } from './public-data.model';

export type DatasetSource = 'uploaded' | 'nist';
export type PressureBasis = 'absolute' | 'partial' | 'relative';
export type DistributionVariable = 'temperature_k' | 'pressure' | 'uptake_mol_kg' | 'points_per_isotherm';
export type HistogramScale = 'linear' | 'log';
export type UptakeMethod = 'linear' | 'nearest';
export type ValueMethod = 'measured' | 'nearest' | 'interpolated';
export type ValueStatus = 'ok' | 'out_of_range' | 'no_point_in_tolerance';

export interface AnalyticsFilter {
    dataset_sources: DatasetSource[];
    dataset_ids: number[];
    adsorbent_ids: number[];
    adsorbate_ids: number[];
    isotherm_ids: number[];
    temperature_min_k: number | null;
    temperature_max_k: number | null;
    pressure_basis: PressureBasis | null;
    include_mixtures: boolean;
}

export interface FacetCount {
    id: number;
    name: string;
    count: number;
}

export interface SelectionFacets {
    materials: FacetCount[];
    adsorbates: FacetCount[];
    datasets: FacetCount[];
    sources: Record<string, number>;
    pressure_bases: Record<string, number>;
    temperature_min_k: number | null;
    temperature_max_k: number | null;
}

export interface SelectionRequest {
    filter: AnalyticsFilter;
    page: number;
    page_size: number;
    material_search: string | null;
    adsorbate_search: string | null;
}

export interface IsothermSummaryView {
    id: number;
    name: string;
    dataset_id: number;
    dataset: string;
    dataset_source: DatasetSource;
    material_id: number;
    material: string;
    adsorbates: EntityReference[];
    temperature_k: number;
    pressure_basis: PressureBasis;
    point_count: number;
}

export interface SelectionResponse {
    items: IsothermSummaryView[];
    pagination: Pagination;
    facets: SelectionFacets;
}

export interface SeriesRequest {
    filter: AnalyticsFilter;
    pressure_unit: string;
    uptake_unit: string;
}

export interface SeriesMetadata {
    isotherm_id: number;
    component_id: number;
    isotherm: string;
    dataset_id: number;
    dataset: string;
    dataset_source: DatasetSource;
    material_id: number;
    material: string;
    adsorbate_id: number;
    adsorbate: string;
    mole_fraction: number | null;
    temperature_k: number;
    uptake_unit: string;
}

export interface SeriesPoint {
    observation_id: number;
    pressure: number;
    uptake: number;
    uptake_stddev: number | null;
}

export interface IsothermSeries extends SeriesMetadata {
    points: SeriesPoint[];
}

export interface SeriesResponse {
    pressure_basis: PressureBasis | null;
    pressure_unit: string;
    series: IsothermSeries[];
    total_series: number;
    truncated: boolean;
    warnings: string[];
}

export interface UptakeAtPressureRequest extends SeriesRequest {
    pressure: number;
    method: UptakeMethod;
    tolerance: number;
}

export interface UptakeAtPressureRow extends SeriesMetadata {
    value: number | null;
    method: ValueMethod | null;
    status: ValueStatus;
    sources: SeriesPoint[];
}

export interface UptakeAtPressureResponse {
    pressure: number;
    pressure_unit: string;
    pressure_basis: PressureBasis | null;
    method: UptakeMethod;
    rows: UptakeAtPressureRow[];
    total_series: number;
    truncated: boolean;
    warnings: string[];
}

export interface DistributionRequest {
    filter: AnalyticsFilter;
    variable: DistributionVariable;
    bins: number;
    scale: HistogramScale;
    pressure_unit: string;
}

export interface HistogramBin {
    lower: number;
    upper: number;
    count: number;
}

export interface DistributionStats {
    count: number;
    min: number | null;
    max: number | null;
    mean: number | null;
    median: number | null;
}

export interface DistributionResponse {
    variable: DistributionVariable;
    unit: string;
    scale: HistogramScale;
    bins: HistogramBin[];
    stats: DistributionStats;
    excluded: number;
}

export interface BasisSpan {
    pressure_basis: PressureBasis;
    pressure_unit: 'Pa' | '1';
    isotherms: number;
    observations: number;
    pressure_min: number | null;
    pressure_max: number | null;
    uptake_min_mol_kg: number | null;
    uptake_max_mol_kg: number | null;
    temperature_min_k: number | null;
    temperature_max_k: number | null;
}

export interface AnalyticsOverviewResponse {
    datasets: number;
    isotherms: number;
    observations: number;
    materials: number;
    adsorbates: number;
    mixture_isotherms: number;
    isotherms_by_source: Record<string, number>;
    pressure_bases: BasisSpan[];
    top_materials: FacetCount[];
    top_adsorbates: FacetCount[];
    temperature_histogram: HistogramBin[];
}
