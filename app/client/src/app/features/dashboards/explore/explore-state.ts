// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { Params } from '@angular/router';
import type {
    AnalyticsFilter,
    DatasetSource,
    DistributionVariable,
    HistogramScale,
    PressureBasis,
    UptakeMethod,
} from '../../../models/analytics.model';
import type { ColorBy, GroupBy } from '../dashboard-charts';

export type ExploreMode = 'isotherms' | 'uptake' | 'distribution';

// The URL is the only place a view is kept: it can be bookmarked or shared,
// and every chart is recomputed from the database when it is opened.
export interface ExploreState {
    filter: AnalyticsFilter;
    selectedIds: number[];
    mode: ExploreMode;
    colorBy: ColorBy;
    logPressure: boolean;
    pressureUnit: string;
    uptakeUnit: string;
    targetPressure: number;
    method: UptakeMethod;
    tolerance: number;
    groupBy: GroupBy;
    variable: DistributionVariable;
    bins: number;
    scale: HistogramScale;
}

export const PRESSURE_UNITS = ['Pa', 'kPa', 'MPa', 'bar', 'mbar', 'atm', 'torr', 'psi'];
export const UPTAKE_UNITS = ['mol/kg', 'mmol/g', 'mg/g', 'g/g', 'wt%', 'cm3(STP)/g'];

export const DEFAULT_FILTER: AnalyticsFilter = {
    dataset_sources: [],
    dataset_ids: [],
    adsorbent_ids: [],
    adsorbate_ids: [],
    isotherm_ids: [],
    temperature_min_k: null,
    temperature_max_k: null,
    pressure_basis: 'absolute',
    include_mixtures: false,
};

export const DEFAULT_EXPLORE_STATE: ExploreState = {
    filter: DEFAULT_FILTER,
    selectedIds: [],
    mode: 'isotherms',
    colorBy: 'temperature',
    logPressure: false,
    pressureUnit: 'bar',
    uptakeUnit: 'mol/kg',
    targetPressure: 1,
    method: 'linear',
    tolerance: 0.05,
    groupBy: 'material',
    variable: 'temperature_k',
    bins: 20,
    scale: 'linear',
};

const ids = (value: unknown): number[] =>
    String(value ?? '')
        .split(',')
        .map(Number)
        .filter((item) => Number.isInteger(item) && item > 0);

const positive = (value: unknown, fallback: number | null): number | null => {
    const parsed = Number(value);
    return value !== undefined && value !== '' && Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
};

function oneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
    return allowed.includes(value as T) ? (value as T) : fallback;
}

export function stateFromQueryParams(params: Params): ExploreState {
    const defaults = DEFAULT_EXPLORE_STATE;
    const sources = String(params['sources'] ?? '')
        .split(',')
        .filter((item): item is DatasetSource => item === 'uploaded' || item === 'nist');
    return {
        filter: {
            dataset_sources: sources,
            dataset_ids: ids(params['datasets']),
            adsorbent_ids: ids(params['materials']),
            adsorbate_ids: ids(params['adsorbates']),
            isotherm_ids: [],
            temperature_min_k: positive(params['tmin'], null),
            temperature_max_k: positive(params['tmax'], null),
            pressure_basis: oneOf<PressureBasis>(params['basis'], ['absolute', 'partial', 'relative'], 'absolute'),
            include_mixtures: params['mixtures'] === '1',
        },
        selectedIds: ids(params['selected']),
        mode: oneOf<ExploreMode>(params['mode'], ['isotherms', 'uptake', 'distribution'], defaults.mode),
        colorBy: oneOf<ColorBy>(params['color'], ['temperature', 'material', 'adsorbate', 'dataset'], defaults.colorBy),
        logPressure: params['log'] === '1',
        pressureUnit: oneOf(params['punit'], PRESSURE_UNITS, defaults.pressureUnit),
        uptakeUnit: oneOf(params['qunit'], UPTAKE_UNITS, defaults.uptakeUnit),
        targetPressure: positive(params['p'], defaults.targetPressure) ?? defaults.targetPressure,
        method: oneOf<UptakeMethod>(params['method'], ['linear', 'nearest'], defaults.method),
        tolerance: Math.min(1, positive(params['tol'], defaults.tolerance) ?? defaults.tolerance),
        groupBy: oneOf<GroupBy>(params['group'], ['material', 'adsorbate'], defaults.groupBy),
        variable: oneOf<DistributionVariable>(params['var'], ['temperature_k', 'pressure', 'uptake_mol_kg', 'points_per_isotherm'], defaults.variable),
        bins: Math.min(100, Math.round(positive(params['bins'], defaults.bins) ?? defaults.bins)),
        scale: oneOf<HistogramScale>(params['scale'], ['linear', 'log'], defaults.scale),
    };
}

export function stateToQueryParams(state: ExploreState): Params {
    const defaults = DEFAULT_EXPLORE_STATE;
    const list = (values: readonly (number | string)[]): string | null => (values.length ? values.join(',') : null);
    const unlessDefault = <T>(value: T, fallback: T): T | null => (value === fallback ? null : value);
    return {
        sources: list(state.filter.dataset_sources),
        datasets: list(state.filter.dataset_ids),
        materials: list(state.filter.adsorbent_ids),
        adsorbates: list(state.filter.adsorbate_ids),
        tmin: state.filter.temperature_min_k,
        tmax: state.filter.temperature_max_k,
        basis: unlessDefault(state.filter.pressure_basis, DEFAULT_FILTER.pressure_basis),
        mixtures: state.filter.include_mixtures ? '1' : null,
        selected: list(state.selectedIds),
        mode: unlessDefault(state.mode, defaults.mode),
        color: unlessDefault(state.colorBy, defaults.colorBy),
        log: state.logPressure ? '1' : null,
        punit: unlessDefault(state.pressureUnit, defaults.pressureUnit),
        qunit: unlessDefault(state.uptakeUnit, defaults.uptakeUnit),
        p: unlessDefault(state.targetPressure, defaults.targetPressure),
        method: unlessDefault(state.method, defaults.method),
        tol: unlessDefault(state.tolerance, defaults.tolerance),
        group: unlessDefault(state.groupBy, defaults.groupBy),
        var: unlessDefault(state.variable, defaults.variable),
        bins: unlessDefault(state.bins, defaults.bins),
        scale: unlessDefault(state.scale, defaults.scale),
    };
}

/** Filter sent to chart queries: explicit selections narrow the matching set. */
export function chartFilter(state: ExploreState): AnalyticsFilter {
    return { ...state.filter, isotherm_ids: state.selectedIds };
}
