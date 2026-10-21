// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { EChartsCoreOption } from 'echarts/core';
import type { FacetCount, HistogramBin, IsothermSeries, UptakeAtPressureRow } from '../../models/analytics.model';

export type ColorBy = 'temperature' | 'material' | 'adsorbate' | 'dataset';
export type GroupBy = 'material' | 'adsorbate';

// Distinguishable categorical hues and a cold-to-hot ramp for temperature.
export const CATEGORICAL_PALETTE = ['#2563eb', '#d97706', '#059669', '#dc2626', '#7c3aed', '#0891b2', '#db2777', '#65a30d', '#475569', '#ea580c'];
export const TEMPERATURE_RAMP = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027', '#a50026'];

// Canvas rendering cannot read CSS variables; these mirror the slate tokens.
const AXIS_COLOR = '#64748b';
const GRID_COLOR = '#e2e8f0';
const TEXT_COLOR = '#334155';
const SURFACE_COLOR = '#ffffff';

export function formatValue(value: number | null | undefined): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return '—';
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e5)) return value.toExponential(2);
    return Number(value.toPrecision(4)).toString();
}

export function pressureLabel(unit: string): string {
    return unit === '1' ? 'p/p₀' : unit;
}

export function temperatureColor(value: number, min: number, max: number): string {
    const ratio = max > min ? (value - min) / (max - min) : 0.5;
    const index = Math.round(Math.min(1, Math.max(0, ratio)) * (TEMPERATURE_RAMP.length - 1));
    return TEMPERATURE_RAMP[index];
}

export function seriesLabel(series: { material: string; adsorbate: string; temperature_k: number }): string {
    return `${series.material} / ${series.adsorbate} @ ${formatValue(series.temperature_k)} K`;
}

function categoryKey(series: IsothermSeries, colorBy: ColorBy): string {
    if (colorBy === 'material') return series.material;
    if (colorBy === 'adsorbate') return series.adsorbate;
    return series.dataset;
}

function valueAxis(name: string, log: boolean): Record<string, unknown> {
    return {
        type: log ? 'log' : 'value',
        name,
        nameLocation: 'middle',
        nameGap: 34,
        nameTextStyle: { color: TEXT_COLOR },
        axisLine: { show: true, lineStyle: { color: AXIS_COLOR } },
        axisLabel: { color: AXIS_COLOR, formatter: (value: number) => formatValue(value) },
        splitLine: { lineStyle: { color: GRID_COLOR } },
        scale: !log,
    };
}

const BASE_OPTION = {
    animation: false,
    textStyle: { fontFamily: 'inherit' },
    toolbox: { right: 8, feature: { dataZoom: { yAxisIndex: 'none' }, restore: {}, saveAsImage: { title: 'Save image' } } },
};

export interface IsothermChartOptions {
    colorBy: ColorBy;
    logPressure: boolean;
    pressureUnit: string;
}

export function buildIsothermOption(series: IsothermSeries[], options: IsothermChartOptions): EChartsCoreOption {
    const temperatures = series.map((item) => item.temperature_k);
    const minT = Math.min(...temperatures);
    const maxT = Math.max(...temperatures);
    const categories = [...new Set(series.map((item) => categoryKey(item, options.colorBy)))];
    const uptakeUnits = [...new Set(series.map((item) => item.uptake_unit))];
    return {
        ...BASE_OPTION,
        grid: { left: 64, right: 24, top: 48, bottom: 88 },
        legend: { type: 'scroll', bottom: 0, textStyle: { color: TEXT_COLOR } },
        tooltip: {
            trigger: 'item',
            formatter: (params: { seriesIndex: number; value: [number, number] }) => {
                const item = series[params.seriesIndex];
                return [
                    `<strong>${seriesLabel(item)}</strong>`,
                    `${item.dataset} (${item.dataset_source}) · ${item.isotherm}`,
                    `p = ${formatValue(params.value[0])} ${pressureLabel(options.pressureUnit)}`,
                    `q = ${formatValue(params.value[1])} ${item.uptake_unit} (measured)`,
                ].join('<br>');
            },
        },
        dataZoom: [{ type: 'inside', xAxisIndex: 0, filterMode: 'none' }, { type: 'inside', yAxisIndex: 0, filterMode: 'none' }],
        xAxis: valueAxis(`Pressure (${pressureLabel(options.pressureUnit)})`, options.logPressure),
        yAxis: valueAxis(uptakeUnits.length === 1 ? `Uptake (${uptakeUnits[0]})` : 'Uptake (mixed units, see warnings)', false),
        series: series.map((item) => {
            const color = options.colorBy === 'temperature'
                ? temperatureColor(item.temperature_k, minT, maxT)
                : CATEGORICAL_PALETTE[categories.indexOf(categoryKey(item, options.colorBy)) % CATEGORICAL_PALETTE.length];
            return {
                type: 'line',
                name: seriesLabel(item),
                color,
                showSymbol: true,
                symbolSize: 6,
                lineStyle: { width: 1.25, opacity: 0.6 },
                data: item.points
                    .filter((point) => !options.logPressure || point.pressure > 0)
                    .map((point) => [point.pressure, point.uptake]),
            };
        }),
    };
}

/** Valid rows grouped into chart series, each ordered by temperature. */
export function groupUptakeRows(rows: UptakeAtPressureRow[], groupBy: GroupBy): { groups: string[]; grouped: UptakeAtPressureRow[][] } {
    const valid = rows.filter((row) => row.status === 'ok' && row.value !== null);
    const groups = [...new Set(valid.map((row) => (groupBy === 'material' ? row.material : row.adsorbate)))];
    const grouped = groups.map((group) =>
        valid
            .filter((row) => (groupBy === 'material' ? row.material : row.adsorbate) === group)
            .sort((a, b) => a.temperature_k - b.temperature_k),
    );
    return { groups, grouped };
}

export function buildUptakeOption(rows: UptakeAtPressureRow[], groupBy: GroupBy, uptakeUnit: string): EChartsCoreOption {
    const { groups, grouped } = groupUptakeRows(rows, groupBy);
    return {
        ...BASE_OPTION,
        grid: { left: 64, right: 24, top: 48, bottom: 88 },
        legend: { type: 'scroll', bottom: 0, textStyle: { color: TEXT_COLOR } },
        tooltip: {
            trigger: 'item',
            formatter: (params: { seriesIndex: number; dataIndex: number }) => {
                const row = grouped[params.seriesIndex][params.dataIndex];
                const sources = row.sources.map((point) => `(${formatValue(point.pressure)}, ${formatValue(point.uptake)})`).join(', ');
                return [
                    `<strong>${seriesLabel(row)}</strong>`,
                    `${row.dataset} · ${row.isotherm}`,
                    `q = ${formatValue(row.value)} ${row.uptake_unit} (${row.method})`,
                    `From observations: ${sources}`,
                ].join('<br>');
            },
        },
        xAxis: valueAxis('Temperature (K)', false),
        yAxis: valueAxis(`Uptake (${uptakeUnit})`, false),
        series: grouped.map((items, index) => {
            const color = CATEGORICAL_PALETTE[index % CATEGORICAL_PALETTE.length];
            return {
                type: 'line',
                name: groups[index],
                color,
                symbolSize: 9,
                lineStyle: { width: 1, type: 'dashed', opacity: 0.5 },
                data: items.map((row) => ({
                    value: [row.temperature_k, row.value],
                    symbol: row.method === 'nearest' ? 'diamond' : 'circle',
                    // Hollow markers flag calculated (interpolated) values.
                    itemStyle: row.method === 'interpolated'
                        ? { color: SURFACE_COLOR, borderColor: color, borderWidth: 2 }
                        : { color },
                })),
            };
        }),
    };
}

export function binLabel(bin: HistogramBin): string {
    return `${formatValue(bin.lower)}–${formatValue(bin.upper)}`;
}

export function buildHistogramOption(bins: HistogramBin[], axisName: string): EChartsCoreOption {
    return {
        ...BASE_OPTION,
        toolbox: { right: 8, feature: { saveAsImage: { title: 'Save image' } } },
        grid: { left: 56, right: 16, top: 40, bottom: 72 },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        xAxis: {
            type: 'category',
            name: axisName,
            nameLocation: 'middle',
            nameGap: 52,
            nameTextStyle: { color: TEXT_COLOR },
            data: bins.map(binLabel),
            axisLabel: { color: AXIS_COLOR, rotate: 30 },
            axisLine: { lineStyle: { color: AXIS_COLOR } },
        },
        yAxis: { type: 'value', name: 'Count', axisLabel: { color: AXIS_COLOR }, splitLine: { lineStyle: { color: GRID_COLOR } } },
        series: [{ type: 'bar', name: 'Count', color: CATEGORICAL_PALETTE[0], barCategoryGap: '4%', data: bins.map((bin) => bin.count) }],
    };
}

export function buildRankingOption(items: FacetCount[], name: string, color = CATEGORICAL_PALETTE[0]): EChartsCoreOption {
    const ordered = [...items].reverse();
    return {
        animation: false,
        textStyle: { fontFamily: 'inherit' },
        grid: { left: 8, right: 40, top: 8, bottom: 8, containLabel: true },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        xAxis: { type: 'value', show: false },
        yAxis: {
            type: 'category',
            data: ordered.map((item) => item.name),
            axisLabel: { color: TEXT_COLOR, width: 160, overflow: 'truncate' },
            axisLine: { show: false },
            axisTick: { show: false },
        },
        series: [{ type: 'bar', name, color, barMaxWidth: 18, label: { show: true, position: 'right', color: AXIS_COLOR }, data: ordered.map((item) => item.count) }],
    };
}
