// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import * as echarts from 'echarts/core';
import type { ResolvedTheme } from '../../core/theme/theme.service';

// Canvas rendering cannot read CSS variables; these mirror the slate tokens of each theme.
const PALETTES: Record<ResolvedTheme, { text: string; axis: string; grid: string; surface: string; border: string }> = {
    light: { text: '#334155', axis: '#64748b', grid: '#e2e8f0', surface: '#ffffff', border: '#cbd5e1' },
    dark: { text: '#c6d0df', axis: '#8c9bb3', grid: '#26364f', surface: '#111a2b', border: '#374a68' },
};

function buildTheme(palette: (typeof PALETTES)[ResolvedTheme]): Record<string, unknown> {
    const axis = {
        nameTextStyle: { color: palette.text },
        axisLine: { lineStyle: { color: palette.axis } },
        axisLabel: { color: palette.axis },
        splitLine: { lineStyle: { color: palette.grid } },
    };
    return {
        textStyle: { color: palette.text },
        categoryAxis: axis,
        valueAxis: axis,
        logAxis: axis,
        legend: { textStyle: { color: palette.text }, pageTextStyle: { color: palette.text }, pageIconColor: palette.axis },
        bar: { label: { color: palette.axis } },
        tooltip: { backgroundColor: palette.surface, borderColor: palette.border, textStyle: { color: palette.text } },
        toolbox: { iconStyle: { borderColor: palette.axis } },
    };
}

export const chartThemeName = (theme: ResolvedTheme): string => `adsmod-${theme}`;

for (const theme of ['light', 'dark'] as const) {
    echarts.registerTheme(chartThemeName(theme), buildTheme(PALETTES[theme]));
}
