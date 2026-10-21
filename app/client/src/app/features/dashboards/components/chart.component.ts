// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { afterNextRender, Component, effect, ElementRef, inject, input, output, signal, viewChild } from '@angular/core';
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import { DataZoomComponent, GridComponent, LegendComponent, ToolboxComponent, TooltipComponent } from 'echarts/components';
import * as echarts from 'echarts/core';
import type { ECharts, EChartsCoreOption } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { ThemeService } from '../../../core/theme/theme.service';
import { chartThemeName } from '../chart-theme';

echarts.use([LineChart, ScatterChart, BarChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, ToolboxComponent, CanvasRenderer]);

export interface ChartPointEvent {
    seriesIndex: number;
    dataIndex: number;
}

@Component({
    selector: 'adsmod-chart',
    standalone: true,
    template: `<div #host class="chart-host" role="img" [attr.aria-label]="ariaLabel()"></div>`,
    styles: [':host { display: block; min-height: inherit; height: 100%; } .chart-host { width: 100%; height: 100%; min-height: inherit; }'],
})
export class ChartComponent {
    readonly option = input.required<EChartsCoreOption>();
    readonly ariaLabel = input.required<string>();
    readonly pointClick = output<ChartPointEvent>();

    private readonly host = viewChild.required<ElementRef<HTMLDivElement>>('host');
    private readonly chart = signal<ECharts | null>(null);
    private readonly rendered = signal(false);
    private readonly theme = inject(ThemeService);

    constructor() {
        afterNextRender(() => this.rendered.set(true));
        // ECharts themes apply at init time, so the chart is re-created when the theme changes.
        effect((onCleanup) => {
            if (!this.rendered()) {
                return;
            }
            const chart = echarts.init(this.host().nativeElement, chartThemeName(this.theme.resolved()), { renderer: 'canvas' });
            chart.on('click', (event) => this.pointClick.emit({ seriesIndex: event.seriesIndex ?? 0, dataIndex: event.dataIndex }));
            const observer = new ResizeObserver(() => chart.resize());
            observer.observe(this.host().nativeElement);
            onCleanup(() => {
                observer.disconnect();
                chart.dispose();
            });
            this.chart.set(chart);
        });
        effect(() => {
            this.chart()?.setOption(this.option(), { notMerge: true });
        });
    }
}
