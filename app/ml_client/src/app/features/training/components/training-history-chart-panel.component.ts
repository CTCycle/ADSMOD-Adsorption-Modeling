import { Component, computed, input, signal } from '@angular/core';
import type { TrainingHistoryPoint, TrainingMetricKey } from '../../../models/training.model';

interface TrainingChartLineConfig {
    color: string;
    dataKey: TrainingMetricKey;
    name: string;
}

interface ChartPoint {
    epoch: number;
    primaryValue: number;
    secondaryValue: number;
    x: number;
    primaryY: number;
    secondaryY: number;
}

@Component({
    selector: 'adsmod-training-history-chart-panel',
    standalone: true,
    template: `
        <div class="chart-panel">
            <div class="chart-title">{{ title() }}</div>
            @if (hasHistory() && chartPoints().length > 0) {
                <div class="chart-wrapper" style="width: 100%; height: 250px;">
                    <svg viewBox="0 0 640 250" preserveAspectRatio="none" style="width: 100%; height: 250px;">
                        @for (grid of yGridLines(); track grid.value) {
                            <line
                                x1="52"
                                [attr.y1]="grid.y"
                                x2="620"
                                [attr.y2]="grid.y"
                                stroke="#e2e8f0"
                                stroke-dasharray="3 3"
                            />
                            <text x="10" [attr.y]="grid.y + 4" font-size="12" fill="#64748b">{{ grid.label }}</text>
                        }

                        <line x1="52" y1="210" x2="620" y2="210" stroke="#cbd5e1"></line>
                        <line x1="52" y1="20" x2="52" y2="210" stroke="#cbd5e1"></line>

                        <path [attr.d]="primaryPath()" fill="none" [attr.stroke]="primaryLine().color" stroke-width="2"></path>
                        <path [attr.d]="secondaryPath()" fill="none" [attr.stroke]="secondaryLine().color" stroke-width="2"></path>

                        @for (point of chartPoints(); track point.epoch) {
                            <text [attr.x]="point.x - 6" y="230" font-size="12" fill="#64748b">{{ point.epoch }}</text>
                            <circle
                                [attr.cx]="point.x"
                                [attr.cy]="point.primaryY"
                                r="8"
                                fill="transparent"
                                (mouseenter)="hoveredEpoch.set(point.epoch)"
                                (mouseleave)="hoveredEpoch.set(null)"
                            ></circle>
                            <circle
                                [attr.cx]="point.x"
                                [attr.cy]="point.secondaryY"
                                r="8"
                                fill="transparent"
                                (mouseenter)="hoveredEpoch.set(point.epoch)"
                                (mouseleave)="hoveredEpoch.set(null)"
                            ></circle>
                        }

                        @if (tooltipPoint()) {
                            <g>
                                <rect x="410" y="24" width="190" height="62" rx="6" fill="rgba(255,255,255,0.95)" style="filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.1));"></rect>
                                <text x="424" y="45" font-size="12" fill="#0f172a">Epoch {{ tooltipPoint()!.epoch }}</text>
                                <text x="424" y="62" font-size="12" [attr.fill]="primaryLine().color">
                                    {{ primaryLine().name }}: {{ formatMetric(tooltipPoint()!.primaryValue) }}
                                </text>
                                <text x="424" y="79" font-size="12" [attr.fill]="secondaryLine().color">
                                    {{ secondaryLine().name }}: {{ formatMetric(tooltipPoint()!.secondaryValue) }}
                                </text>
                            </g>
                        }
                    </svg>

                    <div style="display: flex; gap: 1rem; padding: 0 0.75rem 0.75rem;">
                        <span style="display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; color: var(--slate-600);">
                            <span style="width: 0.75rem; height: 0.75rem; border-radius: 999px;" [style.background]="primaryLine().color"></span>
                            {{ primaryLine().name }}
                        </span>
                        <span style="display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; color: var(--slate-600);">
                            <span style="width: 0.75rem; height: 0.75rem; border-radius: 999px;" [style.background]="secondaryLine().color"></span>
                            {{ secondaryLine().name }}
                        </span>
                    </div>
                </div>
            } @else {
                <div class="chart-placeholder">
                    Waiting for training data...
                    <small>{{ placeholderHint() }}</small>
                </div>
            }
        </div>
    `,
})
export class TrainingHistoryChartPanelComponent {
    readonly title = input.required<string>();
    readonly hasHistory = input.required<boolean>();
    readonly history = input.required<TrainingHistoryPoint[]>();
    readonly primaryLine = input.required<TrainingChartLineConfig>();
    readonly secondaryLine = input.required<TrainingChartLineConfig>();
    readonly placeholderHint = input.required<string>();
    readonly yAxisDomain = input<[number, number] | ['auto', 'auto'] | undefined>(undefined);
    protected readonly hoveredEpoch = signal<number | null>(null);

    protected readonly chartPoints = computed<ChartPoint[]>(() => {
        const history = this.history();
        if (!history.length) {
            return [];
        }

        const domain = this.resolveDomain();
        const minEpoch = history[0]?.epoch ?? 0;
        const maxEpoch = history[history.length - 1]?.epoch ?? minEpoch;
        const epochRange = maxEpoch - minEpoch || 1;
        return history.map((entry) => {
            const primaryValue = typeof entry[this.primaryLine().dataKey] === 'number' ? entry[this.primaryLine().dataKey]! : 0;
            const secondaryValue = typeof entry[this.secondaryLine().dataKey] === 'number' ? entry[this.secondaryLine().dataKey]! : 0;
            const x = history.length === 1 ? 336 : 52 + ((entry.epoch - minEpoch) / epochRange) * 568;
            return {
                epoch: entry.epoch,
                primaryValue,
                secondaryValue,
                x,
                primaryY: this.toChartY(primaryValue, domain.min, domain.max),
                secondaryY: this.toChartY(secondaryValue, domain.min, domain.max),
            };
        });
    });

    protected readonly primaryPath = computed(() => this.buildPath(this.chartPoints().map((point) => [point.x, point.primaryY] as const)));
    protected readonly secondaryPath = computed(() => this.buildPath(this.chartPoints().map((point) => [point.x, point.secondaryY] as const)));
    protected readonly tooltipPoint = computed(() => this.chartPoints().find((point) => point.epoch === this.hoveredEpoch()) ?? null);
    protected readonly yGridLines = computed(() => {
        const domain = this.resolveDomain();
        return Array.from({ length: 4 }, (_, index) => {
            const ratio = index / 3;
            const value = domain.max - (domain.max - domain.min) * ratio;
            return {
                value,
                y: this.toChartY(value, domain.min, domain.max),
                label: this.formatMetric(value),
            };
        });
    });

    protected formatMetric(value: number): string {
        if (this.yAxisDomain()?.[0] === 0 && this.yAxisDomain()?.[1] === 1) {
            return `${(value * 100).toFixed(2)}%`;
        }
        return value.toFixed(4);
    }

    private resolveDomain(): { min: number; max: number } {
        const configuredDomain = this.yAxisDomain();
        if (configuredDomain && typeof configuredDomain[0] === 'number' && typeof configuredDomain[1] === 'number') {
            return { min: configuredDomain[0], max: configuredDomain[1] || 1 };
        }

        const values = this.history().flatMap((entry) => {
            const primaryValue = entry[this.primaryLine().dataKey];
            const secondaryValue = entry[this.secondaryLine().dataKey];
            return [
                typeof primaryValue === 'number' ? primaryValue : undefined,
                typeof secondaryValue === 'number' ? secondaryValue : undefined,
            ].filter((value): value is number => typeof value === 'number');
        });
        const min = values.length ? Math.min(...values) : 0;
        const max = values.length ? Math.max(...values) : 1;
        if (min === max) {
            return { min: min - 0.5, max: max + 0.5 };
        }
        return { min, max };
    }

    private toChartY(value: number, min: number, max: number): number {
        const ratio = (value - min) / (max - min || 1);
        return 210 - ratio * 190;
    }

    private buildPath(points: readonly (readonly [number, number])[]): string {
        return points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'} ${x} ${y}`).join(' ');
    }
}
