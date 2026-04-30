import React from 'react';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import type { TrainingHistoryPoint, TrainingMetricKey } from '../../../types';

interface TrainingChartLineConfig {
    color: string;
    dataKey: TrainingMetricKey;
    name: string;
}

interface TrainingHistoryChartPanelProps {
    title: string;
    hasHistory: boolean;
    history: TrainingHistoryPoint[];
    primaryLine: TrainingChartLineConfig;
    secondaryLine: TrainingChartLineConfig;
    placeholderHint: string;
    yAxisDomain?: [number, number] | ['auto', 'auto'];
}

const TOOLTIP_STYLE = {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '4px',
    border: 'none',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
};

export const TrainingHistoryChartPanel: React.FC<TrainingHistoryChartPanelProps> = ({
    title,
    hasHistory,
    history,
    primaryLine,
    secondaryLine,
    placeholderHint,
    yAxisDomain,
}) => (
    <div className="chart-panel">
        <div className="chart-title">{title}</div>
        {hasHistory ? (
            <div className="chart-wrapper" style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                    <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                        <YAxis domain={yAxisDomain} tick={{ fontSize: 12 }} />
                        <Tooltip
                            labelFormatter={(value) => `Epoch ${value}`}
                            contentStyle={TOOLTIP_STYLE}
                        />
                        <Legend />
                        <Line
                            type="monotone"
                            dataKey={primaryLine.dataKey}
                            stroke={primaryLine.color}
                            strokeWidth={2}
                            dot={false}
                            name={primaryLine.name}
                        />
                        <Line
                            type="monotone"
                            dataKey={secondaryLine.dataKey}
                            stroke={secondaryLine.color}
                            strokeWidth={2}
                            dot={false}
                            name={secondaryLine.name}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        ) : (
            <div className="chart-placeholder">
                Waiting for training data...
                <small>{placeholderHint}</small>
            </div>
        )}
    </div>
);
