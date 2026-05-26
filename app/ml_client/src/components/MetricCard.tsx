import React from 'react';

interface MetricCardProps {
    label: string;
    value: string;
    icon?: string;
    color?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
    label,
    value,
    icon,
    color = 'var(--primary-600)',
}) => (
    <div className="metric-card">
        <div className="metric-label">{icon && <span className="metric-icon">{icon}</span>}{label}</div>
        <div className="metric-value" style={{ color }}>{value}</div>
    </div>
);
