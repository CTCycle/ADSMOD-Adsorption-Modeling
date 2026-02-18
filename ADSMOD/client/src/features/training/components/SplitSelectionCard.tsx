import React from 'react';

interface SplitSelectionCardProps {
    title: string;
    subtitle?: string;
    refreshLabel?: string;
    onRefresh?: () => void;
    leftContent: React.ReactNode;
    rightContent: React.ReactNode;
}

export const SplitSelectionCard: React.FC<SplitSelectionCardProps> = ({
    title,
    subtitle,
    refreshLabel = 'Refresh',
    onRefresh,
    leftContent,
    rightContent,
}) => {
    return (
        <div className="section-container">
            <h3 className="split-selection-title">{title}</h3>
            {subtitle && <p className="split-selection-subtitle">{subtitle}</p>}
            <div className="split-selection-card">
                <div className="split-selection-card-left">
                    <div className="split-selection-card-toolbar">
                        {onRefresh && (
                            <button
                                type="button"
                                className="split-selection-refresh-button"
                                onClick={(event) => {
                                    event.stopPropagation();
                                    onRefresh();
                                }}
                            >
                                🔄 {refreshLabel}
                            </button>
                        )}
                    </div>
                    <div className="split-selection-card-content">{leftContent}</div>
                </div>
                <div className="split-selection-card-right">{rightContent}</div>
            </div>
        </div>
    );
};

export default SplitSelectionCard;
