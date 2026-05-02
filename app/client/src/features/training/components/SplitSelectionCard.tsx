import React from 'react';

interface SplitSelectionCardProps {
    title: string;
    subtitle?: string;
    onRefresh?: () => void;
    leftContent: React.ReactNode;
    rightContent: React.ReactNode;
    hideHeader?: boolean;
}

export const SplitSelectionCard: React.FC<SplitSelectionCardProps> = ({
    title,
    subtitle,
    onRefresh,
    leftContent,
    rightContent,
    hideHeader = false,
}) => {
    return (
        <div className="section-container">
            {!hideHeader && <h3 className="split-selection-title">{title}</h3>}
            {!hideHeader && subtitle && <p className="split-selection-subtitle">{subtitle}</p>}
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
                                Refresh
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
