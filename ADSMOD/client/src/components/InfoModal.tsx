
import React from 'react';
import './InfoModal.css';
import type { InfoModalData, InfoModalValue } from '../types';

interface InfoModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    data: InfoModalData | null;
}

const IconTag = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>
);

const IconCalendar = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
);

const IconDatabase = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
);

const IconActivity = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
);

const IconMaximize = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>
);

const InfoRow: React.FC<{ label: string; value: InfoModalValue; icon: React.ReactNode }> = ({
    label,
    value,
    icon,
}) => {
    const isObject = typeof value === 'object' && value !== null;
    const isFullWidth = isObject || label.length > 20;

    return (
        <div className={`info-row ${isFullWidth ? 'full-width' : ''}`}>
            <div className="info-header-row">
                <div className="info-icon-wrapper">
                    {icon}
                </div>
                <span className="info-label">{label.replace(/_/g, ' ')}</span>
            </div>
            {isObject ? (
                <div className="info-object-container">
                    <pre>{JSON.stringify(value, null, 2)}</pre>
                </div>
            ) : (
                <span className="info-value">
                    {typeof value === 'number' && !Number.isInteger(value)
                        ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
                        : `${value}`}
                </span>
            )}
        </div>
    );
};

export const InfoModal: React.FC<InfoModalProps> = ({ isOpen, onClose, title, data }) => {
    if (!isOpen || !data) return null;

    const getIcon = (key: string) => {
        const lowerKey = key.toLowerCase();
        if (lowerKey.includes('label') || lowerKey.includes('name')) return <IconTag />;
        if (lowerKey.includes('created') || lowerKey.includes('date')) return <IconCalendar />;
        if (lowerKey.includes('samples') || lowerKey.includes('count')) return <IconDatabase />;
        if (lowerKey.includes('fraction') || lowerKey.includes('size')) return <IconActivity />;
        if (lowerKey.includes('max') || lowerKey.includes('min') || lowerKey.includes('length')) return <IconMaximize />;
        return <IconDatabase />;
    };

    return (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
            <div className="info-modal-content">
                <div className="info-modal-header">
                    <h4>{title}</h4>
                    <button className="info-modal-close-btn" onClick={onClose} aria-label="Close">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>

                <div className="info-modal-body">
                    {Object.entries(data)
                        .filter(([, value]) => value !== null && value !== undefined)
                        .map(([key, value]) => (
                            <InfoRow
                                key={key}
                                label={key}
                                value={value}
                                icon={getIcon(key)}
                            />
                        ))}
                </div>

                <div className="info-modal-footer">
                    <button className="primary" onClick={onClose}>
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};
