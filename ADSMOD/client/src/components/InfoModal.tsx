
import React from 'react';

interface InfoModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    data: Record<string, any> | null;
}

export const InfoModal: React.FC<InfoModalProps> = ({ isOpen, onClose, title, data }) => {
    if (!isOpen || !data) return null;

    const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
        if (event.target === event.currentTarget) {
            onClose();
        }
    };

    return (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={handleBackdropClick}>
            <div className="wizard-modal" style={{ maxWidth: '400px', width: '90%' }}>
                <div className="wizard-header" style={{ borderBottom: '1px solid var(--slate-200)', paddingBottom: '16px', marginBottom: '16px' }}>
                    <h4 style={{ margin: 0 }}>{title}</h4>
                </div>

                <div className="wizard-body" style={{ padding: '0' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {Object.entries(data).map(([key, value]) => (
                            <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--slate-100)', paddingBottom: '8px' }}>
                                <span style={{ fontWeight: 600, color: 'var(--slate-600)', fontSize: '0.9rem', textTransform: 'capitalize' }}>
                                    {key.replace(/_/g, ' ')}
                                </span>
                                <span style={{ color: 'var(--slate-800)', fontSize: '0.9rem', maxWidth: '60%', textAlign: 'right', wordBreak: 'break-word' }}>
                                    {String(value)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="wizard-footer" style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--slate-100)', justifyContent: 'flex-end', display: 'flex' }}>
                    <button className="primary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};
