import React from 'react';

export type PageId = 'source' | 'fitting' | 'training';

interface SidebarProps {
    currentPage: PageId;
    onPageChange: (page: PageId) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onPageChange }) => {
    return (
        <nav className="header-tabs" aria-label="Main navigation">
            <button
                className={`header-tab ${currentPage === 'source' ? 'active' : ''}`}
                onClick={() => onPageChange('source')}
                title="Source"
                type="button"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 12V8H6a2 2 0 0 1-2-2c0-1.1.9-2 2-2h12v4" />
                    <path d="M4 6v12c0 1.1.9 2 2 2h14v-4" />
                    <path d="M18 12a2 2 0 0 0-2 2c0 1.1.9 2 2 2h4v-4h-4z" />
                </svg>
                <span className="header-tab-label">source</span>
            </button>
            <button
                className={`header-tab ${currentPage === 'fitting' ? 'active' : ''}`}
                onClick={() => onPageChange('fitting')}
                title="Fitting"
                type="button"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 3v18h18" />
                    <path d="M18 17V9" />
                    <path d="M13 17V5" />
                    <path d="M8 17v-3" />
                </svg>
                <span className="header-tab-label">fitting</span>
            </button>

            <button
                className={`header-tab ${currentPage === 'training' ? 'active' : ''}`}
                onClick={() => onPageChange('training')}
                title="Training"
                type="button"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5" />
                    <path d="M8.5 8.5v.01" />
                    <path d="M16 16v.01" />
                    <path d="M12 12v.01" />
                </svg>
                <span className="header-tab-label">training</span>
            </button>
        </nav>
    );
};
