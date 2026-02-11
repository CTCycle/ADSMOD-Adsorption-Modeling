import React from 'react';

export type PageId = 'config' | 'models' | 'analysis';

interface SidebarProps {
    currentPage: PageId;
    onPageChange: (page: PageId) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onPageChange }) => {
    return (
        <div className="sidebar">
            <button
                className={`sidebar-icon ${currentPage === 'config' ? 'active' : ''}`}
                onClick={() => onPageChange('config')}
                title="Data Source"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 12V8H6a2 2 0 0 1-2-2c0-1.1.9-2 2-2h12v4" />
                    <path d="M4 6v12c0 1.1.9 2 2 2h14v-4" />
                    <path d="M18 12a2 2 0 0 0-2 2c0 1.1.9 2 2 2h4v-4h-4z" />
                </svg>
                <span className="sidebar-label">Sources</span>
            </button>
            <button
                className={`sidebar-icon ${currentPage === 'models' ? 'active' : ''}`}
                onClick={() => onPageChange('models')}
                title="Models & Fitting"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 3v18h18" />
                    <path d="M18 17V9" />
                    <path d="M13 17V5" />
                    <path d="M8 17v-3" />
                </svg>
                <span className="sidebar-label">Fitting</span>
            </button>

            <button
                className={`sidebar-icon ${currentPage === 'analysis' ? 'active' : ''}`}
                onClick={() => onPageChange('analysis')}
                title="Analysis"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5" />
                    <path d="M8.5 8.5v.01" />
                    <path d="M16 16v.01" />
                    <path d="M12 12v.01" />
                </svg>
                <span className="sidebar-label">Training</span>
            </button>

        </div>
    );
};

