import React from 'react';

interface WizardProgressIndicatorProps {
    currentPage: number;
    totalPages: number;
}

export const WizardProgressIndicator: React.FC<WizardProgressIndicatorProps> = ({
    currentPage,
    totalPages,
}) => {
    const pageNumbers = Array.from({ length: totalPages }, (_, index) => index + 1);

    return (
        <div className="wizard-page-indicator">
            {pageNumbers.map((pageNumber, index) => (
                <React.Fragment key={pageNumber}>
                    <span className={`wizard-dot ${currentPage === index ? 'active' : ''}`}>
                        {pageNumber}
                    </span>
                    {index < pageNumbers.length - 1 && (
                        <span className={`wizard-dot-line ${currentPage > index ? 'active' : ''}`} />
                    )}
                </React.Fragment>
            ))}
        </div>
    );
};
