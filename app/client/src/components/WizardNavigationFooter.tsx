import React from 'react';

interface WizardNavigationFooterProps {
    isLoading: boolean;
    isFirstPage: boolean;
    isLastPage: boolean;
    onClose: () => void;
    onNext: () => void;
    onPrevious: () => void;
    onConfirm: () => void;
    confirmIdleLabel: string;
    confirmLoadingLabel: string;
}

export const WizardNavigationFooter: React.FC<WizardNavigationFooterProps> = ({
    isLoading,
    isFirstPage,
    isLastPage,
    onClose,
    onNext,
    onPrevious,
    onConfirm,
    confirmIdleLabel,
    confirmLoadingLabel,
}) => (
    <div className="wizard-footer">
        <button className="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
        </button>
        {!isFirstPage && (
            <button className="secondary" onClick={onPrevious} disabled={isLoading}>
                Previous
            </button>
        )}
        {!isLastPage && (
            <button className="primary" onClick={onNext} disabled={isLoading}>
                Next
            </button>
        )}
        {isLastPage && (
            <button className="primary" onClick={onConfirm} disabled={isLoading}>
                {isLoading ? confirmLoadingLabel : confirmIdleLabel}
            </button>
        )}
    </div>
);
