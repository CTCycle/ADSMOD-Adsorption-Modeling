import { useCallback, useState } from 'react';

interface UseWizardPaginationResult {
    currentPage: number;
    isFirstPage: boolean;
    isLastPage: boolean;
    goToNextPage: () => void;
    goToPreviousPage: () => void;
}

export const useWizardPagination = (
    lastPageIndex: number
): UseWizardPaginationResult => {
    const [currentPage, setCurrentPage] = useState(0);

    const goToNextPage = useCallback(() => {
        setCurrentPage((previousPage) => Math.min(previousPage + 1, lastPageIndex));
    }, [lastPageIndex]);

    const goToPreviousPage = useCallback(() => {
        setCurrentPage((previousPage) => Math.max(previousPage - 1, 0));
    }, []);

    return {
        currentPage,
        isFirstPage: currentPage === 0,
        isLastPage: currentPage === lastPageIndex,
        goToNextPage,
        goToPreviousPage,
    };
};
