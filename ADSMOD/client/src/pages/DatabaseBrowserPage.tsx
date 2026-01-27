import React, { useEffect, useCallback, useRef } from 'react';
import { fetchTableList, fetchTableData } from '../services';

interface TableInfo {
    table_name: string;
    display_name: string;
    category: string;
}

export interface DatabaseBrowserState {
    tables: TableInfo[];
    selectedTable: string;
    tableData: Record<string, unknown>[];
    columns: string[];
    rowCount: number;
    totalRows: number;
    columnCount: number;
    displayName: string;
    loading: boolean;
    lazyLoading: boolean;
    error: string | null;
    tablesLoaded: boolean;
    nextOffset: number;
    hasMore: boolean;
}

export const initialDatabaseBrowserState: DatabaseBrowserState = {
    tables: [],
    selectedTable: '',
    tableData: [],
    columns: [],
    rowCount: 0,
    totalRows: 0,
    columnCount: 0,
    displayName: '',
    loading: false,
    lazyLoading: false,
    error: null,
    tablesLoaded: false,
    nextOffset: 0,
    hasMore: false,
};

const PAGE_SIZE = 50;

interface DatabaseBrowserPageProps {
    state: DatabaseBrowserState;
    onStateChange: (state: DatabaseBrowserState) => void;
}

export const DatabaseBrowserPage: React.FC<DatabaseBrowserPageProps> = ({ state, onStateChange }) => {
    const {
        tables,
        selectedTable,
        tableData,
        columns,
        rowCount,
        totalRows,
        columnCount,
        displayName,
        loading,
        lazyLoading,
        error,
        tablesLoaded,
        nextOffset,
        hasMore,
    } = state;

    const tableScrollRef = useRef<HTMLDivElement | null>(null);
    const loadMoreRef = useRef<HTMLDivElement | null>(null);

    const updateState = useCallback((updates: Partial<DatabaseBrowserState>) => {
        onStateChange({ ...state, ...updates });
    }, [state, onStateChange]);

    const clearTableSelection = useCallback((nextSelectedTable: string) => {
        onStateChange({
            ...state,
            selectedTable: nextSelectedTable,
            tableData: [],
            columns: [],
            rowCount: 0,
            columnCount: 0,
            totalRows: 0,
            displayName: '',
            loading: false,
            lazyLoading: false,
            error: null,
            nextOffset: 0,
            hasMore: false,
        });
    }, [state, onStateChange]);

    // Fetch table list on mount (only once)
    useEffect(() => {
        if (tablesLoaded) return;

        const loadTables = async () => {
            const result = await fetchTableList();
            if (result.error) {
                updateState({ error: result.error, tablesLoaded: true });
            } else {
                updateState({
                    tables: result.tables,
                    selectedTable: '',
                    tableData: [],
                    columns: [],
                    rowCount: 0,
                    columnCount: 0,
                    totalRows: 0,
                    displayName: '',
                    loading: false,
                    lazyLoading: false,
                    error: null,
                    tablesLoaded: true,
                    nextOffset: 0,
                    hasMore: false,
                });
            }
        };
        loadTables();
    }, [tablesLoaded, updateState]);

    // Fetch table data function
    const loadTableData = useCallback(async (tableName: string, offset = 0) => {
        if (!tableName) return;

        if (offset === 0) {
            onStateChange({
                ...state,
                selectedTable: tableName,
                loading: true,
                lazyLoading: false,
                error: null,
                tableData: [],
                totalRows: 0,
                rowCount: 0,
                nextOffset: 0,
                hasMore: true,
            });
        } else {
            updateState({ lazyLoading: true });
        }

        const result = await fetchTableData(tableName, PAGE_SIZE, offset);

        if (result.error) {
            onStateChange({
                ...state,
                selectedTable: tableName,
                error: result.error,
                loading: false,
                lazyLoading: false,
            });
        } else {
            const newData = offset === 0 ? result.data : [...state.tableData, ...result.data];
            const fetchedCount = result.data.length;
            const computedOffset = offset + fetchedCount;
            const resolvedHasMore = result.totalRows > 0
                ? computedOffset < result.totalRows
                : fetchedCount === PAGE_SIZE;
            onStateChange({
                ...state,
                selectedTable: tableName,
                tableData: newData,
                columns: result.columns,
                rowCount: newData.length,
                totalRows: result.totalRows, // Note: result.totalRows comes from backend
                columnCount: result.columnCount,
                displayName: result.displayName,
                loading: false,
                lazyLoading: false,
                nextOffset: computedOffset,
                hasMore: resolvedHasMore,
            });
        }
    }, [state, onStateChange, updateState]);

    // Fetch data when table selection changes
    const handleTableChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newTable = e.target.value;
        if (!newTable) {
            clearTableSelection('');
            return;
        }
        loadTableData(newTable, 0);
    };

    const handleRefresh = () => {
        if (!selectedTable) {
            clearTableSelection('');
            return;
        }
        loadTableData(selectedTable, 0);
    };

    const requestNextPage = useCallback(() => {
        if (!selectedTable || loading || lazyLoading || !hasMore) {
            return;
        }
        loadTableData(selectedTable, nextOffset);
    }, [selectedTable, loading, lazyLoading, hasMore, nextOffset, loadTableData]);

    const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
        if (scrollHeight - scrollTop <= clientHeight + 50) {
            // Near bottom
            requestNextPage();
        }
    }, [requestNextPage]);

    useEffect(() => {
        const loadTarget = loadMoreRef.current;
        if (!loadTarget) {
            return;
        }

        const observer = new IntersectionObserver(
            (entries) => {
                if (entries.some((entry) => entry.isIntersecting)) {
                    requestNextPage();
                }
            },
            {
                root: tableScrollRef.current,
                rootMargin: '200px',
                threshold: 0.1,
            }
        );

        observer.observe(loadTarget);

        return () => {
            observer.disconnect();
        };
    }, [requestNextPage]);

    const emptyMessage = selectedTable
        ? 'No data available in this table.'
        : 'Select data to view table contents.';

    const tableLabel = selectedTable ? displayName : 'Select data';

    return (
        <div className="browser-page">
            <div className="browser-header">
                <div className="browser-title-section">

                    <h1 className="browser-title">Database Browser</h1>
                    <p className="browser-subtitle">
                        Browse model fitting results and uploaded adsorption data.
                    </p>
                </div>
            </div>

            <div className="browser-controls">
                <div className="browser-select-group">
                    <label className="browser-select-label">Select Table</label>
                    <div className="browser-select-row">
                        <select
                            className="select-input browser-select"
                            value={selectedTable}
                            onChange={handleTableChange}
                            disabled={loading && !lazyLoading}
                        >
                            <option value="">Select data</option>
                            {Object.entries(
                                tables.reduce((acc, table) => {
                                    (acc[table.category] ??= []).push(table);
                                    return acc;
                                }, {} as Record<string, TableInfo[]>)
                            ).map(([category, categoryTables]) => (
                                <optgroup key={category} label={category}>
                                    {categoryTables.map((table) => (
                                        <option key={table.table_name} value={table.table_name}>
                                            {table.display_name}
                                        </option>
                                    ))}
                                </optgroup>
                            ))}
                        </select>
                        <button
                            className="browser-refresh-btn"
                            onClick={handleRefresh}
                            disabled={(loading && !lazyLoading) || !selectedTable}
                            title="Refresh data"
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="23 4 23 10 17 10" />
                                <polyline points="1 20 1 14 7 14" />
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                            </svg>
                        </button>
                    </div>
                </div>

                <div className="browser-stats">
                    <span className="browser-stat-label">Statistics</span>
                    <span className="browser-stat-item">Rows: <strong>{totalRows > 0 ? `${rowCount} / ${totalRows}` : rowCount}</strong></span>
                    <span className="browser-stat-item">Columns: <strong>{columnCount}</strong></span>
                    <span className="browser-stat-item">Table: <strong className="browser-stat-table">{tableLabel}</strong></span>
                </div>
            </div>

            {error && (
                <div className="browser-error">
                    {error}
                </div>
            )}

            <div className="browser-table-container">
                {loading && !tableData.length ? (
                    <div className="browser-loading">
                        <div className="browser-spinner"></div>
                        <span>Loading data...</span>
                    </div>
                ) : tableData.length > 0 ? (
                    <div className="browser-table-scroll" onScroll={handleScroll} ref={tableScrollRef}>
                        <table className="browser-table">
                            <thead>
                                <tr>
                                    {columns.map((col) => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {tableData.map((row, idx) => (
                                    <tr key={idx}>
                                        {columns.map((col) => (
                                            <td key={col}>
                                                {row[col] !== null && row[col] !== undefined
                                                    ? String(row[col])
                                                    : ''}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                                {lazyLoading && (
                                    <tr>
                                        <td colSpan={columns.length} style={{ textAlign: 'center', padding: '10px' }}>
                                            <div className="browser-spinner" style={{ display: 'inline-block', width: '20px', height: '20px', border: '2px solid rgba(0,0,0,0.1)', borderLeftColor: 'var(--primary-color)' }}></div>
                                        </td>
                                    </tr>
                                )}
                                {columns.length > 0 && (
                                    <tr className="browser-scroll-sentinel-row">
                                        <td colSpan={columns.length}>
                                            <div ref={loadMoreRef} className="browser-scroll-sentinel"></div>
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="browser-empty">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <ellipse cx="12" cy="5" rx="9" ry="3" />
                            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                        </svg>
                        <p>{emptyMessage}</p>
                    </div>
                )}
            </div>
        </div>
    );
};
