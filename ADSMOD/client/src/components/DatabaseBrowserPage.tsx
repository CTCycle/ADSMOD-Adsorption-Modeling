import React, { useEffect, useCallback } from 'react';
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
    columnCount: number;
    displayName: string;
    loading: boolean;
    error: string | null;
    tablesLoaded: boolean;
}

export const initialDatabaseBrowserState: DatabaseBrowserState = {
    tables: [],
    selectedTable: '',
    tableData: [],
    columns: [],
    rowCount: 0,
    columnCount: 0,
    displayName: '',
    loading: false,
    error: null,
    tablesLoaded: false,
};

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
        columnCount,
        displayName,
        loading,
        error,
        tablesLoaded,
    } = state;

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
            displayName: '',
            loading: false,
            error: null,
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
                    displayName: '',
                    loading: false,
                    error: null,
                    tablesLoaded: true,
                });
            }
        };
        loadTables();
    }, [tablesLoaded, updateState]);

    // Fetch table data function
    const loadTableData = useCallback(async (tableName: string) => {
        if (!tableName) return;

        onStateChange({ ...state, selectedTable: tableName, loading: true, error: null });

        const result = await fetchTableData(tableName);

        if (result.error) {
            onStateChange({
                ...state,
                selectedTable: tableName,
                error: result.error,
                tableData: [],
                columns: [],
                rowCount: 0,
                columnCount: 0,
                displayName: '',
                loading: false,
            });
        } else {
            onStateChange({
                ...state,
                selectedTable: tableName,
                tableData: result.data,
                columns: result.columns,
                rowCount: result.rowCount,
                columnCount: result.columnCount,
                displayName: result.displayName,
                loading: false,
            });
        }
    }, [state, onStateChange]);

    // Fetch data when table selection changes
    const handleTableChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newTable = e.target.value;
        if (!newTable) {
            clearTableSelection('');
            return;
        }
        loadTableData(newTable);
    };

    const handleRefresh = () => {
        if (!selectedTable) {
            clearTableSelection('');
            return;
        }
        loadTableData(selectedTable);
    };

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
                            disabled={loading}
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
                            disabled={loading || !selectedTable}
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
                    <span className="browser-stat-item">Rows: <strong>{rowCount}</strong></span>
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
                {loading ? (
                    <div className="browser-loading">
                        <div className="browser-spinner"></div>
                        <span>Loading data...</span>
                    </div>
                ) : tableData.length > 0 ? (
                    <div className="browser-table-scroll">
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
