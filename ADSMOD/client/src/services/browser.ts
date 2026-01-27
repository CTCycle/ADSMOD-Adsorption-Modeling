import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, extractErrorMessage, HTTP_TIMEOUT } from './http';

export async function fetchTableList(): Promise<{ tables: { table_name: string; display_name: string; category: string }[]; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/browser/tables`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { tables: [], error: message };
        }

        const data = await response.json();
        return { tables: data.tables || [], error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { tables: [], error: error.message };
        }
        return { tables: [], error: 'An unknown error occurred.' };
    }
}

export async function fetchTableData(
    tableName: string,
    limit = 50,
    offset = 0
): Promise<{
    data: Record<string, unknown>[];
    columns: string[];
    rowCount: number;
    totalRows: number;
    columnCount: number;
    displayName: string;
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/browser/data/${encodeURIComponent(tableName)}?limit=${limit}&offset=${offset}`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return {
                data: [],
                columns: [],
                rowCount: 0,
                totalRows: 0,
                columnCount: 0,
                displayName: '',
                error: message,
            };
        }

        const result = await response.json();
        return {
            data: result.data || [],
            columns: result.columns || [],
            rowCount: result.row_count || 0,
            totalRows: result.total_rows || 0,
            columnCount: result.column_count || 0,
            displayName: result.display_name || tableName,
            error: null,
        };
    } catch (error) {
        if (error instanceof Error) {
            return {
                data: [],
                columns: [],
                rowCount: 0,
                totalRows: 0,
                columnCount: 0,
                displayName: '',
                error: error.message,
            };
        }
        return {
            data: [],
            columns: [],
            rowCount: 0,
            totalRows: 0,
            columnCount: 0,
            displayName: '',
            error: 'An unknown error occurred.',
        };
    }
}
