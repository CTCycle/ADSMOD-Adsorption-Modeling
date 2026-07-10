export interface DatasetMetadata { tags: string[]; description: string; }
export interface DatasetSummary { name: string; source: 'uploaded'; created_at: string; row_count: number; column_count: number; tags: string[]; description: string; }
export interface DatasetRowsPage { dataset_name: string; columns: string[]; rows: DatasetRow[]; offset: number; limit: number; total_rows: number; }
export interface DatasetRow { row_id: number; [key: string]: unknown; }
export interface DatasetRowMutation { operation: 'insert' | 'update' | 'delete'; row_id?: number; values?: Record<string, unknown>; }
export interface DatasetUploadResponse { status: string; dataset: DatasetSummary; summary: string; }