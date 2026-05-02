export const HTTP_TIMEOUT = 120000; // 120 seconds

export async function fetchWithTimeout(url: string, options: RequestInit, timeout: number): Promise<Response> {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

export function extractErrorMessage(response: Response, data: unknown): string {
    if (typeof data === 'object' && data !== null) {
        const obj = data as Record<string, unknown>;
        if (typeof obj.detail === 'string' && obj.detail) {
            return obj.detail;
        }
        if (typeof obj.message === 'string' && obj.message) {
            return obj.message;
        }
    }
    return `HTTP error ${response.status}`;
}
