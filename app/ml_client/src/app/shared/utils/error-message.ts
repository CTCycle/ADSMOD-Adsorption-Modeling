export function extractErrorMessage(response: Response, data: unknown): string {
    if (typeof data === 'object' && data !== null) {
        const obj = data as Record<string, unknown>;
        const detailValue = obj['detail'];
        const messageValue = obj['message'];

        if (typeof detailValue === 'string' && detailValue) {
            return detailValue;
        }
        if (Array.isArray(detailValue) && detailValue.length > 0) {
            return detailValue
                .map((entry) => {
                    if (typeof entry !== 'object' || entry === null) {
                        return String(entry);
                    }

                    const detail = entry as Record<string, unknown>;
                    const locationValue = detail['loc'];
                    const detailMessageValue = detail['msg'];
                    const location = Array.isArray(locationValue) ? locationValue.join('.') : '';
                    const message = typeof detailMessageValue === 'string' ? detailMessageValue : JSON.stringify(detail);
                    return location ? `${location}: ${message}` : message;
                })
                .join('; ');
        }
        if (typeof messageValue === 'string' && messageValue) {
            return messageValue;
        }
    }

    return `HTTP error ${response.status}`;
}
