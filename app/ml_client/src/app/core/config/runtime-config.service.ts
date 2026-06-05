import { normalizeApiBaseUrl } from './api-base-url';

interface RuntimeConfigWindow {
    __ADSMOD_RUNTIME_CONFIG__?: {
        apiBaseUrl?: string;
    };
}

const getRuntimeConfigWindow = (): RuntimeConfigWindow | undefined =>
    typeof window === 'undefined' ? undefined : (window as unknown as RuntimeConfigWindow);

export const getRuntimeApiBaseUrl = (): string => {
    const runtimeApiBaseUrl = getRuntimeConfigWindow()?.__ADSMOD_RUNTIME_CONFIG__?.apiBaseUrl;
    return normalizeApiBaseUrl(runtimeApiBaseUrl || '');
};
