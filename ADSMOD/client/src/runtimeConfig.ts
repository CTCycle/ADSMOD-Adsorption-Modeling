import { setApiBaseUrl } from './constants';

type TauriInvoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

type DesktopRuntimeConfig = {
    apiOrigin?: string;
    api_origin?: string;
};

function getTauriInvoke(): TauriInvoke | null {
    if (typeof window === 'undefined') {
        return null;
    }

    const tauriCore = window.__TAURI__?.core;
    if (!tauriCore || typeof tauriCore.invoke !== 'function') {
        return null;
    }

    return tauriCore.invoke as TauriInvoke;
}

export function isDesktopRuntime(): boolean {
    return getTauriInvoke() !== null;
}

export async function initializeRuntimeConfig(): Promise<void> {
    const invoke = getTauriInvoke();
    if (!invoke) {
        return;
    }

    try {
        const runtime = await invoke<DesktopRuntimeConfig>('get_runtime_config');
        const apiOrigin = runtime.apiOrigin || runtime.api_origin;
        if (typeof apiOrigin === 'string' && apiOrigin.trim()) {
            setApiBaseUrl(apiOrigin);
        }
    } catch (error) {
        console.error('Failed to resolve desktop runtime config', error);
    }
}
