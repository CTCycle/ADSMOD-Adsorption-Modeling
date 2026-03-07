/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}

interface TauriCoreApi {
    invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
}

interface Window {
    __TAURI__?: {
        core?: TauriCoreApi;
    };
}
