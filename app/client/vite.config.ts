import path from 'path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
    const envDir = path.resolve(__dirname, '../../settings')
    const settingsEnv = loadEnv(mode, envDir, '')
    const localEnv = loadEnv(mode, __dirname, '')
    const env = { ...localEnv, ...settingsEnv }

    const uiHost = env.UI_HOST || '127.0.0.1'
    const uiPort = Number(env.UI_PORT || 5173)
    const coreApiHost = env.CORE_SERVICE_HOST || env.FASTAPI_HOST || '127.0.0.1'
    const coreApiPort = Number(env.CORE_SERVICE_PORT || env.FASTAPI_PORT || 8000)
    const coreApiTarget = `http://${coreApiHost}:${coreApiPort}`

    return {
        envDir,
        plugins: [react()],
        server: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api': {
                    target: coreApiTarget,
                    changeOrigin: true,
                },
            },
        },
        preview: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api': {
                    target: coreApiTarget,
                    changeOrigin: true,
                },
            },
        },
    }
})
