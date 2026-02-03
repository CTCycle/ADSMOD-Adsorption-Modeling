import path from 'path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
    const envDir = path.resolve(__dirname, '../settings')
    const settingsEnv = loadEnv(mode, envDir, '')
    const localEnv = loadEnv(mode, __dirname, '')
    const env = { ...localEnv, ...settingsEnv }

    const uiHost = env.UI_HOST || '127.0.0.1'
    const uiPort = Number(env.UI_PORT || 7861)
    const apiHost = env.FASTAPI_HOST || '127.0.0.1'
    const apiPort = Number(env.FASTAPI_PORT || 8000)
    const apiTarget = `http://${apiHost}:${apiPort}`

    return {
        envDir,
        plugins: [react()],
        server: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                },
            },
        },
        preview: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                },
            },
        },
    }
})
