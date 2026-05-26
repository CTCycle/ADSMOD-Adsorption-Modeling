import path from 'path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
    const envDir = path.resolve(__dirname, '../../settings')
    const settingsEnv = loadEnv(mode, envDir, '')
    const localEnv = loadEnv(mode, __dirname, '')
    const env = { ...localEnv, ...settingsEnv }

    const uiHost = env.ML_UI_HOST || env.UI_HOST || '127.0.0.1'
    const uiPort = Number(env.ML_UI_PORT || 5174)
    const mlApiHost = env.ML_SERVICE_HOST || '127.0.0.1'
    const mlApiPort = Number(env.ML_SERVICE_PORT || 8001)
    const mlApiTarget = `http://${mlApiHost}:${mlApiPort}`

    return {
        envDir,
        plugins: [react()],
        server: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api/training': {
                    target: mlApiTarget,
                    changeOrigin: true,
                },
            },
        },
        preview: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api/training': {
                    target: mlApiTarget,
                    changeOrigin: true,
                },
            },
        },
    }
})
