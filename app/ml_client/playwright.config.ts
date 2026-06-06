import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests/visual',
    outputDir: '../../QA/ml-visual-artifacts',
    fullyParallel: false,
    workers: 1,
    reporter: [['line']],
    snapshotPathTemplate: '{testDir}/__snapshots__/{projectName}/{testFileName}/{arg}{ext}',
    projects: [
        { name: 'tauri-1440x920', use: { viewport: { width: 1440, height: 920 } } },
        { name: 'wide-1480x920', use: { viewport: { width: 1480, height: 920 } } },
        { name: 'desktop-1360x900', use: { viewport: { width: 1360, height: 900 } } },
        { name: 'compact-1200x900', use: { viewport: { width: 1200, height: 900 } } },
        { name: 'tablet-900x900', use: { viewport: { width: 900, height: 900 } } },
        { name: 'narrow-768x900', use: { viewport: { width: 768, height: 900 } } },
        { name: 'mobile-600x900', use: { viewport: { width: 600, height: 900 } } },
    ],
    use: {
        baseURL: 'http://127.0.0.1:5174',
        channel: 'msedge',
        colorScheme: 'light',
        locale: 'en-US',
        deviceScaleFactor: 1,
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure',
        video: 'off',
    },
    webServer: {
        command: 'npm run dev',
        url: 'http://127.0.0.1:5174/training/processing',
        reuseExistingServer: !process.env.CI,
        stdout: 'pipe',
        stderr: 'pipe',
        timeout: 180000,
    },
});
