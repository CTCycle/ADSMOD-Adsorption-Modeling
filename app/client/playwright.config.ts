// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { defineConfig } from '@playwright/test';

const frontendPort = Number(process.env.ADSMOD_FRONTEND_TEST_PORT ?? 5173);

export default defineConfig({
    testDir: './tests/visual',
    outputDir: '../../assets/QA/core-visual-artifacts',
    fullyParallel: false,
    workers: 1,
    reporter: [['line']],
    projects: [
        { name: 'viewport-1440x920', use: { viewport: { width: 1440, height: 920 } } },
        { name: 'known-overflow-1280x720', use: { viewport: { width: 1280, height: 720 } } },
        { name: 'tablet-900x900', use: { viewport: { width: 900, height: 900 } } },
        { name: 'mobile-600x900', use: { viewport: { width: 600, height: 900 } } },
    ],
    use: {
        baseURL: `http://127.0.0.1:${frontendPort}`,
        channel: 'msedge',
        colorScheme: 'light',
        locale: 'en-US',
        deviceScaleFactor: 1,
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure',
        video: 'off',
    },
    webServer: {
        command: `npm run dev -- --port ${frontendPort}`,
        url: `http://127.0.0.1:${frontendPort}/datasets`,
        reuseExistingServer: !process.env.CI,
        stdout: 'pipe',
        stderr: 'pipe',
        timeout: 180000,
    },
});
