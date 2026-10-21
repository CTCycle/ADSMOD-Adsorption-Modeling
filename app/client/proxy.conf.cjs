// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

const { loadRuntimeSettings } = require('./scripts/runtime-settings.cjs');
const runtime = loadRuntimeSettings();
const backendTarget = runtime.backend_url;

module.exports = {
  '/api/v1': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' },
  '/health': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' },
};
