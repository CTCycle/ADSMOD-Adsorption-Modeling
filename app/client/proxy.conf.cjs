// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');

const repositoryRoot = path.resolve(__dirname, '../..');
const configuredDataDir = String(process.env.ADSMOD_DATA_DIR || '').trim();
const dataDir = configuredDataDir
  ? path.resolve(repositoryRoot, configuredDataDir)
  : path.join(repositoryRoot, 'data');
const canonicalConfig = JSON.parse(fs.readFileSync(path.join(dataDir, 'adsmod.json'), 'utf8'));
const runtime = canonicalConfig.runtime;
const backendTarget = `http://${runtime.host}:${Number(runtime.backend_port)}`;

module.exports = {
  '/api/v1': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' },
  '/health': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' }
};
