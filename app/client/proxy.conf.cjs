const fs = require('node:fs');
const path = require('node:path');

const repositoryRoot = path.resolve(__dirname, '../..');
const configuredResourcesDir = String(process.env.ADSMOD_RESOURCES_DIR || '').trim();
const resourcesDir = configuredResourcesDir
  ? path.resolve(repositoryRoot, configuredResourcesDir)
  : path.join(repositoryRoot, 'resources');
const canonicalConfig = JSON.parse(fs.readFileSync(path.join(resourcesDir, 'adsmod.json'), 'utf8'));
const runtime = canonicalConfig.runtime;
const backendTarget = `http://${runtime.host}:${Number(runtime.backend_port)}`;

module.exports = {
  '/api/v1': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' },
  '/health': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' }
};
