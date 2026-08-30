const fs = require('node:fs');
const path = require('node:path');

const repositoryRoot = path.resolve(__dirname, '../..');
const canonicalConfig = JSON.parse(
  fs.readFileSync(path.join(repositoryRoot, 'app', 'resources', 'adsmod.json'), 'utf8')
);
const runtime = canonicalConfig.runtime;
const coreApiHost = runtime.host;
const coreApiPort = Number(runtime.core_port);
const mlApiHost = runtime.host;
const mlApiPort = Number(runtime.ml_port);

module.exports = {
  '/api/v1/training': {
    target: `http://${mlApiHost}:${mlApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn'
  },
  '/api/v1': {
    target: `http://${coreApiHost}:${coreApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn'
  },
  '/ml-health': {
    target: `http://${mlApiHost}:${mlApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn',
    pathRewrite: { '^/ml-health': '/health' }
  },
  '/health': {
    target: `http://${coreApiHost}:${coreApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn'
  }
};
