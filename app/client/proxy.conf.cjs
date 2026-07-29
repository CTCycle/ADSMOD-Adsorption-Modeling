const fs = require('node:fs');
const path = require('node:path');

const canonicalConfig = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../resources/adsmod.json'), 'utf8')
);
const runtime = canonicalConfig.runtime;
const coreApiHost = runtime.host;
const coreApiPort = Number(runtime.core_port);
const mlApiHost = runtime.host;
const mlApiPort = Number(runtime.ml_port);

module.exports = {
  '/api/training': {
    target: `http://${mlApiHost}:${mlApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn'
  },
  '/api': {
    target: `http://${coreApiHost}:${coreApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn'
  }
};
