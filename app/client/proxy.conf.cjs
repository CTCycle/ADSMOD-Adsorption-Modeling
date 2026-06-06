const fs = require('node:fs');
const path = require('node:path');

const readEnvFile = (filePath) => {
  if (!fs.existsSync(filePath)) {
    return {};
  }

  return fs
    .readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .reduce((acc, line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith(';')) {
        return acc;
      }

      const separatorIndex = trimmed.indexOf('=');
      if (separatorIndex < 0) {
        return acc;
      }

      acc[trimmed.slice(0, separatorIndex).trim()] = trimmed.slice(separatorIndex + 1).trim();
      return acc;
    }, {});
};

const settingsEnv = readEnvFile(path.resolve(__dirname, '../../settings/.env'));
const env = { ...process.env, ...settingsEnv };
const coreApiHost = env.CORE_SERVICE_HOST || env.FASTAPI_HOST || '127.0.0.1';
const coreApiPort = Number(env.CORE_SERVICE_PORT || env.FASTAPI_PORT || 8000);

module.exports = {
  '/api': {
    target: `http://${coreApiHost}:${coreApiPort}`,
    changeOrigin: true,
    secure: false,
    logLevel: 'warn'
  }
};
