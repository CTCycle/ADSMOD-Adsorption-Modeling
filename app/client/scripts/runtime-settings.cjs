// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function loadRuntimeSettings(repositoryRoot = path.resolve(__dirname, '../../..')) {
  const appRoot = path.join(repositoryRoot, 'app');
  const embeddedPython = path.join(repositoryRoot, 'app', 'server', '.venv', 'Scripts', 'python.exe');
  const python = process.env.ADSMOD_PYTHON || (fs.existsSync(embeddedPython) ? embeddedPython : 'python');
  const pythonPath = [appRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
  const result = spawnSync(python, ['-m', 'server.runtime_config', '--json'], {
    cwd: repositoryRoot,
    encoding: 'utf8',
    env: { ...process.env, PYTHONPATH: pythonPath },
  });

  if (result.error || result.status !== 0) {
    throw new Error('ADSMOD runtime settings could not be loaded from SQLite. Initialize the application database and verify settings/.env.');
  }
  const runtimeLine = String(result.stdout || '').split(/\r?\n/).find((line) => line.startsWith('ADSMOD_RUNTIME:'));
  if (!runtimeLine) throw new Error('ADSMOD runtime settings did not return a valid runtime record.');

  let runtime;
  try {
    runtime = JSON.parse(runtimeLine.slice('ADSMOD_RUNTIME:'.length));
  } catch {
    throw new Error('ADSMOD runtime settings returned invalid JSON.');
  }
  const backendPort = Number(runtime.backend_port);
  const frontendPort = Number(runtime.frontend_port);
  if (!['127.0.0.1', 'localhost', '::1'].includes(runtime.host)
    || !Number.isInteger(backendPort) || backendPort < 1024 || backendPort > 65535
    || !Number.isInteger(frontendPort) || frontendPort < 1024 || frontendPort > 65535
    || backendPort === frontendPort) {
    throw new Error('ADSMOD runtime settings contain an invalid local endpoint.');
  }
  return { ...runtime, backend_port: backendPort, frontend_port: frontendPort };
}

module.exports = { loadRuntimeSettings };
