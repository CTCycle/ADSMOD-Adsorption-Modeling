import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const clientRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(clientRoot, '../..');
const require = createRequire(import.meta.url);
const { loadRuntimeSettings } = require('./runtime-settings.cjs');
const runtime = loadRuntimeSettings(repositoryRoot);
const angularCli = path.join(clientRoot, 'node_modules', '@angular', 'cli', 'bin', 'ng.js');
const child = spawn(process.execPath, [
  angularCli,
  'serve',
  '--host',
  runtime.host,
  '--port',
  String(runtime.frontend_port),
  '--proxy-config',
  'proxy.conf.cjs',
], { cwd: clientRoot, stdio: 'inherit' });

child.on('exit', (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
