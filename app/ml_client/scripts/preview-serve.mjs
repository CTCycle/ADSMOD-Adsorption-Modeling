import { spawn } from 'node:child_process';
import path from 'node:path';

const cliEntrypoint = path.resolve('node_modules', '@angular', 'cli', 'bin', 'ng.js');
const forwardedArgs = process.argv.slice(2);

const hasFlag = (name) => forwardedArgs.some((arg, index) => arg === name || arg.startsWith(`${name}=`) || (index > 0 && forwardedArgs[index - 1] === name));

const args = [
    'serve',
    '--configuration',
    'production',
    '--proxy-config',
    'proxy.conf.cjs',
    ...(hasFlag('--host') ? [] : ['--host', '127.0.0.1']),
    ...(hasFlag('--port') ? [] : ['--port', '4174']),
    ...forwardedArgs,
];

const child = spawn(process.execPath, [cliEntrypoint, ...args], {
    stdio: 'inherit',
    env: process.env,
});

child.on('exit', (code, signal) => {
    if (signal) {
        process.kill(process.pid, signal);
        return;
    }
    process.exit(code ?? 0);
});
