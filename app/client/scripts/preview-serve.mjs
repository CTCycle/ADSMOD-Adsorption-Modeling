import { createReadStream, existsSync, statSync } from 'node:fs';
import { Buffer } from 'node:buffer';
import { createServer, request as requestHttp } from 'node:http';
import { createRequire } from 'node:module';
import { dirname, extname, isAbsolute, join, relative, resolve } from 'node:path';
import { URL, fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const clientRoot = resolve(__dirname, '..');
const defaultDistRoot = resolve(clientRoot, 'dist', 'browser');
const require = createRequire(import.meta.url);
const defaultProxyRules = require(resolve(clientRoot, 'proxy.conf.cjs'));

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 4173;
const HOP_BY_HOP_HEADERS = new Set([
    'connection',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
]);

const MIME_TYPES = new Map([
    ['.css', 'text/css; charset=utf-8'],
    ['.gif', 'image/gif'],
    ['.html', 'text/html; charset=utf-8'],
    ['.ico', 'image/x-icon'],
    ['.jpeg', 'image/jpeg'],
    ['.jpg', 'image/jpeg'],
    ['.js', 'text/javascript; charset=utf-8'],
    ['.json', 'application/json; charset=utf-8'],
    ['.map', 'application/json; charset=utf-8'],
    ['.png', 'image/png'],
    ['.svg', 'image/svg+xml'],
    ['.txt', 'text/plain; charset=utf-8'],
    ['.wasm', 'application/wasm'],
    ['.webp', 'image/webp'],
    ['.woff', 'font/woff'],
    ['.woff2', 'font/woff2'],
]);

function parseFlag(args, name, fallback) {
    for (let index = 0; index < args.length; index += 1) {
        const argument = args[index];
        if (argument === name && index + 1 < args.length) {
            return args[index + 1];
        }
        if (argument.startsWith(`${name}=`)) {
            return argument.slice(name.length + 1);
        }
    }
    return fallback;
}

export function parseArgs(args = process.argv.slice(2)) {
    const host = parseFlag(args, '--host', DEFAULT_HOST);
    const rawPort = parseFlag(args, '--port', DEFAULT_PORT);
    const port = Number(rawPort);
    if (!host || !Number.isInteger(port) || port < 1 || port > 65535) {
        throw new Error(`Invalid preview server address: host=${host || '<empty>'}, port=${rawPort}`);
    }
    return { host, port };
}

function getProxyRule(pathname, proxyRules) {
    return Object.entries(proxyRules).find(([prefix]) => pathname === prefix || pathname.startsWith(`${prefix}/`))?.[1] ?? null;
}

function responseHeadersWithoutHopByHop(headers) {
    return Object.fromEntries(
        Object.entries(headers).filter(([name]) => !HOP_BY_HOP_HEADERS.has(name.toLowerCase())),
    );
}

function requestHeadersForProxy(headers, target, changeOrigin) {
    const forwarded = { ...headers };
    for (const header of HOP_BY_HOP_HEADERS) {
        delete forwarded[header];
    }
    delete forwarded.host;
    if (changeOrigin) {
        forwarded.host = target.host;
    }
    return forwarded;
}

function sendText(response, statusCode, message, headers = {}) {
    const body = Buffer.from(message, 'utf8');
    response.writeHead(statusCode, {
        'content-length': body.length,
        'content-type': 'text/plain; charset=utf-8',
        ...headers,
    });
    response.end(body);
}

function proxyRequest(request, response, proxyRule) {
    const target = new URL(proxyRule.target);
    const targetUrl = new URL(request.url ?? '/', target);
    const upstream = requestHttp(targetUrl, {
        headers: requestHeadersForProxy(request.headers, target, proxyRule.changeOrigin === true),
        method: request.method,
    }, (upstreamResponse) => {
        response.writeHead(upstreamResponse.statusCode ?? 502, responseHeadersWithoutHopByHop(upstreamResponse.headers));
        upstreamResponse.pipe(response);
    });

    upstream.on('error', (error) => {
        if (!response.headersSent) {
            sendText(response, 502, `Backend proxy failed: ${error.message}\n`);
        } else {
            response.destroy(error);
        }
    });
    request.pipe(upstream);
}

function resolveStaticFile(root, pathname, indexPath) {
    let decodedPath;
    try {
        decodedPath = decodeURIComponent(pathname);
    } catch {
        return { statusCode: 400, message: 'Malformed URL path.\n' };
    }
    if (decodedPath.includes('\0')) {
        return { statusCode: 400, message: 'Unsafe URL path.\n' };
    }

    const rootPath = resolve(root);
    const candidate = resolve(rootPath, `.${decodedPath.replaceAll('\\', '/')}`);
    const candidateRelativePath = relative(rootPath, candidate);
    if (candidateRelativePath.startsWith('..') || isAbsolute(candidateRelativePath)) {
        return { statusCode: 403, message: 'Unsafe URL path.\n' };
    }

    try {
        const stats = statSync(candidate);
        if (stats.isFile()) {
            return { filePath: candidate };
        }
        if (stats.isDirectory()) {
            const directoryIndex = join(candidate, 'index.html');
            if (existsSync(directoryIndex) && statSync(directoryIndex).isFile()) {
                return { filePath: directoryIndex };
            }
        }
    } catch {
        // A missing static asset may still be an Angular client-side route.
    }

    if (!extname(decodedPath)) {
        return { filePath: indexPath };
    }
    return { statusCode: 404, message: 'Not found.\n' };
}

function serveStatic(request, response, root, indexPath) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
        sendText(response, 405, 'Method not allowed.\n', { allow: 'GET, HEAD' });
        return;
    }

    const resolved = resolveStaticFile(root, new URL(request.url ?? '/', 'http://localhost').pathname, indexPath);
    if (resolved.statusCode) {
        sendText(response, resolved.statusCode, resolved.message);
        return;
    }

    const stats = statSync(resolved.filePath);
    response.writeHead(200, {
        'cache-control': resolved.filePath === indexPath ? 'no-cache' : 'public, max-age=31536000, immutable',
        'content-length': stats.size,
        'content-type': MIME_TYPES.get(extname(resolved.filePath).toLowerCase()) ?? 'application/octet-stream',
    });
    if (request.method === 'HEAD') {
        response.end();
        return;
    }
    createReadStream(resolved.filePath).on('error', (error) => response.destroy(error)).pipe(response);
}

export function createPreviewServer({ root = defaultDistRoot, proxyRules = defaultProxyRules } = {}) {
    const resolvedRoot = resolve(root);
    const indexPath = join(resolvedRoot, 'index.html');
    if (!existsSync(indexPath) || !statSync(indexPath).isFile()) {
        throw new Error(`Frontend bundle is missing: ${indexPath}. Run npm run build before npm run preview.`);
    }

    return createServer((request, response) => {
        let requestUrl;
        try {
            requestUrl = new URL(request.url ?? '/', 'http://localhost');
        } catch {
            sendText(response, 400, 'Malformed request URL.\n');
            return;
        }

        const proxyRule = getProxyRule(requestUrl.pathname, proxyRules);
        if (proxyRule) {
            proxyRequest(request, response, proxyRule);
            return;
        }
        serveStatic(request, response, resolvedRoot, indexPath);
    });
}

export function startPreviewServer(options = {}) {
    const server = createPreviewServer(options);
    const host = options.host ?? DEFAULT_HOST;
    const port = options.port ?? DEFAULT_PORT;
    return new Promise((resolveServer, reject) => {
        const onError = (error) => {
            server.off('listening', onListening);
            reject(error);
        };
        const onListening = () => {
            server.off('error', onError);
            resolveServer(server);
        };
        server.once('error', onError);
        server.once('listening', onListening);
        server.listen(port, host);
    });
}

async function main() {
    try {
        const server = await startPreviewServer(parseArgs());
        const address = server.address();
        const displayHost = typeof address === 'object' && address ? address.address : DEFAULT_HOST;
        const displayPort = typeof address === 'object' && address ? address.port : DEFAULT_PORT;
        console.log(`ADSMOD static preview serving dist/browser at http://${displayHost}:${displayPort}`);

        const shutdown = () => {
            server.close(() => process.exit(0));
        };
        process.once('SIGINT', shutdown);
        process.once('SIGTERM', shutdown);
    } catch (error) {
        console.error(`[preview] ${error.message}`);
        process.exitCode = 1;
    }
}

const isMainModule = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (isMainModule) {
    void main();
}
