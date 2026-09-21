import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { createServer, request as requestHttp } from 'node:http';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import test from 'node:test';

import { createPreviewServer, startPreviewServer } from './preview-serve.mjs';

async function listen(server) {
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    return server.address().port;
}

async function close(server) {
    if (!server) {
        return;
    }
    server.close();
    await once(server, 'close');
}

function request(port, options = {}, body = '') {
    return new Promise((resolve, reject) => {
        const requestOptions = {
            host: '127.0.0.1',
            method: 'GET',
            port,
            ...options,
        };
        const outgoing = requestHttp(requestOptions, (response) => {
            const chunks = [];
            response.on('data', (chunk) => chunks.push(chunk));
            response.on('end', () => resolve({
                body: Buffer.concat(chunks).toString('utf8'),
                headers: response.headers,
                statusCode: response.statusCode,
            }));
        });
        outgoing.on('error', reject);
        if (body) {
            outgoing.write(body);
        }
        outgoing.end();
    });
}

test('static preview serves files, SPA routes, HEAD, and rejects unsafe paths', async (t) => {
    const root = await mkdtemp(join(tmpdir(), 'adsmod-preview-static-'));
    await mkdir(join(root, 'assets'));
    await writeFile(join(root, 'index.html'), '<html><body>ADSMOD shell</body></html>');
    await writeFile(join(root, 'assets', 'app.js'), 'console.log("ok");');
    const server = await startPreviewServer({ root, port: 0, host: '127.0.0.1', proxyRules: {} });
    const port = server.address().port;

    t.after(async () => {
        await close(server);
        await rm(root, { force: true, recursive: true });
    });

    const rootResponse = await request(port, { path: '/' });
    assert.equal(rootResponse.statusCode, 200);
    assert.match(rootResponse.headers['content-type'], /text\/html/);
    assert.match(rootResponse.body, /ADSMOD shell/);

    const routeResponse = await request(port, { path: '/datasets?tab=saved' });
    assert.equal(routeResponse.statusCode, 200);
    assert.match(routeResponse.body, /ADSMOD shell/);

    const headResponse = await request(port, { method: 'HEAD', path: '/assets/app.js' });
    assert.equal(headResponse.statusCode, 200);
    assert.equal(headResponse.body, '');
    assert.equal(Number(headResponse.headers['content-length']), Buffer.byteLength('console.log("ok");'));

    const traversalResponse = await request(port, { path: '/..%5Csecret.txt' });
    assert.equal(traversalResponse.statusCode, 403);

    const missingAssetResponse = await request(port, { path: '/missing.js' });
    assert.equal(missingAssetResponse.statusCode, 404);
});

test('preview proxies configured API paths with request bodies and response metadata', async (t) => {
    let receivedBody = '';
    let receivedHost = '';
    const backend = createServer((requestMessage, response) => {
        const chunks = [];
        requestMessage.on('data', (chunk) => chunks.push(chunk));
        requestMessage.on('end', () => {
            receivedBody = Buffer.concat(chunks).toString('utf8');
            receivedHost = requestMessage.headers.host;
            response.writeHead(201, {
                connection: 'close',
                'content-type': 'application/json',
                'x-backend-check': 'passed',
            });
            response.end(JSON.stringify({ path: requestMessage.url, body: receivedBody }));
        });
    });
    const backendPort = await listen(backend);
    const root = await mkdtemp(join(tmpdir(), 'adsmod-preview-proxy-'));
    await writeFile(join(root, 'index.html'), '<html>proxy fixture</html>');
    const preview = await startPreviewServer({
        host: '127.0.0.1',
        port: 0,
        root,
        proxyRules: {
            '/api/v1': { changeOrigin: true, target: `http://127.0.0.1:${backendPort}` },
            '/health': { changeOrigin: true, target: `http://127.0.0.1:${backendPort}` },
        },
    });
    const previewPort = preview.address().port;

    t.after(async () => {
        await close(preview);
        await close(backend);
        await rm(root, { force: true, recursive: true });
    });

    const apiResponse = await request(previewPort, {
        headers: { 'content-type': 'text/plain', 'x-client-check': 'passed' },
        method: 'POST',
        path: '/api/v1/echo?source=test',
    }, 'request-body');
    assert.equal(apiResponse.statusCode, 201);
    assert.equal(apiResponse.headers['x-backend-check'], 'passed');
    assert.notEqual(apiResponse.headers.connection, 'close');
    assert.deepEqual(JSON.parse(apiResponse.body), {
        body: 'request-body',
        path: '/api/v1/echo?source=test',
    });
    assert.equal(receivedBody, 'request-body');
    assert.equal(receivedHost, `127.0.0.1:${backendPort}`);
});

test('preview fails immediately when the built index is missing', async () => {
    const root = await mkdtemp(join(tmpdir(), 'adsmod-preview-missing-'));
    try {
        assert.throws(
            () => createPreviewServer({ root, proxyRules: {} }),
            /Frontend bundle is missing:.*Run npm run build before npm run preview/,
        );
    } finally {
        await rm(root, { force: true, recursive: true });
    }
});
