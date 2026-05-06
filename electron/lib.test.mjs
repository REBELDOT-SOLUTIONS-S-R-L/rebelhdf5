// @vitest-environment node
import { createRequire } from 'node:module';
import fsp from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const {
  checkBackendHealth,
  closeServer,
  findAvailablePort,
  getMimeType,
  isInside,
  isTrustedLocalOrigin,
  listen,
  resolveStaticFile,
} = require('./lib.cjs');

describe('getMimeType', () => {
  it('returns the mapped type for known extensions', () => {
    expect(getMimeType('foo.js')).toBe('text/javascript; charset=utf-8');
    expect(getMimeType('foo.wasm')).toBe('application/wasm');
    expect(getMimeType('plugin.so')).toBe('application/octet-stream');
    expect(getMimeType('a/b/index.html')).toBe('text/html; charset=utf-8');
  });

  it('falls back to octet-stream for unknown extensions', () => {
    expect(getMimeType('foo.xyz')).toBe('application/octet-stream');
    expect(getMimeType('noext')).toBe('application/octet-stream');
  });
});

describe('isTrustedLocalOrigin', () => {
  it('accepts loopback origins', () => {
    expect(isTrustedLocalOrigin('http://127.0.0.1:4096')).toBe(true);
    expect(isTrustedLocalOrigin('http://localhost:8080')).toBe(true);
  });

  it('rejects everything else', () => {
    expect(isTrustedLocalOrigin('https://127.0.0.1:4096')).toBe(false);
    expect(isTrustedLocalOrigin('http://example.com')).toBe(false);
    expect(isTrustedLocalOrigin('http://127.0.0.2:4096')).toBe(false);
    expect(isTrustedLocalOrigin('file://')).toBe(false);
  });
});

describe('isInside', () => {
  it('returns true for paths inside the parent', () => {
    expect(isInside('/a/b', '/a/b/c')).toBe(true);
    expect(isInside('/a/b', '/a/b')).toBe(true);
    expect(isInside('/a/b', '/a/b/c/d.html')).toBe(true);
  });

  it('returns false for path-traversal escapes', () => {
    expect(isInside('/a/b', '/a/c')).toBe(false);
    expect(isInside('/a/b', '/etc/passwd')).toBe(false);
    expect(isInside('/a/b', '/a/b/../c')).toBe(false);
  });
});

describe('resolveStaticFile', () => {
  let distDir;

  beforeEach(async () => {
    distDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'mhdf5-dist-'));
    await fsp.writeFile(path.join(distDir, 'index.html'), '<!doctype html>');
    await fsp.mkdir(path.join(distDir, 'assets'));
    await fsp.writeFile(
      path.join(distDir, 'assets', 'app.js'),
      'console.log("hi")',
    );
  });

  afterEach(async () => {
    await fsp.rm(distDir, { recursive: true, force: true });
  });

  it('serves an existing file', async () => {
    const resolved = await resolveStaticFile('/assets/app.js', distDir);
    expect(resolved).toBe(path.join(distDir, 'assets', 'app.js'));
  });

  it('serves index.html for the root URL', async () => {
    const resolved = await resolveStaticFile('/', distDir);
    expect(resolved).toBe(path.join(distDir, 'index.html'));
  });

  it('falls back to index.html for unknown non-asset paths (SPA routing)', async () => {
    const resolved = await resolveStaticFile('/view', distDir);
    expect(resolved).toBe(path.join(distDir, 'index.html'));
  });

  it('returns null for missing assets (no SPA fallback)', async () => {
    const resolved = await resolveStaticFile('/assets/missing.js', distDir);
    expect(resolved).toBeNull();
  });

  it('decodes percent-encoded paths before resolving', async () => {
    const resolved = await resolveStaticFile('/assets/%61pp.js', distDir);
    expect(resolved).toBe(path.join(distDir, 'assets', 'app.js'));
  });
});

describe('findAvailablePort', () => {
  it('returns the preferred port when free', async () => {
    const probe = http.createServer();
    const address = await listen(probe, 0, '127.0.0.1');
    const preferred = address.port;
    await closeServer(probe);

    const port = await findAvailablePort(preferred);
    expect(port).toBe(preferred);
  });

  it('skips a port that is already in use and returns the next free one', async () => {
    const probe = http.createServer();
    const address = await listen(probe, 0, '127.0.0.1');
    const taken = address.port;

    try {
      const port = await findAvailablePort(taken);
      expect(port).toBeGreaterThan(taken);
      expect(port).toBeLessThanOrEqual(taken + 100);
    } finally {
      await closeServer(probe);
    }
  });

  it('skips reserved ports', async () => {
    const probe = http.createServer();
    const address = await listen(probe, 0, '127.0.0.1');
    const free = address.port;
    await closeServer(probe);

    const port = await findAvailablePort(free, new Set([free]));
    expect(port).not.toBe(free);
  });
});

describe('checkBackendHealth', () => {
  let server = null;

  afterEach(async () => {
    if (server) {
      await closeServer(server);
      server = null;
    }
  });

  it('resolves true when /api/health returns 200', async () => {
    server = http.createServer((request, response) => {
      if (request.url === '/api/health') {
        response.writeHead(200);
        response.end('ok');
        return;
      }
      response.writeHead(404);
      response.end();
    });
    const address = await listen(server, 0, '127.0.0.1');

    expect(await checkBackendHealth(address.port)).toBe(true);
  });

  it('resolves false on non-200 responses', async () => {
    server = http.createServer((_request, response) => {
      response.writeHead(503);
      response.end();
    });
    const address = await listen(server, 0, '127.0.0.1');

    expect(await checkBackendHealth(address.port)).toBe(false);
  });

  it('resolves false when nothing is listening on the port', async () => {
    const probe = http.createServer();
    const address = await listen(probe, 0, '127.0.0.1');
    const closedPort = address.port;
    await closeServer(probe);

    expect(await checkBackendHealth(closedPort)).toBe(false);
  });
});
