'use strict';

const fsp = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');

const MIME_TYPES = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.ico', 'image/x-icon'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml; charset=utf-8'],
  ['.wasm', 'application/wasm'],
  ['.webp', 'image/webp'],
  ['.so', 'application/octet-stream'],
]);

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function listen(server, port, host) {
  return new Promise((resolve, reject) => {
    function cleanup() {
      server.off('error', onError);
      server.off('listening', onListening);
    }

    function onError(error) {
      cleanup();
      reject(error);
    }

    function onListening() {
      cleanup();
      resolve(server.address());
    }

    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port, host);
  });
}

function closeServer(server) {
  if (!server) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    server.close(() => resolve());
  });
}

function checkBackendHealth(port) {
  return new Promise((resolve) => {
    const request = http.get(
      `http://127.0.0.1:${String(port)}/api/health`,
      (response) => {
        response.resume();
        resolve(response.statusCode === 200);
      },
    );

    request.on('error', () => {
      resolve(false);
    });

    request.setTimeout(500, () => {
      request.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(port, attempts = 40, intervalMs = 100) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (await checkBackendHealth(port)) {
      return;
    }

    await delay(intervalMs);
  }

  throw new Error(
    `Python backend did not become healthy on port ${String(port)}.`,
  );
}

async function findAvailablePort(preferredPort, reservedPorts = new Set()) {
  for (let port = preferredPort; port < preferredPort + 100; port += 1) {
    if (reservedPorts.has(port)) {
      continue;
    }

    const probe = http.createServer();
    try {
      await listen(probe, port, '127.0.0.1');
      await closeServer(probe);
      return port;
    } catch (error) {
      await closeServer(probe).catch(() => {});
      if (error.code !== 'EADDRINUSE') {
        throw error;
      }
    }
  }

  throw new Error(
    `Could not find a free backend port near ${String(preferredPort)}.`,
  );
}

function isTrustedLocalOrigin(origin) {
  return (
    origin.startsWith('http://127.0.0.1:') ||
    origin.startsWith('http://localhost:')
  );
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return (
    relative === '' ||
    (!relative.startsWith('..') && !path.isAbsolute(relative))
  );
}

async function resolveStaticFile(requestUrl, distDir) {
  const url = new URL(requestUrl ?? '/', 'http://127.0.0.1');
  const pathname = decodeURIComponent(url.pathname);
  const requestedPath = path.normalize(path.join(distDir, pathname));

  if (!isInside(distDir, requestedPath)) {
    return null;
  }

  try {
    const stat = await fsp.stat(requestedPath);
    if (stat.isDirectory()) {
      return path.join(requestedPath, 'index.html');
    }

    return requestedPath;
  } catch {
    if (pathname.startsWith('/assets/')) {
      return null;
    }

    return path.join(distDir, 'index.html');
  }
}

function getMimeType(filePath) {
  return MIME_TYPES.get(path.extname(filePath)) ?? 'application/octet-stream';
}

module.exports = {
  MIME_TYPES,
  checkBackendHealth,
  closeServer,
  delay,
  findAvailablePort,
  getMimeType,
  isInside,
  isTrustedLocalOrigin,
  listen,
  resolveStaticFile,
  waitForBackend,
};
