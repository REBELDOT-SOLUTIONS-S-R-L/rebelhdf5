'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');

const { app, BrowserWindow, dialog, session, shell } = require('electron');

const projectRoot = path.resolve(__dirname, '..');
const distDir = path.join(projectRoot, 'dist');
const appIcon = path.join(projectRoot, 'public', 'favicon.ico');
const backendScript = path.join(projectRoot, 'scripts', 'backend_server.py');
const appPreferredPort = Number(process.env.REBELHDF5_DESKTOP_PORT ?? process.env.ELECTRON_APP_PORT) || 4096;
const backendPreferredPort = Number(process.env.PYTHON_BACKEND_PORT ?? process.env.MERGE_SERVER_PORT) || 4095;
const backendDir = process.env.PYTHON_BACKEND_DIR
  ?? process.env.MERGE_SERVER_DIR
  ?? path.resolve(projectRoot, '..');
const lehomePython = path.resolve(projectRoot, '..', 'ROBOTICS-lehome-challenge', '.venv', 'bin', 'python');
const backendPython = process.env.PYTHON_BACKEND_PYTHON
  ?? (fs.existsSync(lehomePython) ? lehomePython : 'python3');
const smokeTest = process.argv.includes('--smoke-test');

app.setName('rebelHDF5');
if (process.platform === 'linux' && typeof app.setDesktopName === 'function') {
  app.setDesktopName('rebelhdf5.desktop');
}

let backendProcess = null;
let staticServer = null;
let mainWindow = null;
let quitting = false;

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

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function checkBackendHealth(port) {
  return new Promise((resolve) => {
    const request = http.get(`http://127.0.0.1:${String(port)}/api/health`, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });

    request.on('error', () => {
      resolve(false);
    });

    request.setTimeout(500, () => {
      request.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(port) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (await checkBackendHealth(port)) {
      return;
    }

    await delay(100);
  }

  throw new Error(`Python backend did not become healthy on port ${String(port)}.`);
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

  throw new Error(`Could not find a free backend port near ${String(preferredPort)}.`);
}

function isTrustedLocalOrigin(origin) {
  return origin.startsWith('http://127.0.0.1:')
    || origin.startsWith('http://localhost:');
}

function configureFileSystemAccess() {
  session.defaultSession.setPermissionCheckHandler((_webContents, permission, origin) => {
    return permission === 'fileSystem' && isTrustedLocalOrigin(origin);
  });

  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback, details) => {
    const origin = details?.requestingUrl ?? webContents.getURL();
    callback(permission === 'fileSystem' && isTrustedLocalOrigin(origin));
  });

  session.defaultSession.on('file-system-access-restricted', (_event, details, callback) => {
    callback(isTrustedLocalOrigin(details.origin) ? 'allow' : 'deny');
  });
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

async function resolveStaticFile(requestUrl) {
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

async function startStaticServer(port) {
  if (!fs.existsSync(path.join(distDir, 'index.html'))) {
    throw new Error('dist/index.html does not exist. Run `pnpm build` before `pnpm desktop:run`.');
  }

  const server = http.createServer(async (request, response) => {
    try {
      const filePath = await resolveStaticFile(request.url);
      if (!filePath) {
        response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
        response.end('Not found');
        return;
      }

      const body = await fsp.readFile(filePath);
      response.writeHead(200, {
        'cache-control': 'no-store',
        'content-type': MIME_TYPES.get(path.extname(filePath)) ?? 'application/octet-stream',
      });
      response.end(body);
    } catch (error) {
      response.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' });
      response.end(error instanceof Error ? error.message : String(error));
    }
  });

  const address = await listen(server, port, '127.0.0.1');
  staticServer = server;
  return `http://127.0.0.1:${String(address.port)}/`;
}

function startBackend(port) {
  backendProcess = spawn(
    backendPython,
    [backendScript, '--dir', backendDir, '--port', String(port)],
    {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    },
  );

  backendProcess.stdout?.on('data', (data) => {
    const text = data.toString().trim();
    if (text) {
      console.log(`[backend-server] ${text}`);
    }
  });

  backendProcess.stderr?.on('data', (data) => {
    const text = data.toString().trim();
    if (text) {
      console.error(`[backend-server] ${text}`);
    }
  });

  backendProcess.on('exit', (code, signal) => {
    if (code !== 0 && code !== null) {
      console.error(`[backend-server] exited with code ${String(code)}`);
    } else if (signal) {
      console.error(`[backend-server] exited with signal ${signal}`);
    }
    backendProcess = null;
  });

  backendProcess.on('error', (error) => {
    console.error(`[backend-server] failed to start: ${error.message}`);
    backendProcess = null;
  });
}

function stopBackend() {
  const runningBackend = backendProcess;
  if (!runningBackend) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    let closed = false;
    const timeout = setTimeout(() => {
      if (!closed) {
        runningBackend.kill('SIGKILL');
      }
      resolve();
    }, 2500);

    runningBackend.once('close', () => {
      closed = true;
      clearTimeout(timeout);
      resolve();
    });

    runningBackend.kill('SIGTERM');
    backendProcess = null;
  });
}

async function shutdown() {
  await Promise.all([
    stopBackend(),
    closeServer(staticServer),
  ]);
  staticServer = null;
}

async function createWindow(appUrl, backendPort) {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1000,
    minHeight: 700,
    title: 'rebelHDF5',
    icon: appIcon,
    webPreferences: {
      additionalArguments: [`--backend-port=${String(backendPort)}`],
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.cjs'),
      sandbox: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: 'deny' };
  });

  await mainWindow.loadURL(appUrl);
}

async function boot() {
  try {
    const backendPort = await findAvailablePort(backendPreferredPort);
    const appPort = await findAvailablePort(appPreferredPort, new Set([backendPort]));
    const appUrl = await startStaticServer(appPort);

    startBackend(backendPort);

    if (smokeTest) {
      try {
        await waitForBackend(backendPort);
      } finally {
        await shutdown();
      }
      app.exit(0);
      return;
    }

    await createWindow(appUrl, backendPort);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    await shutdown();

    if (smokeTest) {
      app.exit(1);
      return;
    }

    await dialog.showMessageBox({
      type: 'error',
      title: 'rebelHDF5 failed to start',
      message,
    });
    app.quit();
  }
}

app.whenReady().then(() => {
  configureFileSystemAccess();
  void boot();
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', (event) => {
  if (quitting) {
    return;
  }

  event.preventDefault();
  quitting = true;
  void shutdown().finally(() => {
    app.quit();
  });
});

for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.once(signal, () => {
    void shutdown().finally(() => {
      process.exit(0);
    });
  });
}
