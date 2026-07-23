'use strict';

const { spawn } = require('node:child_process');
const { randomUUID } = require('node:crypto');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');

const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  session,
  shell,
} = require('electron');

const {
  checkBackendHealth,
  closeServer,
  findAvailablePort,
  getMimeType,
  isTrustedLocalOrigin,
  listen,
  resolveStaticFile,
  waitForBackend,
} = require('./lib.cjs');

const projectRoot = path.resolve(__dirname, '..');
const distDir = path.join(projectRoot, 'dist');
const appIcon = path.join(projectRoot, 'public', 'favicon.ico');
const backendScript = path.join(projectRoot, 'scripts', 'backend_server.py');
const appPreferredPort =
  Number(process.env.REBELHDF5_DESKTOP_PORT ?? process.env.ELECTRON_APP_PORT) ||
  4096;
const backendPreferredPort =
  Number(process.env.PYTHON_BACKEND_PORT ?? process.env.MERGE_SERVER_PORT) ||
  4095;
const backendDir =
  process.env.PYTHON_BACKEND_DIR ??
  process.env.MERGE_SERVER_DIR ??
  path.resolve(projectRoot, '..');
const localVenvPython = path.resolve(projectRoot, '.venv', 'bin', 'python');
const lehomePython = path.resolve(
  projectRoot,
  '..',
  'ROBOTICS-lehome-challenge',
  '.venv',
  'bin',
  'python',
);
const backendPython =
  process.env.PYTHON_BACKEND_PYTHON ??
  (fs.existsSync(localVenvPython)
    ? localVenvPython
    : fs.existsSync(lehomePython)
      ? lehomePython
      : 'python3');
const smokeTest = process.argv.includes('--smoke-test');
const backendIpcPrefix = 'REBELHDF5_IPC ';
const outputAuthorizationTimeoutMs = 5000;

app.setName('rebelHDF5');
if (process.platform === 'linux' && typeof app.setDesktopName === 'function') {
  app.setDesktopName('rebelhdf5.desktop');
}

let backendProcess = null;
let staticServer = null;
let mainWindow = null;
let quitting = false;
let backendStdoutBuffer = '';
const pendingOutputAuthorizations = new Map();

function rejectPendingOutputAuthorizations(message) {
  for (const { reject, timeout } of pendingOutputAuthorizations.values()) {
    clearTimeout(timeout);
    reject(new Error(message));
  }
  pendingOutputAuthorizations.clear();
}

function handleBackendStdout(data) {
  backendStdoutBuffer += data.toString();
  const lines = backendStdoutBuffer.split('\n');
  backendStdoutBuffer = lines.pop() ?? '';

  for (const line of lines) {
    if (!line.startsWith(backendIpcPrefix)) {
      if (line.trim()) {
        console.log(`[backend-server] ${line}`);
      }
      continue;
    }

    try {
      const response = JSON.parse(line.slice(backendIpcPrefix.length));
      const pending = pendingOutputAuthorizations.get(response.token);
      if (!pending) {
        continue;
      }

      clearTimeout(pending.timeout);
      pendingOutputAuthorizations.delete(response.token);
      if (
        response.ok === true &&
        typeof response.path === 'string' &&
        response.path.length > 0
      ) {
        pending.resolve({
          path: response.path,
          authorization: response.token,
        });
      } else {
        pending.reject(
          new Error(
            typeof response.error === 'string'
              ? response.error
              : 'The backend rejected the selected output folder.',
          ),
        );
      }
    } catch (error) {
      console.error(
        `[backend-server] Invalid desktop IPC response: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }
}

function authorizeOutputDirectory(selectedPath) {
  const runningBackend = backendProcess;
  if (!runningBackend?.stdin?.writable) {
    return Promise.reject(new Error('The Python backend is not running.'));
  }

  const token = randomUUID();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingOutputAuthorizations.delete(token);
      reject(new Error('Timed out while authorizing the output folder.'));
    }, outputAuthorizationTimeoutMs);

    pendingOutputAuthorizations.set(token, { resolve, reject, timeout });
    runningBackend.stdin.write(
      `${JSON.stringify({
        type: 'authorize-output-directory',
        token,
        path: selectedPath,
      })}\n`,
      (error) => {
        if (!error) {
          return;
        }
        const pending = pendingOutputAuthorizations.get(token);
        if (!pending) {
          return;
        }
        clearTimeout(pending.timeout);
        pendingOutputAuthorizations.delete(token);
        pending.reject(error);
      },
    );
  });
}

ipcMain.handle('rebelhdf5:choose-directory', async (event, defaultPath) => {
  if (!mainWindow || event.sender.id !== mainWindow.webContents.id) {
    return undefined;
  }
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose LeRobot output folder',
    defaultPath:
      typeof defaultPath === 'string' && defaultPath.length > 0
        ? defaultPath
        : undefined,
    properties: ['openDirectory', 'createDirectory'],
  });
  if (result.canceled) {
    return undefined;
  }
  return authorizeOutputDirectory(result.filePaths[0]);
});

function configureFileSystemAccess() {
  session.defaultSession.setPermissionCheckHandler(
    (_webContents, permission, origin) => {
      return permission === 'fileSystem' && isTrustedLocalOrigin(origin);
    },
  );

  session.defaultSession.setPermissionRequestHandler(
    (webContents, permission, callback, details) => {
      const origin = details?.requestingUrl ?? webContents.getURL();
      callback(permission === 'fileSystem' && isTrustedLocalOrigin(origin));
    },
  );

  session.defaultSession.on(
    'file-system-access-restricted',
    (_event, details, callback) => {
      callback(isTrustedLocalOrigin(details.origin) ? 'allow' : 'deny');
    },
  );
}

async function startStaticServer(port) {
  if (!fs.existsSync(path.join(distDir, 'index.html'))) {
    throw new Error(
      'dist/index.html does not exist. Run `pnpm build` before `pnpm desktop:run`.',
    );
  }

  const server = http.createServer(async (request, response) => {
    try {
      const filePath = await resolveStaticFile(request.url, distDir);
      if (!filePath) {
        response.writeHead(404, {
          'content-type': 'text/plain; charset=utf-8',
        });
        response.end('Not found');
        return;
      }

      const body = await fsp.readFile(filePath);
      response.writeHead(200, {
        'cache-control': 'no-store',
        'content-type': getMimeType(filePath),
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
    [
      backendScript,
      '--dir',
      backendDir,
      '--port',
      String(port),
      '--output-authorization-stdin',
    ],
    {
      cwd: projectRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    },
  );

  backendStdoutBuffer = '';
  backendProcess.stdout?.on('data', handleBackendStdout);
  backendProcess.stdin?.on('error', (error) => {
    rejectPendingOutputAuthorizations(
      `Could not authorize the output folder: ${error.message}`,
    );
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
    rejectPendingOutputAuthorizations('The Python backend stopped.');
    backendProcess = null;
  });

  backendProcess.on('error', (error) => {
    console.error(`[backend-server] failed to start: ${error.message}`);
    rejectPendingOutputAuthorizations('The Python backend failed to start.');
    backendProcess = null;
  });
}

function stopBackend() {
  const runningBackend = backendProcess;
  if (!runningBackend) {
    return Promise.resolve();
  }
  rejectPendingOutputAuthorizations('The Python backend is shutting down.');

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
  await Promise.all([stopBackend(), closeServer(staticServer)]);
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
    const appPort = await findAvailablePort(
      appPreferredPort,
      new Set([backendPort]),
    );
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
