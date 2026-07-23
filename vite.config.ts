import { type ChildProcess, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';
import { checker } from 'vite-plugin-checker';

const backendServerPort =
  Number(process.env.PYTHON_BACKEND_PORT ?? process.env.MERGE_SERVER_PORT) ||
  4095;
const backendServerDir =
  process.env.PYTHON_BACKEND_DIR ??
  process.env.MERGE_SERVER_DIR ??
  path.resolve(import.meta.dirname, '..');
const lehomePython = path.resolve(
  import.meta.dirname,
  '..',
  'ROBOTICS-lehome-challenge',
  '.venv',
  'bin',
  'python',
);
const backendPython =
  process.env.PYTHON_BACKEND_PYTHON ??
  (fs.existsSync(lehomePython) ? lehomePython : 'python3');

/**
 * Vite plugin that starts scripts/backend_server.py alongside the dev server.
 *
 * Set `PYTHON_BACKEND_DIR` to point the server at a specific directory of
 * HDF5 files. `MERGE_SERVER_DIR` is still accepted for backwards compatibility.
 *
 *   PYTHON_BACKEND_DIR=/path/to/datasets pnpm start
 */
function backendServer(): Plugin {
  let child: ChildProcess | null = null;
  let stopping = false;
  const port = backendServerPort;
  const dir = backendServerDir;

  async function kill(): Promise<void> {
    const runningChild = child;
    if (runningChild) {
      await new Promise<void>((resolve) => {
        let closed = false;
        const timeout = setTimeout(() => {
          if (!closed) {
            runningChild.kill('SIGKILL');
          }
          resolve();
        }, 2500);

        runningChild.once('close', () => {
          closed = true;
          clearTimeout(timeout);
          resolve();
        });

        runningChild.kill('SIGTERM');
        child = null;
      });
    }
  }

  function killSync() {
    if (child) {
      child.kill('SIGTERM');
      child = null;
    }
  }

  async function shutdown(signal: NodeJS.Signals): Promise<void> {
    if (stopping) {
      return;
    }

    stopping = true;
    await kill();
    process.removeAllListeners(signal);
    process.kill(process.pid, signal);
  }

  return {
    name: 'backend-server',
    apply: 'serve',

    configureServer(server) {
      const script = path.resolve(
        import.meta.dirname,
        'scripts/backend_server.py',
      );

      child = spawn(
        backendPython,
        [script, '--dir', dir, '--port', String(port)],
        {
          stdio: ['ignore', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONUNBUFFERED: '1' },
        },
      );

      child.stdout?.on('data', (data: Buffer) => {
        const text = data.toString().trim();
        if (text) {
          server.config.logger.info(`  [backend-server] ${text}`);
        }
      });

      child.stderr?.on('data', (data: Buffer) => {
        const text = data.toString().trim();
        if (text) {
          server.config.logger.error(`  [backend-server] ${text}`);
        }
      });

      child.on('exit', (code) => {
        if (code !== null && code !== 0) {
          server.config.logger.error(
            `  [backend-server] exited with code ${String(code)}`,
          );
        }
        child = null;
      });

      child.on('error', (error) => {
        server.config.logger.error(
          `  [backend-server] failed to start: ${error.message}`,
        );
        child = null;
      });

      // Clean up on Vite shutdown. Signal listeners must explicitly exit,
      // otherwise Ctrl+C only stops the child backend and leaves Vite alive.
      process.on('exit', killSync);
      process.once('SIGINT', () => {
        void shutdown('SIGINT');
      });
      process.once('SIGTERM', () => {
        void shutdown('SIGTERM');
      });
      process.once('SIGHUP', () => {
        void shutdown('SIGHUP');
      });
    },

    async closeBundle() {
      await kill();
    },
  };
}

export default defineConfig({
  server: { open: true },
  define: {
    __PYTHON_BACKEND_PORT__: JSON.stringify(backendServerPort),
    global: 'globalThis',
  },
  build: { sourcemap: process.env.SOURCEMAP === '1' },

  plugins: [
    react(),
    { ...checker({ typescript: true }), apply: 'serve' }, // dev only to reduce build time
    backendServer(),
  ],

  // Import HDF5 compression plugins as static assets
  assetsInclude: ['**/*.so'],
});
