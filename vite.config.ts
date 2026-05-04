import { type ChildProcess, spawn } from 'node:child_process';
import path from 'node:path';

import react from '@vitejs/plugin-react-swc';
import { type Plugin, defineConfig } from 'vite';
import { checker } from 'vite-plugin-checker';

const mergeServerPort = Number(process.env.MERGE_SERVER_PORT) || 4095;

/**
 * Vite plugin that starts scripts/merge_server.py alongside the dev server.
 *
 * Set the `MERGE_SERVER_DIR` env var to point the server at a specific
 * directory of HDF5 files. The server is auto-detected by the web app.
 *
 *   MERGE_SERVER_DIR=/path/to/datasets pnpm start
 */
function mergeServer(): Plugin {
  let child: ChildProcess | null = null;
  const port = mergeServerPort;
  const dir = process.env.MERGE_SERVER_DIR || path.resolve(__dirname, '..');

  function kill() {
    if (!child) {
      return;
    }

    child.kill('SIGTERM');
    child = null;
  }

  return {
    name: 'merge-server',
    apply: 'serve',

    configureServer() {
      const script = path.resolve(__dirname, 'scripts/merge_server.py');

      child = spawn('python3', [script, '--dir', dir, '--port', String(port)], {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      child.stdout?.on('data', (data: Buffer) => {
        const text = data.toString().trim();
        if (text) {
          console.log(`  [merge-server] ${text}`);
        }
      });

      child.stderr?.on('data', (data: Buffer) => {
        const text = data.toString().trim();
        if (text) {
          console.error(`  [merge-server] ${text}`);
        }
      });

      child.on('exit', (code) => {
        if (code !== null && code !== 0) {
          console.error(`  [merge-server] exited with code ${String(code)}`);
        }
        child = null;
      });

      child.on('error', (error) => {
        console.error(`  [merge-server] failed to start: ${error.message}`);
        child = null;
      });

      // Clean up on Vite shutdown.
      process.on('exit', kill);
      process.on('SIGINT', kill);
      process.on('SIGTERM', kill);
    },

    closeBundle() {
      kill();
    },
  };
}

export default defineConfig({
  server: { open: true },
  define: {
    __MERGE_SERVER_PORT__: JSON.stringify(mergeServerPort),
  },
  plugins: [
    react(),
    { ...checker({ typescript: true }), apply: 'serve' }, // dev only to reduce build time
    mergeServer(),
  ],

  // Import HDF5 compression plugins as static assets
  assetsInclude: ['**/*.so'],

  // `es2020` required by @h5web/h5wasm for BigInt `123n` notation support
  optimizeDeps: { esbuildOptions: { target: 'es2020' } },
  build: {
    target: 'es2020',
    sourcemap: true,
  },
});
