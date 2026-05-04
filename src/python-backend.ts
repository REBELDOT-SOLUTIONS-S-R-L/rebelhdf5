/**
 * Client for the local Python backend server (scripts/backend_server.py).
 *
 * The server is auto-detected on localhost:4095. When available it provides
 * native HDF5 cut/merge/append which is orders of magnitude faster than the
 * WASM path for large (multi-GB) files — especially those with video data.
 */

import type {
  DatasetProcessingCutRange,
  DatasetProcessingOperation,
  DatasetProcessingProgress,
  DatasetProcessingResultMeta,
} from './pose-trace/types';

const DEFAULT_PORT = __PYTHON_BACKEND_PORT__;
const BASE_URL = `http://127.0.0.1:${DEFAULT_PORT}`;
const HEALTH_TIMEOUT_MS = 2000;

export const PYTHON_BACKEND_BASE_URL = BASE_URL;

export interface PythonBackendStatus {
  available: boolean;
  rootDir: string | null;
  version: number | null;
  indexing?: boolean;
  indexReady?: boolean;
  indexedFileCount?: number;
  indexError?: string | null;
}

export interface PythonFileEntry {
  name: string;
  path: string;
  relativePath?: string;
  size: number;
}

export interface PythonFileListingResult {
  directory: string;
  recursive: boolean;
  files: PythonFileEntry[];
}

export interface PythonResolvedFilesResult {
  resolved: Record<string, string | null>;
  indexing?: boolean;
  indexReady?: boolean;
  indexedFileCount?: number;
  indexError?: string | null;
}

export interface PythonScanFileInfo {
  name: string;
  path: string;
  demoCount: number;
  demoNames: string[];
  keys: string[];
  keyCounts?: Record<string, number>;
}

export interface PythonScanResult {
  files: PythonScanFileInfo[];
  commonKeys: string[];
}

export interface PythonProcessRequest {
  paths: string[];
  selectedKeys: string[];
  outputName: string;
  operation: DatasetProcessingOperation;
  cutRange?: DatasetProcessingCutRange;
}

export interface PythonProcessCallbacks {
  onProgress?: (progress: DatasetProcessingProgress) => void;
}

export type PythonProcessResult = DatasetProcessingResultMeta & {
  fileSize: number;
  downloadUrl: string;
};

async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const body: { error?: string } = await response.json();
    return body.error ?? `Server returned ${String(response.status)}`;
  } catch {
    return `Server returned ${String(response.status)}`;
  }
}

/** Check whether the Python server is running. */
export async function checkBackend(timeoutMs = HEALTH_TIMEOUT_MS): Promise<PythonBackendStatus> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(`${BASE_URL}/api/health`, {
      signal: controller.signal,
    });

    if (!response.ok) {
      return { available: false, rootDir: null, version: null };
    }

    const data: {
      status: string;
      rootDir: string;
      version?: number;
      indexing?: boolean;
      indexReady?: boolean;
      indexedFileCount?: number;
      indexError?: string | null;
    } = await response.json();
    return {
      available: data.status === 'ok',
      rootDir: data.rootDir,
      version: Number.isFinite(data.version) ? data.version ?? null : null,
      indexing: data.indexing,
      indexReady: data.indexReady,
      indexedFileCount: data.indexedFileCount,
      indexError: data.indexError ?? null,
    };
  } catch {
    return { available: false, rootDir: null, version: null };
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

export function pollBackendStatus(
  onStatus: (status: PythonBackendStatus) => void,
  unavailableIntervalMs = 1000,
  availableIntervalMs = 10000,
): () => void {
  let cancelled = false;
  let timeoutId: ReturnType<typeof globalThis.setTimeout> | null = null;

  async function tick() {
    const status = await checkBackend();
    if (cancelled) {
      return;
    }

    onStatus(status);
    timeoutId = globalThis.setTimeout(
      () => {
        void tick();
      },
      status.available && !status.indexing ? availableIntervalMs : unavailableIntervalMs,
    );
  }

  void tick();

  return () => {
    cancelled = true;
    if (timeoutId) {
      globalThis.clearTimeout(timeoutId);
    }
  };
}

/** List HDF5 files in a directory on the server. */
export async function listFiles(
  directory?: string,
  recursive = false,
): Promise<PythonFileListingResult> {
  const query = new URLSearchParams();
  if (directory) {
    query.set('dir', directory);
  }
  if (recursive) {
    query.set('recursive', '1');
  }
  const params = query.size > 0 ? `?${query.toString()}` : '';
  const response = await fetch(`${BASE_URL}/api/files${params}`);

  if (!response.ok) {
    const body: { error?: string } = await response.json();
    throw new Error(body.error ?? `Server returned ${String(response.status)}`);
  }

  const data: PythonFileListingResult = await response.json();
  return data;
}

/** Resolve filenames to absolute paths on the server. */
export async function resolveFiles(
  names: string[],
): Promise<PythonResolvedFilesResult> {
  const response = await fetch(`${BASE_URL}/api/resolve-files`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names }),
  });

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }

  const data: PythonResolvedFilesResult = await response.json();
  return data;
}

/** Scan files for their keys and demo counts. */
export async function scanFiles(paths: string[]): Promise<PythonScanResult> {
  const response = await fetch(`${BASE_URL}/api/scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  });

  if (!response.ok) {
    const body: { error?: string } = await response.json();
    throw new Error(body.error ?? `Server returned ${String(response.status)}`);
  }

  const data: PythonScanResult = await response.json();
  return data;
}

/** Run a dataset processing operation with SSE progress streaming. */
export async function runProcess(
  request: PythonProcessRequest,
  callbacks: PythonProcessCallbacks,
): Promise<PythonProcessResult> {
  const payload = JSON.stringify({
    paths: request.paths,
    selectedKeys: request.selectedKeys,
    outputName: request.outputName,
    operation: request.operation,
    cutRange: request.cutRange,
  });

  const endpoints = request.operation === 'merge'
    ? ['/api/process', '/api/merge']
    : ['/api/process'];

  let response: Response | null = null;
  let lastError: string | null = null;

  for (const endpoint of endpoints) {
    response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
    });

    if (response.ok) {
      break;
    }

    lastError = await parseErrorResponse(response);

    if (response.status === 404 && endpoint === '/api/process' && request.operation === 'merge') {
      continue;
    }

    if (response.status === 404 && endpoint === '/api/process') {
      throw new Error(
        'Your Python backend is outdated and does not support this operation. Restart `scripts/backend_server.py` to load the current API.',
      );
    }

    throw new Error(lastError);
  }

  if (!response?.ok) {
    throw new Error(lastError ?? 'Failed to start dataset processing.');
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Server did not return a readable stream.');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult: PythonProcessResult | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';

    for (const part of parts) {
      const trimmed = part.trim();
      if (!trimmed.startsWith('data: ')) {
        continue;
      }

      let event: Record<string, unknown>;
      try {
        event = JSON.parse(trimmed.slice(6));
      } catch {
        continue;
      }

      if (event.type === 'progress') {
        callbacks.onProgress?.({
          phase: event.phase as DatasetProcessingProgress['phase'],
          overallDemoIndex: event.overallDemoIndex as number,
          overallDemoCount: event.overallDemoCount as number,
          currentSourceName: event.currentSourceName as string,
          currentDemoName: event.currentDemoName as string,
        });
      } else if (event.type === 'done') {
        const fileName = event.fileName as string;
        finalResult = {
          fileName,
          demoCount: event.demoCount as number,
          selectedKeyCount: event.selectedKeyCount as number,
          fileSize: event.fileSize as number,
          downloadUrl: `${BASE_URL}/api/download/${encodeURIComponent(fileName)}`,
        };
      } else if (event.type === 'error') {
        throw new Error(event.message as string);
      }
    }
  }

  if (!finalResult) {
    throw new Error('Processing stream ended without a completion event.');
  }

  return finalResult;
}
