import { fetchBuffer } from '../fetch-utils';
import { FileService, type H5File } from '../stores';
import type {
  ObjectDistributionRequest,
  ObjectDistributionResult,
  DemoInfo,
  DemoRow,
  DatasetProcessingProgress,
  DatasetProcessingRequest,
  DatasetProcessingResultMeta,
  DatasetProcessingSourceInfo,
  DemoVideoFrames,
  DemoVideoInfo,
  DemoVideoKey,
  ParsedArticulation,
  PoseTraceSource,
} from './types';

type OpenLocalSourcePayload = {
  file: File;
};

type OpenRemoteSourcePayload = {
  buffer: ArrayBuffer;
  name: string;
};

type LoadDemoRowsPayload = {
  sourceId: string;
  demoName: string;
};

type GetDatasetProcessingInfoPayload = {
  sourceId: string;
};

type LoadObjectDistributionPayload = ObjectDistributionRequest;
type ProcessDatasetPayload = DatasetProcessingRequest;

type ListDemoVideosPayload = {
  sourceId: string;
  demoName: string;
};

type LoadDemoVideoPayload = {
  sourceId: string;
  demoName: string;
  videoKey: DemoVideoKey;
};

type CloseSourcePayload = {
  sourceId: string;
};

type OpenSourceResult = {
  sourceId: string;
  datasetName: string;
  demos: DemoInfo[];
  articulation: ParsedArticulation | null;
};

type LoadDemoVideoResult = DemoVideoInfo & {
  framesBuffer: ArrayBuffer;
};

type PoseTraceWorkerRequest =
  | { id: number; type: 'openLocalSource'; payload: OpenLocalSourcePayload }
  | { id: number; type: 'openRemoteSource'; payload: OpenRemoteSourcePayload }
  | { id: number; type: 'loadDemoRows'; payload: LoadDemoRowsPayload }
  | { id: number; type: 'getDatasetProcessingInfo'; payload: GetDatasetProcessingInfoPayload }
  | { id: number; type: 'loadObjectDistribution'; payload: LoadObjectDistributionPayload }
  | { id: number; type: 'processDataset'; payload: ProcessDatasetPayload }
  | { id: number; type: 'listDemoVideos'; payload: ListDemoVideosPayload }
  | { id: number; type: 'loadDemoVideo'; payload: LoadDemoVideoPayload }
  | { id: number; type: 'closeSource'; payload: CloseSourcePayload };

type PoseTraceWorkerResponse =
  | {
      id: number;
      ok: true;
      result:
        | OpenSourceResult
        | DemoRow[]
        | DatasetProcessingSourceInfo
        | ObjectDistributionResult
        | DatasetProcessingResultMeta
        | DemoVideoInfo[]
        | LoadDemoVideoResult
        | null;
    }
  | { id: number; ok: false; error: string }
  | { id: number; type: 'progress'; progress: DatasetProcessingProgress }
  | { id: number; type: 'chunk'; data: ArrayBuffer; index: number; total: number };

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
  onProgress?: (progress: DatasetProcessingProgress) => void;
  onChunk?: (chunk: ArrayBuffer, index: number, total: number) => void;
};

let worker: Worker | null = null;
let nextRequestId = 0;
const pendingRequests = new Map<number, PendingRequest>();

function stripExtension(filename: string): string {
  return filename.replace(/\.(hdf5|h5)$/i, '');
}

function ensureWorker(): Worker {
  if (worker) {
    return worker;
  }

  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (event: MessageEvent<PoseTraceWorkerResponse>) => {
    const data = event.data;
    const pending = pendingRequests.get(data.id);
    if (!pending) {
      return;
    }

    if ('type' in data && data.type === 'progress') {
      pending.onProgress?.(data.progress);
      return;
    }

    if ('type' in data && data.type === 'chunk') {
      pending.onChunk?.(data.data, data.index, data.total);
      return;
    }

    pendingRequests.delete(data.id);
    if (data.ok) {
      pending.resolve(data.result);
      return;
    }

    pending.reject(new Error(data.error));
  };
  worker.onerror = (event) => {
    const message = event.message || 'Pose Trace worker failed.';
    for (const pending of pendingRequests.values()) {
      pending.reject(new Error(message));
    }
    pendingRequests.clear();
  };

  return worker;
}

function callWorker<T>(
  type: PoseTraceWorkerRequest['type'],
  payload: PoseTraceWorkerRequest['payload'],
  transfer: Transferable[] = [],
): Promise<T> {
  const instance = ensureWorker();
  const id = nextRequestId;
  nextRequestId += 1;

  return new Promise<T>((resolve, reject) => {
    pendingRequests.set(id, {
      resolve: (value) => resolve(value as T),
      reject,
    });
    instance.postMessage({ id, type, payload } as PoseTraceWorkerRequest, transfer);
  });
}

async function readLocalFileFresh(serverPath: string): Promise<ArrayBuffer | null> {
  const readFile = globalThis.rebelHdf5Desktop?.readFile;
  if (typeof readFile !== 'function') {
    return null;
  }

  try {
    return await readFile(serverPath);
  } catch {
    return null;
  }
}

export async function openPoseTraceSource(file: H5File): Promise<PoseTraceSource> {
  let result: OpenSourceResult;

  if (file.service === FileService.Local) {
    // Prefer reading fresh bytes from disk via the Electron preload when
    // possible. The in-memory File blob becomes unreadable after the file is
    // modified externally (e.g. articulation attrs rewritten through the
    // Python backend), so the WORKERFS mount path fails with a stale-
    // reference FileReaderSync error.
    const freshBuffer = file.serverPath
      ? await readLocalFileFresh(file.serverPath)
      : null;

    if (freshBuffer) {
      result = await callWorker<OpenSourceResult>(
        'openRemoteSource',
        { buffer: freshBuffer, name: file.name },
        [freshBuffer],
      );
    } else {
      result = await callWorker<OpenSourceResult>('openLocalSource', { file: file.file });
    }
  } else {
    const buffer = await fetchBuffer(file.resolvedUrl);
    result = await callWorker<OpenSourceResult>(
      'openRemoteSource',
      { buffer, name: file.name },
      [buffer],
    );
  }

  return {
    sourceId: result.sourceId,
    datasetName: result.datasetName || stripExtension(file.name),
    demos: result.demos,
    articulation: result.articulation,
    cleanup: () => {
      void callWorker<null>('closeSource', { sourceId: result.sourceId }).catch(() => {
        // Ignore best-effort cleanup failures during route changes/unmounts.
      });
    },
  };
}

export function loadDemoRows(source: PoseTraceSource, demoName: string): Promise<DemoRow[]> {
  return callWorker<DemoRow[]>('loadDemoRows', {
    sourceId: source.sourceId,
    demoName,
  });
}

export function getDatasetProcessingInfo(
  source: PoseTraceSource,
): Promise<DatasetProcessingSourceInfo> {
  return callWorker<DatasetProcessingSourceInfo>('getDatasetProcessingInfo', {
    sourceId: source.sourceId,
  });
}

export function loadObjectDistribution(
  request: ObjectDistributionRequest,
): Promise<ObjectDistributionResult> {
  return callWorker<ObjectDistributionResult>('loadObjectDistribution', request);
}

export function processDataset(
  request: DatasetProcessingRequest,
  callbacks: {
    onProgress?: (progress: DatasetProcessingProgress) => void;
    onChunk?: (chunk: ArrayBuffer, index: number, total: number) => void;
  },
): Promise<DatasetProcessingResultMeta> {
  const instance = ensureWorker();
  const id = nextRequestId;
  nextRequestId += 1;

  return new Promise<DatasetProcessingResultMeta>((resolve, reject) => {
    pendingRequests.set(id, {
      resolve: (value) => resolve(value as DatasetProcessingResultMeta),
      reject,
      onProgress: callbacks.onProgress,
      onChunk: callbacks.onChunk,
    });
    instance.postMessage({ id, type: 'processDataset', payload: request } as never);
  });
}

export function listDemoVideos(
  source: PoseTraceSource,
  demoName: string,
): Promise<DemoVideoInfo[]> {
  return callWorker<DemoVideoInfo[]>('listDemoVideos', {
    sourceId: source.sourceId,
    demoName,
  });
}

export async function loadDemoVideoFrames(
  source: PoseTraceSource,
  demoName: string,
  videoKey: DemoVideoKey,
): Promise<DemoVideoFrames> {
  const result = await callWorker<LoadDemoVideoResult>('loadDemoVideo', {
    sourceId: source.sourceId,
    demoName,
    videoKey,
  });

  return {
    ...result,
    frames: new Uint8Array(result.framesBuffer),
  };
}
