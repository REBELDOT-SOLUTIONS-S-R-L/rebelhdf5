import { fetchBuffer } from '../fetch-utils';
import { FileService, type H5File } from '../stores';
import type {
  DemoInfo,
  DemoRow,
  DatasetProcessingRequest,
  DatasetProcessingResult,
  DatasetProcessingSourceInfo,
  DemoVideoFrames,
  DemoVideoInfo,
  DemoVideoKey,
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
};

type LoadDemoVideoResult = DemoVideoInfo & {
  framesBuffer: ArrayBuffer;
};

type PoseTraceWorkerRequest =
  | { id: number; type: 'openLocalSource'; payload: OpenLocalSourcePayload }
  | { id: number; type: 'openRemoteSource'; payload: OpenRemoteSourcePayload }
  | { id: number; type: 'loadDemoRows'; payload: LoadDemoRowsPayload }
  | { id: number; type: 'getDatasetProcessingInfo'; payload: GetDatasetProcessingInfoPayload }
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
        | DatasetProcessingResult
        | DemoVideoInfo[]
        | LoadDemoVideoResult
        | null;
    }
  | { id: number; ok: false; error: string };

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
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
    const { id } = event.data;
    const pending = pendingRequests.get(id);
    if (!pending) {
      return;
    }

    pendingRequests.delete(id);
    if (event.data.ok) {
      pending.resolve(event.data.result);
      return;
    }

    pending.reject(new Error(event.data.error));
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

export async function openPoseTraceSource(file: H5File): Promise<PoseTraceSource> {
  let result: OpenSourceResult;

  if (file.service === FileService.Local) {
    result = await callWorker<OpenSourceResult>('openLocalSource', { file: file.file });
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

export async function processDataset(
  request: DatasetProcessingRequest,
): Promise<DatasetProcessingResult> {
  const result = await callWorker<DatasetProcessingResult>('processDataset', request);
  return {
    ...result,
    fileBuffer: result.fileBuffer,
  };
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
