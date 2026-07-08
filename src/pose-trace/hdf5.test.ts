import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { fetchBuffer } from '../fetch-utils';
import { FileService, type H5File } from '../stores';
import { loadDemoRows, openPoseTraceSource, processDataset } from './hdf5';
import type { PoseTraceSource } from './types';

vi.mock('../fetch-utils', () => ({
  fetchBuffer: vi.fn(),
}));

const fetchBufferMock = vi.mocked(fetchBuffer);

interface PostedMessage {
  id: number;
  type: string;
  payload: unknown;
}

// A controllable stand-in for the real Web Worker. The hdf5 module lazily
// creates a single worker and caches it, so all tests share one instance.
class FakeWorker {
  public static instances: FakeWorker[] = [];
  public onmessage: ((event: { data: unknown }) => void) | null = null;
  public onerror: ((event: { message: string }) => void) | null = null;
  public readonly posted: Array<{
    msg: PostedMessage;
    transfer: Transferable[];
  }> = [];

  public constructor() {
    FakeWorker.instances.push(this);
  }

  public postMessage(msg: PostedMessage, transfer: Transferable[] = []): void {
    this.posted.push({ msg, transfer });
  }

  public terminate(): void {}

  /** Simulate a message coming back from the worker. */
  public emit(data: unknown): void {
    this.onmessage?.({ data });
  }

  /** Simulate a worker-level error. */
  public emitError(message: string): void {
    this.onerror?.({ message });
  }

  public lastPosted(): { msg: PostedMessage; transfer: Transferable[] } {
    const last = this.posted.at(-1);
    if (!last) {
      throw new Error('No message has been posted to the worker.');
    }
    return last;
  }
}

// Install the fake Worker before the module lazily instantiates one.
vi.stubGlobal('Worker', FakeWorker as unknown as typeof Worker);

function theWorker(): FakeWorker {
  const instance = FakeWorker.instances[0];
  if (!instance) {
    throw new Error('Worker has not been created yet.');
  }
  return instance;
}

function remoteFile(name = 'remote.hdf5'): H5File {
  return {
    url: `u-${name}`,
    name,
    service: FileService.Url,
    resolvedUrl: `https://example.com/${name}`,
  };
}

function localFile(name = 'local.hdf5', serverPath?: string): H5File {
  return {
    url: `u-${name}`,
    name,
    service: FileService.Local,
    resolvedUrl: `blob:${name}`,
    serverPath,
    file: new File([new Uint8Array([1, 2, 3])], name),
  };
}

function fakeSource(sourceId: string): PoseTraceSource {
  return {
    sourceId,
    datasetName: sourceId,
    demos: [],
    articulation: null,
    cleanup: () => {},
  };
}

beforeEach(() => {
  globalThis.rebelHdf5Desktop = undefined;
});

afterEach(() => {
  // Drain any pending message handlers between tests by resolving nothing new.
  vi.clearAllMocks();
});

describe('callWorker request routing', () => {
  it('increments request ids and resolves the matching pending promise', async () => {
    const rowsA = [{ step: 0 }] as unknown;
    const rowsB = [{ step: 1 }] as unknown;

    const pA = loadDemoRows(fakeSource('sa'), 'demo_0');
    const pB = loadDemoRows(fakeSource('sb'), 'demo_1');

    const worker = theWorker();
    const posts = worker.posted.slice(-2);
    expect(posts).toHaveLength(2);
    const [postA, postB] = posts;
    // IDs are sequential and distinct.
    expect(postB.msg.id).toBe(postA.msg.id + 1);
    expect(postA.msg.type).toBe('loadDemoRows');
    expect(postA.msg.payload).toEqual({ sourceId: 'sa', demoName: 'demo_0' });

    // Respond out of order; each promise must resolve with its own result.
    worker.emit({ id: postB.msg.id, ok: true, result: rowsB });
    worker.emit({ id: postA.msg.id, ok: true, result: rowsA });

    await expect(pA).resolves.toBe(rowsA);
    await expect(pB).resolves.toBe(rowsB);
  });

  it('rejects a request when the worker replies with ok:false', async () => {
    const p = loadDemoRows(fakeSource('s'), 'demo_0');
    const { msg } = theWorker().lastPosted();
    theWorker().emit({ id: msg.id, ok: false, error: 'boom' });
    await expect(p).rejects.toThrow('boom');
  });

  it('ignores responses for unknown ids', () => {
    // Should not throw when a stray message arrives for an unregistered id.
    expect(() =>
      theWorker().emit({ id: 999_999, ok: true, result: null }),
    ).not.toThrow();
  });
});

describe('worker.onerror', () => {
  it('rejects all pending requests with the error message', async () => {
    const p1 = loadDemoRows(fakeSource('s1'), 'demo_0');
    const p2 = loadDemoRows(fakeSource('s2'), 'demo_1');

    theWorker().emitError('worker crashed');

    await expect(p1).rejects.toThrow('worker crashed');
    await expect(p2).rejects.toThrow('worker crashed');
  });

  it('falls back to a default message when none is provided', async () => {
    const p = loadDemoRows(fakeSource('s'), 'demo_0');
    theWorker().emitError('');
    await expect(p).rejects.toThrow('Pose Trace worker failed.');
  });
});

describe('processDataset', () => {
  it('forwards progress and chunk callbacks, then resolves on the final result', async () => {
    const onProgress = vi.fn();
    const onChunk = vi.fn();
    const meta = { fileName: 'out.hdf5' } as unknown;

    const promise = processDataset({ operation: 'merge' } as never, {
      onProgress,
      onChunk,
    });

    const { msg } = theWorker().lastPosted();
    expect(msg.type).toBe('processDataset');

    const progress = { phase: 'copying' };
    theWorker().emit({ id: msg.id, type: 'progress', progress });
    const chunk = new ArrayBuffer(8);
    theWorker().emit({
      id: msg.id,
      type: 'chunk',
      data: chunk,
      index: 2,
      total: 5,
    });

    // Progress/chunk messages must not resolve the promise.
    expect(onProgress).toHaveBeenCalledWith(progress);
    expect(onChunk).toHaveBeenCalledWith(chunk, 2, 5);

    theWorker().emit({ id: msg.id, ok: true, result: meta });
    await expect(promise).resolves.toBe(meta);
  });
});

describe('openPoseTraceSource', () => {
  const openResult = {
    sourceId: 'src-1',
    datasetName: 'my-dataset',
    demos: [],
    articulation: null,
  };

  it('reads a local file fresh via rebelHdf5Desktop and transfers the buffer', async () => {
    const buffer = new ArrayBuffer(16);
    const readFile = vi.fn().mockResolvedValue(buffer);
    globalThis.rebelHdf5Desktop = { readFile };

    const promise = openPoseTraceSource(
      localFile('local.hdf5', '/data/local.hdf5'),
    );

    await vi.waitFor(() => {
      expect(theWorker().lastPosted().msg.type).toBe('openRemoteSource');
    });
    const { msg, transfer } = theWorker().lastPosted();
    expect(readFile).toHaveBeenCalledWith('/data/local.hdf5');
    expect(msg.payload).toEqual({ buffer, name: 'local.hdf5' });
    expect(transfer).toEqual([buffer]);
    expect(fetchBufferMock).not.toHaveBeenCalled();

    theWorker().emit({ id: msg.id, ok: true, result: openResult });
    const source = await promise;
    expect(source.sourceId).toBe('src-1');
    expect(source.datasetName).toBe('my-dataset');
  });

  it('falls back to openLocalSource when no fresh buffer is available', async () => {
    // No rebelHdf5Desktop and no serverPath → uses the in-memory File blob.
    const promise = openPoseTraceSource(localFile('local.hdf5'));

    await vi.waitFor(() => {
      expect(theWorker().lastPosted().msg.type).toBe('openLocalSource');
    });
    const { msg } = theWorker().lastPosted();
    expect((msg.payload as { file: File }).file).toBeInstanceOf(File);

    theWorker().emit({ id: msg.id, ok: true, result: openResult });
    await expect(promise).resolves.toMatchObject({ sourceId: 'src-1' });
  });

  it('fetches remote files via fetchBuffer', async () => {
    const buffer = new ArrayBuffer(32);
    fetchBufferMock.mockResolvedValue(buffer);

    const promise = openPoseTraceSource(remoteFile('remote.hdf5'));

    await vi.waitFor(() => {
      expect(theWorker().lastPosted().msg.type).toBe('openRemoteSource');
    });
    const { msg, transfer } = theWorker().lastPosted();
    expect(fetchBufferMock).toHaveBeenCalledWith(
      'https://example.com/remote.hdf5',
    );
    expect(msg.payload).toEqual({ buffer, name: 'remote.hdf5' });
    expect(transfer).toEqual([buffer]);

    theWorker().emit({ id: msg.id, ok: true, result: openResult });
    await expect(promise).resolves.toMatchObject({ sourceId: 'src-1' });
  });

  it('derives datasetName from the filename when the worker returns an empty name', async () => {
    fetchBufferMock.mockResolvedValue(new ArrayBuffer(4));
    const promise = openPoseTraceSource(remoteFile('scene.hdf5'));

    await vi.waitFor(() => {
      expect(theWorker().lastPosted().msg.type).toBe('openRemoteSource');
    });
    const { msg } = theWorker().lastPosted();
    theWorker().emit({
      id: msg.id,
      ok: true,
      result: { ...openResult, datasetName: '' },
    });
    const source = await promise;
    // Extension stripped from the filename.
    expect(source.datasetName).toBe('scene');
  });

  it('cleanup posts a closeSource message and swallows errors', async () => {
    fetchBufferMock.mockResolvedValue(new ArrayBuffer(4));
    const promise = openPoseTraceSource(remoteFile('c.hdf5'));
    await vi.waitFor(() => {
      expect(theWorker().lastPosted().msg.type).toBe('openRemoteSource');
    });
    const openMsg = theWorker().lastPosted().msg;
    theWorker().emit({ id: openMsg.id, ok: true, result: openResult });
    const source = await promise;

    source.cleanup();
    const { msg } = theWorker().lastPosted();
    expect(msg.type).toBe('closeSource');
    expect(msg.payload).toEqual({ sourceId: 'src-1' });

    // Rejecting the cleanup request must not surface an unhandled rejection.
    theWorker().emit({ id: msg.id, ok: false, error: 'already closed' });
    await Promise.resolve();
  });
});
