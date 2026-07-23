import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  addBackendRoot,
  checkBackend,
  type DatasetArticulation,
  getDatasetAttributes,
  listFiles,
  pollBackendStatus,
  PYTHON_BACKEND_BASE_URL,
  resolveFiles,
  runLeRobotConvert,
  runProcess,
  scanFiles,
  updateDatasetArticulation,
  type PythonLeRobotConvertRequest,
} from './python-backend';

type FetchMock = ReturnType<typeof vi.fn<typeof fetch>>;
let fetchMock: FetchMock;

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return Response.json(body, {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });
}

function assertString(value: unknown): asserts value is string {
  if (typeof value !== 'string') {
    throw new TypeError('Expected a string value.');
  }
}

function sseResponse(events: unknown[]): Response {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  return new Response(body, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

beforeEach(() => {
  fetchMock = vi.fn<typeof fetch>();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('PYTHON_BACKEND_BASE_URL', () => {
  it('points at 127.0.0.1 with the configured port', () => {
    expect(PYTHON_BACKEND_BASE_URL).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/u);
  });
});

describe('checkBackend', () => {
  it('reports available=true when the server returns status=ok', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        status: 'ok',
        rootDir: '/data',
        rootDirs: ['/data', '/extra'],
        version: 3,
        indexReady: true,
        indexedFileCount: 12,
      }),
    );
    const result = await checkBackend();
    expect(result).toMatchObject({
      available: true,
      rootDir: '/data',
      rootDirs: ['/data', '/extra'],
      version: 3,
      indexReady: true,
      indexedFileCount: 12,
      indexError: null,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `${PYTHON_BACKEND_BASE_URL}/api/health`,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('reports available=false on non-ok responses', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 500 }));
    const result = await checkBackend();
    expect(result.available).toBe(false);
    expect(result.rootDir).toBeNull();
  });

  it('reports available=false when fetch throws', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('refused'));
    const result = await checkBackend();
    expect(result.available).toBe(false);
  });
});

describe('listFiles', () => {
  it('omits query params when no directory and not recursive', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ directory: '/', recursive: false, files: [] }),
    );
    await listFiles();
    expect(fetchMock).toHaveBeenCalledWith(
      `${PYTHON_BACKEND_BASE_URL}/api/files`,
    );
  });

  it('encodes the directory and recursive flag into the query', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ directory: '/data', recursive: true, files: [] }),
    );
    await listFiles('/data', true);
    const calledWith = fetchMock.mock.calls[0]?.[0] as string;
    const url = new URL(calledWith);
    expect(url.pathname).toBe('/api/files');
    expect(url.searchParams.get('dir')).toBe('/data');
    expect(url.searchParams.get('recursive')).toBe('1');
  });

  it('throws the server-supplied error message on failure', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: 'no such dir' }, { status: 400 }),
    );
    await expect(listFiles('/missing')).rejects.toThrow('no such dir');
  });
});

describe('resolveFiles', () => {
  it('POSTs cleaned-up name→path map and returns the parsed body', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ resolved: { 'a.h5': '/abs/a.h5' } }),
    );
    const result = await resolveFiles(['a.h5'], {
      'a.h5': '/abs/a.h5',
      'skip.h5': undefined,
    });

    expect(result.resolved['a.h5']).toBe('/abs/a.h5');
    expect(fetchMock).toHaveBeenCalledWith(
      `${PYTHON_BACKEND_BASE_URL}/api/resolve-files`,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          names: ['a.h5'],
          paths: { 'a.h5': '/abs/a.h5' },
        }),
      }),
    );
  });

  it('throws on non-ok responses', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: 'forbidden path' }, { status: 403 }),
    );
    await expect(resolveFiles([])).rejects.toThrow('forbidden path');
  });
});

describe('getDatasetAttributes', () => {
  const result = {
    path: '/data/f.h5',
    attrs: { total: 12 },
    articulation: {
      name: 'robot',
      joint_number: 7,
      segmentation: {},
      end_effectors: {},
    },
    articulationSource: 'attribute',
    groups: [{ path: '/data', attrs: { total: 12 } }],
  };

  it('POSTs the path and returns the parsed attributes result', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(result));
    const payload = await getDatasetAttributes('/data/f.h5');

    expect(payload).toMatchObject({
      path: '/data/f.h5',
      articulationSource: 'attribute',
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `${PYTHON_BACKEND_BASE_URL}/api/dataset-attributes`,
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: '/data/f.h5' }),
      }),
    );
  });

  it('throws the backend error message on non-ok responses', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: 'no such file' }, { status: 404 }),
    );
    await expect(getDatasetAttributes('/missing.h5')).rejects.toThrow(
      'no such file',
    );
  });

  it('falls back to a status-based message when the error body is not JSON', async () => {
    fetchMock.mockResolvedValueOnce(new Response('boom', { status: 500 }));
    await expect(getDatasetAttributes('/x.h5')).rejects.toThrow(/500/u);
  });
});

describe('updateDatasetArticulation', () => {
  const articulation: DatasetArticulation = {
    name: 'robot',
    joint_number: 7,
    segmentation: { arm: { target: '[0:7]', obs: '[0:7]' } },
    end_effectors: { left: { pose: '[0:7]', gripper: '[7:8]' } },
  };

  it('POSTs path + articulation and returns the parsed result', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        path: '/data/f.h5',
        attrs: {},
        articulation,
        articulationSource: 'attribute',
      }),
    );
    const payload = await updateDatasetArticulation('/data/f.h5', articulation);

    expect(payload.articulation).toEqual(articulation);
    expect(fetchMock).toHaveBeenCalledWith(
      `${PYTHON_BACKEND_BASE_URL}/api/dataset-attributes/articulation`,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ path: '/data/f.h5', articulation }),
      }),
    );
  });

  it('throws on non-ok responses', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: 'write failed' }, { status: 400 }),
    );
    await expect(
      updateDatasetArticulation('/data/f.h5', articulation),
    ).rejects.toThrow('write failed');
  });
});

describe('addBackendRoot', () => {
  it('POSTs the path and parses the response', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        rootDirs: ['/r'],
        added: 1,
        indexedFileCount: 5,
        indexReady: true,
        indexing: false,
        indexError: null,
      }),
    );
    const result = await addBackendRoot('/r');
    expect(result.rootDirs).toEqual(['/r']);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ path: '/r' }),
      }),
    );
  });
});

describe('scanFiles', () => {
  it('POSTs the path list and returns the parsed body', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        files: [
          {
            name: 'a.h5',
            path: '/a.h5',
            demoCount: 3,
            demoNames: ['d0', 'd1', 'd2'],
            keys: ['x'],
          },
        ],
        commonKeys: ['x'],
      }),
    );
    const result = await scanFiles(['/a.h5']);
    expect(result.commonKeys).toEqual(['x']);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ paths: ['/a.h5'] }),
      }),
    );
  });
});

describe('runProcess (SSE streaming)', () => {
  it('forwards progress events and resolves with the done event', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        {
          type: 'progress',
          phase: 'reading',
          overallDemoIndex: 0,
          overallDemoCount: 2,
          currentSourceName: 'a.h5',
          currentDemoName: 'demo_0',
        },
        {
          type: 'progress',
          phase: 'writing',
          overallDemoIndex: 1,
          overallDemoCount: 2,
          currentSourceName: 'a.h5',
          currentDemoName: 'demo_1',
        },
        {
          type: 'done',
          fileName: 'merged.h5',
          demoCount: 2,
          selectedKeyCount: 4,
          fileSize: 1024,
        },
      ]),
    );
    const onProgress =
      vi.fn<NonNullable<Parameters<typeof runProcess>[1]['onProgress']>>();

    const result = await runProcess(
      {
        paths: ['/a.h5'],
        selectedKeys: ['x'],
        outputName: 'merged.h5',
        operation: 'merge',
      },
      { onProgress },
    );

    expect(onProgress).toHaveBeenCalledTimes(2);
    expect(onProgress).toHaveBeenNthCalledWith(1, {
      phase: 'reading',
      overallDemoIndex: 0,
      overallDemoCount: 2,
      currentSourceName: 'a.h5',
      currentDemoName: 'demo_0',
    });
    expect(result).toEqual({
      fileName: 'merged.h5',
      demoCount: 2,
      selectedKeyCount: 4,
      fileSize: 1024,
      downloadUrl: `${PYTHON_BACKEND_BASE_URL}/api/download/merged.h5`,
    });
  });

  it('falls back from /api/process to /api/merge on 404 for merge operations', async () => {
    fetchMock
      .mockResolvedValueOnce(
        Response.json({ error: 'unknown route' }, { status: 404 }),
      )
      .mockResolvedValueOnce(
        sseResponse([
          {
            type: 'done',
            fileName: 'merged.h5',
            demoCount: 1,
            selectedKeyCount: 1,
            fileSize: 1,
          },
        ]),
      );

    const result = await runProcess(
      {
        paths: ['/a.h5'],
        selectedKeys: ['x'],
        outputName: 'merged.h5',
        operation: 'merge',
      },
      {},
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      `${PYTHON_BACKEND_BASE_URL}/api/process`,
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      `${PYTHON_BACKEND_BASE_URL}/api/merge`,
    );
    expect(result.fileName).toBe('merged.h5');
  });

  it('throws a helpful error when /api/process is missing on a non-merge op', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: 'unknown route' }, { status: 404 }),
    );
    await expect(
      runProcess(
        {
          paths: ['/a.h5'],
          selectedKeys: ['x'],
          outputName: 'out.h5',
          operation: 'cut',
        },
        {},
      ),
    ).rejects.toThrow(/outdated/u);
  });

  it('rejects when the stream ends without a done event', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        {
          type: 'progress',
          phase: 'reading',
          overallDemoIndex: 0,
          overallDemoCount: 1,
          currentSourceName: 'a',
          currentDemoName: 'd',
        },
      ]),
    );
    await expect(
      runProcess(
        {
          paths: ['/a.h5'],
          selectedKeys: [],
          outputName: 'out.h5',
          operation: 'cut',
        },
        {},
      ),
    ).rejects.toThrow(/without a completion event/u);
  });

  it('propagates server-side error events', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([{ type: 'error', message: 'kaboom' }]),
    );
    await expect(
      runProcess(
        {
          paths: ['/a.h5'],
          selectedKeys: [],
          outputName: 'out.h5',
          operation: 'cut',
        },
        {},
      ),
    ).rejects.toThrow('kaboom');
  });

  it('ignores malformed SSE JSON lines without aborting the stream', async () => {
    const body =
      `data: not-json\n\n` +
      `: heartbeat comment\n\n` +
      `data: ${JSON.stringify({
        type: 'done',
        fileName: 'merged.h5',
        demoCount: 1,
        selectedKeyCount: 1,
        fileSize: 1,
      })}\n\n`;
    fetchMock.mockResolvedValueOnce(
      new Response(body, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      }),
    );

    const result = await runProcess(
      {
        paths: ['/a.h5'],
        selectedKeys: [],
        outputName: 'merged.h5',
        operation: 'merge',
      },
      {},
    );
    expect(result.fileName).toBe('merged.h5');
  });
});

describe('runLeRobotConvert (SSE streaming)', () => {
  function basicRequest(): PythonLeRobotConvertRequest {
    return {
      paths: ['/a.h5'],
      outputName: 'out',
      outputDirectory: '/chosen-output',
      outputDirectoryAuthorization: 'output-authorization',
      skipFailed: false,
      defaultTask: 'task',
      taskRules: [],
      outputVersion: 'v3.0',
      videoCodec: 'h264',
    };
  }

  it('forwards progress and resolves with the done payload', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        {
          type: 'progress',
          phase: 'reading',
          overallDemoIndex: 0,
          overallDemoCount: 1,
          currentSourceName: 'a.h5',
          currentDemoName: 'demo_0',
        },
        {
          type: 'done',
          fileName: 'out',
          demoCount: 1,
          selectedKeyCount: 1,
          fileSize: 1,
          outputPath: '/abs/out',
          outputType: 'directory',
          skippedDemoCount: 0,
          totalFrames: 10,
          taskCount: 1,
        },
      ]),
    );

    const onProgress =
      vi.fn<
        NonNullable<Parameters<typeof runLeRobotConvert>[1]['onProgress']>
      >();
    const result = await runLeRobotConvert(basicRequest(), { onProgress });

    expect(onProgress).toHaveBeenCalledTimes(1);
    expect(result.outputType).toBe('directory');
    expect(result.taskCount).toBe(1);
    const rawRequestBody = fetchMock.mock.calls[0]?.[1]?.body;
    assertString(rawRequestBody);
    const requestBody = JSON.parse(rawRequestBody) as Record<string, unknown>;
    expect(requestBody.outputVersion).toBe('v3.0');
    expect(requestBody.videoCodec).toBe('h264');
    expect(requestBody.outputDirectory).toBe('/chosen-output');
    expect(requestBody.outputDirectoryAuthorization).toBe(
      'output-authorization',
    );
  });

  it('forwards nonfatal warning events', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        { type: 'warning', message: 'NVENC failed; using CPU.' },
        {
          type: 'done',
          fileName: 'out',
          demoCount: 1,
          selectedKeyCount: 0,
          fileSize: 1,
        },
      ]),
    );
    const onWarning = vi.fn();
    await runLeRobotConvert(basicRequest(), { onWarning });
    expect(onWarning).toHaveBeenCalledWith('NVENC failed; using CPU.');
  });

  it('throws the server-supplied error message on non-ok responses', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: 'bad request' }, { status: 400 }),
    );
    await expect(runLeRobotConvert(basicRequest(), {})).rejects.toThrow(
      'bad request',
    );
  });

  it('throws when the response has no body stream', async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 200 }));
    await expect(runLeRobotConvert(basicRequest(), {})).rejects.toThrow(
      'Server did not return a readable stream.',
    );
  });

  it('rejects when the stream ends without a done event', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([]));
    await expect(runLeRobotConvert(basicRequest(), {})).rejects.toThrow(
      /without a completion event/u,
    );
  });

  it('propagates server-side error events', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([{ type: 'error', message: 'conversion failed' }]),
    );
    await expect(runLeRobotConvert(basicRequest(), {})).rejects.toThrow(
      'conversion failed',
    );
  });

  it('ignores malformed SSE JSON lines', async () => {
    const body =
      `data: this-is-not-json\n\n` +
      `data: ${JSON.stringify({
        type: 'done',
        fileName: 'out',
        demoCount: 1,
        selectedKeyCount: 1,
        fileSize: 1,
      })}\n\n`;
    fetchMock.mockResolvedValueOnce(
      new Response(body, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      }),
    );
    const result = await runLeRobotConvert(basicRequest(), {});
    expect(result.fileName).toBe('out');
  });
});

describe('pollBackendStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('invokes onStatus on every tick and stops after cancel', async () => {
    fetchMock.mockResolvedValue(
      Response.json({
        status: 'ok',
        rootDir: '/data',
        rootDirs: ['/data'],
        version: 1,
        indexReady: true,
        indexedFileCount: 0,
      }),
    );

    const onStatus = vi.fn();
    const cancel = pollBackendStatus(onStatus, 100, 200);

    await vi.waitFor(() => {
      expect(onStatus).toHaveBeenCalledTimes(1);
    });
    expect(onStatus.mock.calls[0]?.[0].available).toBe(true);

    await vi.advanceTimersByTimeAsync(200);
    expect(onStatus).toHaveBeenCalledTimes(2);

    cancel();
    await vi.advanceTimersByTimeAsync(1000);
    expect(onStatus).toHaveBeenCalledTimes(2);
  });

  it('reports available=false when the server returns an error', async () => {
    fetchMock.mockResolvedValue(new Response('', { status: 500 }));
    const onStatus = vi.fn();
    const cancel = pollBackendStatus(onStatus, 50, 5_000);

    await vi.waitFor(() => {
      expect(onStatus).toHaveBeenCalled();
    });
    expect(onStatus.mock.calls[0]?.[0].available).toBe(false);

    cancel();
  });
});
