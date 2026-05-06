import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  fetchBuffer,
  FetchError,
  NetworkError,
  safeFetch,
} from './fetch-utils';

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubFetch(impl: typeof fetch) {
  vi.stubGlobal('fetch', impl);
}

describe('safeFetch', () => {
  it('returns the response when fetch resolves with ok=true', async () => {
    const response = new Response('hi', { status: 200 });
    stubFetch(async () => response);
    const result = await safeFetch('https://x');
    expect(result).toBe(response);
  });

  it('throws NetworkError when fetch itself throws', async () => {
    stubFetch(async () => {
      throw new TypeError('disconnected');
    });
    await expect(safeFetch('https://x')).rejects.toBeInstanceOf(NetworkError);
  });

  it('throws FetchError on non-ok responses, preserving status', async () => {
    stubFetch(
      async () =>
        new Response('nope', { status: 404, statusText: 'Not Found' }),
    );
    await expect(safeFetch('https://x')).rejects.toMatchObject({
      name: 'FetchError',
      status: 404,
      statusText: 'Not Found',
    });
  });
});

describe('fetchBuffer', () => {
  it('returns the response body as an ArrayBuffer', async () => {
    const buf = new Uint8Array([1, 2, 3, 4]).buffer;
    stubFetch(async () => new Response(buf, { status: 200 }));
    const result = await fetchBuffer('https://x');
    expect(result).toBeInstanceOf(ArrayBuffer);
    expect(new Uint8Array(result)).toEqual(new Uint8Array([1, 2, 3, 4]));
  });

  it('propagates FetchError on non-ok responses', async () => {
    stubFetch(async () => new Response('', { status: 500 }));
    await expect(fetchBuffer('https://x')).rejects.toBeInstanceOf(FetchError);
  });
});
