import { act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  FileService,
  type LocalFile,
  type RemoteFile,
  useStore,
} from './stores';

const initialState = useStore.getState();

beforeEach(() => {
  globalThis.localStorage.clear();
});

afterEach(() => {
  // Reset store between tests so they don't share state.
  act(() => {
    useStore.setState(initialState, true);
  });
});

function makeRemoteFile(url: string, name = url): RemoteFile {
  return {
    url,
    name,
    service: FileService.Url,
    resolvedUrl: url,
  };
}

function makeLocalFile(name = 'local.h5'): LocalFile {
  const file = new File([], name);
  return {
    url: `blob:fake-${name}`,
    name,
    service: FileService.Local,
    resolvedUrl: `blob:fake-${name}`,
    file,
  };
}

describe('useStore', () => {
  it('starts with no opened files and an expanded sidebar', () => {
    expect(useStore.getState().opened).toEqual([]);
    expect(useStore.getState().sidebarMayCollapse).toBe(false);
  });

  it('appends files via openFiles', () => {
    const a = makeRemoteFile('https://x/a.h5');
    const b = makeRemoteFile('https://x/b.h5');

    act(() => {
      useStore.getState().openFiles([a]);
      useStore.getState().openFiles([b]);
    });

    expect(useStore.getState().opened).toEqual([a, b]);
  });

  it('removes a file at the given index', () => {
    const a = makeRemoteFile('https://x/a.h5');
    const b = makeRemoteFile('https://x/b.h5');
    const c = makeRemoteFile('https://x/c.h5');

    act(() => {
      useStore.getState().openFiles([a, b, c]);
      useStore.getState().removeFileAt(1);
    });

    expect(useStore.getState().opened).toEqual([a, c]);
  });

  it('toggles the sidebar', () => {
    act(() => useStore.getState().toggleSidebar());
    expect(useStore.getState().sidebarMayCollapse).toBe(true);
    act(() => useStore.getState().toggleSidebar());
    expect(useStore.getState().sidebarMayCollapse).toBe(false);
  });
});

describe('persistence', () => {
  it('writes remote files to localStorage but excludes local files', async () => {
    const remote = makeRemoteFile('https://x/remote.h5');
    const local = makeLocalFile('only-local.h5');

    act(() => {
      useStore.getState().openFiles([remote, local]);
    });

    // Wait one microtask tick so the persist middleware flushes.
    await Promise.resolve();

    const raw = globalThis.localStorage.getItem('rebel-hdf5-viewer');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw ?? '{}') as {
      state: { opened: { url: string }[] };
    };
    const persistedUrls = parsed.state.opened.map((f) => f.url);
    expect(persistedUrls).toContain('https://x/remote.h5');
    expect(persistedUrls).not.toContain(local.url);
  });

  it('persists the sidebarMayCollapse flag', async () => {
    act(() => useStore.getState().toggleSidebar());
    await Promise.resolve();

    const raw = globalThis.localStorage.getItem('rebel-hdf5-viewer');
    const parsed = JSON.parse(raw ?? '{}') as {
      state: { sidebarMayCollapse: boolean };
    };
    expect(parsed.state.sidebarMayCollapse).toBe(true);
  });
});
