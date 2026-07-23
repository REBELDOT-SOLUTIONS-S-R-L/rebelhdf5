import '@testing-library/jest-dom/vitest';

import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

import ImageDataPolyfill from './src/test-utils/ImageDataPolyfill';
import ResizeObserverPolyfill from './src/test-utils/ResizeObserverPolyfill';

const store: Record<string, string> = {};

const localStorageMock: Storage = {
  getItem: vi.fn((key: string) => store[key] ?? null),
  setItem: vi.fn((key: string, value: string) => {
    store[key] = value;
  }),
  removeItem: vi.fn((key: string) => {
    Reflect.deleteProperty(store, key);
  }),
  clear: vi.fn(() => {
    for (const key of Object.keys(store)) {
      Reflect.deleteProperty(store, key);
    }
  }),
  key: vi.fn((index: number) => Object.keys(store)[index] ?? null),
  get length() {
    return Object.keys(store).length;
  },
};

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true,
});

// jsdom does not implement the canvas `ImageData` global that browsers provide.
// Supply a minimal, spec-compatible shim so canvas helpers can be unit-tested.
if (!Reflect.has(globalThis, 'ImageData')) {
  Object.defineProperty(globalThis, 'ImageData', {
    value: ImageDataPolyfill,
    writable: true,
    configurable: true,
  });
}

// jsdom implements neither matchMedia nor ResizeObserver; several pages/charts
// depend on them. Provide inert shims so components can mount under test.
if (typeof globalThis.matchMedia !== 'function') {
  Object.defineProperty(globalThis, 'matchMedia', {
    value: (query: string): MediaQueryList =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList,
    writable: true,
    configurable: true,
  });
}

if (!Reflect.has(globalThis, 'ResizeObserver')) {
  Object.defineProperty(globalThis, 'ResizeObserver', {
    value: ResizeObserverPolyfill,
    writable: true,
    configurable: true,
  });
}

afterEach(() => {
  cleanup();
});
