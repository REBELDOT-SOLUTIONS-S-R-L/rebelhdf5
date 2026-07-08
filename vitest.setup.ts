import '@testing-library/jest-dom/vitest';

import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

const store: Record<string, string> = {};

const localStorageMock: Storage = {
  getItem: vi.fn((key: string) => store[key] ?? null),
  setItem: vi.fn((key: string, value: string) => {
    store[key] = value;
  }),
  removeItem: vi.fn((key: string) => {
    delete store[key];
  }),
  clear: vi.fn(() => {
    Object.keys(store).forEach((key) => delete store[key]);
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
if (globalThis.ImageData === undefined) {
  class ImageDataPolyfill {
    public readonly data: Uint8ClampedArray;
    public readonly width: number;
    public readonly height: number;

    public constructor(
      dataOrWidth: Uint8ClampedArray | number,
      widthOrHeight: number,
      height?: number,
    ) {
      if (typeof dataOrWidth === 'number') {
        this.width = dataOrWidth;
        this.height = widthOrHeight;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      } else {
        this.data = dataOrWidth;
        this.width = widthOrHeight;
        this.height = height ?? dataOrWidth.length / 4 / widthOrHeight;
      }
    }
  }

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

if (typeof globalThis.ResizeObserver === 'undefined') {
  class ResizeObserverPolyfill {
    public observe(): void {}
    public unobserve(): void {}
    public disconnect(): void {}
  }
  Object.defineProperty(globalThis, 'ResizeObserver', {
    value: ResizeObserverPolyfill,
    writable: true,
    configurable: true,
  });
}

afterEach(() => {
  cleanup();
});
