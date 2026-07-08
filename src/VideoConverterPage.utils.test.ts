import { afterEach, describe, expect, it, vi } from 'vitest';

import type { DemoInfo } from './pose-trace/types';
import {
  buildDownloadFilename,
  createFrameRenderer,
  formatDemoOption,
  getSupportedMimeType,
  sanitizeFilenamePart,
} from './VideoConverterPage.utils';

function demo(overrides: Partial<DemoInfo>): DemoInfo {
  return {
    name: 'demo_0',
    episode_index: null,
    num_samples: null,
    success: null,
    source_episode_index: null,
    ...overrides,
  };
}

describe('formatDemoOption', () => {
  it('shows only the name when no metadata is present', () => {
    expect(formatDemoOption(demo({ name: 'demo_3' }))).toBe('demo_3');
  });

  it('appends samples, success and source when present', () => {
    expect(
      formatDemoOption(
        demo({
          name: 'demo_1',
          num_samples: 42,
          success: true,
          source_episode_index: 7,
        }),
      ),
    ).toBe('demo_1 | samples=42 | success=1 | source=7');
  });

  it('renders success=0 for a failed demo and includes samples=0', () => {
    expect(
      formatDemoOption(demo({ name: 'd', num_samples: 0, success: false })),
    ).toBe('d | samples=0 | success=0');
  });
});

describe('sanitizeFilenamePart', () => {
  it('replaces unsafe characters with underscores', () => {
    expect(sanitizeFilenamePart('a b/c:d*e')).toBe('a_b_c_d_e');
  });

  it('keeps alphanumerics, dot, underscore and dash', () => {
    expect(sanitizeFilenamePart('File-1.2_v3')).toBe('File-1.2_v3');
  });
});

describe('buildDownloadFilename', () => {
  it('joins sanitized dataset/demo names with the video key and .webm', () => {
    expect(buildDownloadFilename('data set/1', 'demo 2', 'cam:front')).toBe(
      'data_set_1-demo_2-cam:front.webm',
    );
  });
});

describe('getSupportedMimeType', () => {
  const original = globalThis.MediaRecorder;

  afterEach(() => {
    globalThis.MediaRecorder = original;
  });

  it('returns null when MediaRecorder is unavailable', () => {
    // @ts-expect-error deliberately removing the global for the test
    delete globalThis.MediaRecorder;
    expect(getSupportedMimeType()).toBeNull();
  });

  it('returns generic webm when isTypeSupported is missing', () => {
    globalThis.MediaRecorder = {} as unknown as typeof MediaRecorder;
    expect(getSupportedMimeType()).toBe('video/webm');
  });

  it('returns the first supported candidate (vp9 preferred)', () => {
    globalThis.MediaRecorder = {
      isTypeSupported: (type: string) => type === 'video/webm;codecs=vp9',
    } as unknown as typeof MediaRecorder;
    expect(getSupportedMimeType()).toBe('video/webm;codecs=vp9');
  });

  it('falls back to vp8 when vp9 is unsupported', () => {
    globalThis.MediaRecorder = {
      isTypeSupported: (type: string) => type === 'video/webm;codecs=vp8',
    } as unknown as typeof MediaRecorder;
    expect(getSupportedMimeType()).toBe('video/webm;codecs=vp8');
  });

  it('returns null when nothing is supported', () => {
    globalThis.MediaRecorder = {
      isTypeSupported: () => false,
    } as unknown as typeof MediaRecorder;
    expect(getSupportedMimeType()).toBeNull();
  });
});

describe('createFrameRenderer', () => {
  function fakeContext(): {
    context: CanvasRenderingContext2D;
    lastImageData: () => ImageData | null;
  } {
    let captured: ImageData | null = null;
    const context = {
      putImageData: (imageData: ImageData) => {
        // Copy the pixels out, since the renderer reuses one backing buffer.
        captured = new ImageData(
          new Uint8ClampedArray(imageData.data),
          imageData.width,
          imageData.height,
        );
      },
    } as unknown as CanvasRenderingContext2D;
    return { context, lastImageData: () => captured };
  }

  it('copies RGBA source data directly', () => {
    const { context, lastImageData } = fakeContext();
    const render = createFrameRenderer(context, 1, 1, 4);
    const frames = new Uint8Array([10, 20, 30, 40]);
    render(frames, 0);
    expect(Array.from(lastImageData()!.data)).toEqual([10, 20, 30, 40]);
  });

  it('expands RGB source data with an opaque alpha channel', () => {
    const { context, lastImageData } = fakeContext();
    const render = createFrameRenderer(context, 1, 1, 3);
    render(new Uint8Array([10, 20, 30]), 0);
    expect(Array.from(lastImageData()!.data)).toEqual([10, 20, 30, 255]);
  });

  it('replicates a grayscale value across RGB with opaque alpha', () => {
    const { context, lastImageData } = fakeContext();
    const render = createFrameRenderer(context, 1, 1, 1);
    render(new Uint8Array([128]), 0);
    expect(Array.from(lastImageData()!.data)).toEqual([128, 128, 128, 255]);
  });

  it('reads the correct frame using frameIndex and channel stride', () => {
    const { context, lastImageData } = fakeContext();
    const render = createFrameRenderer(context, 1, 1, 3);
    // Two frames of one RGB pixel each; frame 1 is [40,50,60].
    const frames = new Uint8Array([10, 20, 30, 40, 50, 60]);
    render(frames, 1);
    expect(Array.from(lastImageData()!.data)).toEqual([40, 50, 60, 255]);
  });

  it('renders every pixel of a multi-pixel grayscale frame', () => {
    const { context, lastImageData } = fakeContext();
    const render = createFrameRenderer(context, 2, 1, 1);
    render(new Uint8Array([100, 200]), 0);
    expect(Array.from(lastImageData()!.data)).toEqual([
      100, 100, 100, 255, 200, 200, 200, 255,
    ]);
  });
});
