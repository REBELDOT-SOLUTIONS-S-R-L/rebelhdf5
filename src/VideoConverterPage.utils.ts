import { type DemoInfo, type DemoVideoKey } from './pose-trace/types';

/** Build a human-readable label for a demo option in the dropdown. */
export function formatDemoOption(demo: DemoInfo): string {
  const parts = [demo.name];
  if (demo.num_samples !== null) {
    parts.push(`samples=${demo.num_samples}`);
  }
  if (demo.success !== null) {
    parts.push(`success=${demo.success ? 1 : 0}`);
  }
  if (demo.source_episode_index !== null) {
    parts.push(`source=${demo.source_episode_index}`);
  }
  return parts.join(' | ');
}

/** Replace any character unsafe for a filename with an underscore. */
export function sanitizeFilenamePart(value: string): string {
  return value.replaceAll(/[^\w\-.]/gu, '_');
}

/** Build the `.webm` download filename for a converted demo video. */
export function buildDownloadFilename(
  datasetName: string,
  demoName: string,
  videoKey: DemoVideoKey,
): string {
  return `${sanitizeFilenamePart(datasetName)}-${sanitizeFilenamePart(demoName)}-${videoKey}.webm`;
}

/**
 * Return the best WebM MIME type the current MediaRecorder supports, or `null`
 * when recording is not available at all.
 */
export function getSupportedMimeType(): string | null {
  if (typeof MediaRecorder === 'undefined') {
    return null;
  }

  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];

  if (typeof MediaRecorder.isTypeSupported !== 'function') {
    return candidates[candidates.length - 1];
  }

  return (
    candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ??
    null
  );
}

/**
 * Build a reusable frame renderer that writes a single frame of interleaved
 * channel data into the given 2D context. Supports RGBA (4), RGB (3, opaque
 * alpha) and grayscale (1, value replicated across RGB) source layouts.
 */
export function createFrameRenderer(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  channels: number,
): (frames: Uint8Array, frameIndex: number) => void {
  const pixelCount = width * height;
  const bytesPerFrame = pixelCount * channels;
  const rgbaBuffer = new Uint8ClampedArray(pixelCount * 4);
  const imageData = new ImageData(rgbaBuffer, width, height);

  return (frames, frameIndex) => {
    const frameOffset = frameIndex * bytesPerFrame;

    if (channels === 4) {
      rgbaBuffer.set(
        frames.subarray(frameOffset, frameOffset + pixelCount * 4),
      );
    } else if (channels === 3) {
      for (
        let sourceIndex = frameOffset, targetIndex = 0;
        targetIndex < rgbaBuffer.length;
        targetIndex += 4, sourceIndex += 3
      ) {
        rgbaBuffer[targetIndex] = frames[sourceIndex];
        rgbaBuffer[targetIndex + 1] = frames[sourceIndex + 1];
        rgbaBuffer[targetIndex + 2] = frames[sourceIndex + 2];
        rgbaBuffer[targetIndex + 3] = 255;
      }
    } else {
      for (
        let sourceIndex = frameOffset, targetIndex = 0;
        targetIndex < rgbaBuffer.length;
        targetIndex += 4, sourceIndex += 1
      ) {
        const value = frames[sourceIndex];
        rgbaBuffer[targetIndex] = value;
        rgbaBuffer[targetIndex + 1] = value;
        rgbaBuffer[targetIndex + 2] = value;
        rgbaBuffer[targetIndex + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
  };
}
