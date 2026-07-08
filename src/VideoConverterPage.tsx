import { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import { FiDownload } from 'react-icons/fi';
import { Link, createSearchParams, useSearchParams } from 'react-router-dom';

import {
  listDemoVideos,
  loadDemoVideoFrames,
  openPoseTraceSource,
} from './pose-trace/hdf5';
import type {
  DemoVideoFrames,
  DemoVideoInfo,
  DemoVideoKey,
  PoseTraceSource,
} from './pose-trace/types';
import { type H5File, useStore } from './stores';
import styles from './VideoConverterPage.module.css';
import { resolveFileUrl } from './utils';
import {
  buildDownloadFilename,
  createFrameRenderer,
  formatDemoOption,
  getSupportedMimeType,
} from './VideoConverterPage.utils';

const PREVIEW_FPS = 30;

interface ResolvedFileState {
  file: H5File | null;
  loading: boolean;
  error: string | null;
}

interface SourceState {
  source: PoseTraceSource | null;
  loading: boolean;
  error: string | null;
}

interface CachedDownload {
  selectionKey: string;
  url: string;
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function triggerDownload(url: string, filename: string) {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
}

function useResolvedFile(fileUrl: string | null): ResolvedFileState {
  const opened = useStore((state) => state.opened);
  const openFiles = useStore((state) => state.openFiles);

  const [state, setState] = useState<ResolvedFileState>({
    file: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!fileUrl) {
      setState({ file: null, loading: false, error: null });
      return;
    }

    const openedFile = opened.find((file) => file.url === fileUrl);
    if (openedFile) {
      setState({ file: openedFile, loading: false, error: null });
      return;
    }

    let cancelled = false;
    setState({ file: null, loading: true, error: null });

    resolveFileUrl(fileUrl)
      .then((resolvedFile) => {
        if (cancelled) {
          return;
        }

        if (!resolvedFile) {
          setState({
            file: null,
            loading: false,
            error: 'This file cannot be reopened automatically. Open it again from the home page.',
          });
          return;
        }

        openFiles([resolvedFile]);
        setState({ file: resolvedFile, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setState({
          file: null,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        });
      });

    return () => {
      cancelled = true;
    };
  }, [fileUrl, openFiles, opened]);

  return state;
}

function usePoseTraceSource(file: H5File | null): SourceState {
  const [state, setState] = useState<SourceState>({
    source: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    let cleanup = () => {};

    if (!file) {
      setState({ source: null, loading: false, error: null });
      return () => {
        cleanup();
      };
    }

    setState({ source: null, loading: true, error: null });

    openPoseTraceSource(file)
      .then((source) => {
        if (cancelled) {
          source.cleanup();
          return;
        }

        cleanup = source.cleanup;
        setState({ source, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setState({
          source: null,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        });
      });

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [file]);

  return state;
}

async function encodeVideoToBlob(
  video: DemoVideoFrames,
  mimeType: string,
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = video.width;
  canvas.height = video.height;

  const context = canvas.getContext('2d', { alpha: false });
  if (!context) {
    throw new Error('Could not create a canvas context for video encoding.');
  }

  const renderFrame = createFrameRenderer(context, video.width, video.height, video.channels);
  const stream = canvas.captureStream(PREVIEW_FPS);
  const recorder = new MediaRecorder(stream, { mimeType });
  const chunks: BlobPart[] = [];

  const stopPromise = new Promise<Blob>((resolve, reject) => {
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };
    recorder.onerror = () => {
      reject(new Error('Video encoding failed.'));
    };
    recorder.onstop = () => {
      resolve(new Blob(chunks, { type: mimeType }));
    };
  });

  recorder.start();

  const [track] = stream.getVideoTracks();
  const requestFrame = typeof (track as CanvasCaptureMediaStreamTrack).requestFrame === 'function'
    ? () => (track as CanvasCaptureMediaStreamTrack).requestFrame()
    : null;
  const frameDuration = 1000 / PREVIEW_FPS;

  for (let frameIndex = 0; frameIndex < video.frameCount; frameIndex += 1) {
    renderFrame(video.frames, frameIndex);
    requestFrame?.();
    await wait(frameDuration);
  }

  await wait(frameDuration);
  recorder.stop();

  const blob = await stopPromise;
  stream.getTracks().forEach((mediaTrack) => mediaTrack.stop());
  return blob;
}

function EmptyState({
  openedFileCount,
}: {
  openedFileCount: number;
}) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Video Converter</h2>
      <p className={styles.emptyText}>
        Open an HDF5 file in rebelHDF5, then switch to this page to preview and save demo videos.
      </p>
      <div className={styles.emptyActions}>
        <Link className={styles.openBtn} to="/">
          Open HDF5
        </Link>
        {openedFileCount > 0 && (
          <span>{openedFileCount} opened file{openedFileCount === 1 ? '' : 's'} available in the sidebar.</span>
        )}
      </div>
    </div>
  );
}

function VideoConverterPage() {
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const opened = useStore((state) => state.opened);

  const { file, loading: fileLoading, error: fileError } = useResolvedFile(fileUrl);
  const {
    source,
    loading: sourceLoading,
    error: sourceError,
  } = usePoseTraceSource(file);

  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [videoOptions, setVideoOptions] = useState<DemoVideoInfo[]>([]);
  const [videoOptionsLoading, setVideoOptionsLoading] = useState(false);
  const [videoOptionsError, setVideoOptionsError] = useState<string | null>(null);
  const [selectedVideoKey, setSelectedVideoKey] = useState<DemoVideoKey | null>(null);
  const [videoData, setVideoData] = useState<DemoVideoFrames | null>(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cachedDownloadRef = useRef<CachedDownload | null>(null);
  const supportedMimeType = useMemo(() => getSupportedMimeType(), []);

  const demos = source?.demos ?? [];

  useEffect(() => {
    if (demos.length === 0) {
      setSelectedDemo(null);
      return;
    }

    setSelectedDemo((current) =>
      current && demos.some((demo) => demo.name === current) ? current : demos[0].name,
    );
  }, [demos]);

  useEffect(() => {
    if (!source || !selectedDemo) {
      setVideoOptions([]);
      setSelectedVideoKey(null);
      setVideoData(null);
      setVideoError(null);
      setVideoOptionsLoading(false);
      setVideoOptionsError(null);
      return;
    }

    let cancelled = false;
    setVideoOptions([]);
    setSelectedVideoKey(null);
    setVideoData(null);
    setVideoError(null);
    setVideoOptionsLoading(true);
    setVideoOptionsError(null);

    listDemoVideos(source, selectedDemo)
      .then((nextVideos) => {
        if (cancelled) {
          return;
        }

        setVideoOptions(nextVideos);
        setSelectedVideoKey((current) => {
          if (current && nextVideos.some((video) => video.key === current)) {
            return current;
          }
          // Prefer a third-person/overview camera when available, otherwise the first one.
          const preferred = nextVideos.find((video) =>
            /(?:^|_)(?:top|overview|third_person|external|front|scene)(?:_|$)/iu.test(video.key),
          );
          return preferred?.key ?? nextVideos[0]?.key ?? null;
        });
        setVideoOptionsLoading(false);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setVideoOptions([]);
        setSelectedVideoKey(null);
        setVideoOptionsLoading(false);
        setVideoOptionsError(error instanceof Error ? error.message : String(error));
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDemo, source]);

  useEffect(() => {
    if (!source || !selectedDemo || !selectedVideoKey) {
      setVideoData(null);
      setVideoLoading(false);
      setVideoError(null);
      return;
    }

    let cancelled = false;
    setVideoData(null);
    setVideoLoading(true);
    setVideoError(null);
    setSaveError(null);

    loadDemoVideoFrames(source, selectedDemo, selectedVideoKey)
      .then((nextVideo) => {
        if (cancelled) {
          return;
        }

        setVideoData(nextVideo);
        setVideoLoading(false);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setVideoData(null);
        setVideoLoading(false);
        setVideoError(error instanceof Error ? error.message : String(error));
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDemo, selectedVideoKey, source]);

  const selectedVideoInfo = useMemo(
    () => videoOptions.find((video) => video.key === selectedVideoKey) ?? null,
    [selectedVideoKey, videoOptions],
  );

  // The schema discovers cameras dynamically, so we don't enforce a specific set.

  const selectionText = useMemo(() => {
    if (!selectedDemo) {
      return null;
    }

    const demo = demos.find((entry) => entry.name === selectedDemo);
    return {
      demo: selectedDemo,
      samples: demo?.num_samples ?? 'n/a',
      source: demo?.source_episode_index ?? 'n/a',
      success: demo?.success != null ? Number(demo.success) : 'n/a',
    };
  }, [demos, selectedDemo]);

  const selectionKey = `${file?.url ?? fileUrl ?? 'file'}:${selectedDemo ?? 'demo'}:${selectedVideoKey ?? 'video'}`;
  const currentFrameLabel = videoData ? currentFrameIndex + 1 : 0;
  const currentTimeSeconds = videoData ? currentFrameIndex / PREVIEW_FPS : 0;
  const totalTimeSeconds = videoData ? videoData.frameCount / PREVIEW_FPS : 0;

  useEffect(() => {
    if (!cachedDownloadRef.current || cachedDownloadRef.current.selectionKey === selectionKey) {
      return;
    }

    URL.revokeObjectURL(cachedDownloadRef.current.url);
    cachedDownloadRef.current = null;
  }, [selectionKey]);

  useEffect(() => () => {
    if (cachedDownloadRef.current) {
      URL.revokeObjectURL(cachedDownloadRef.current.url);
      cachedDownloadRef.current = null;
    }
  }, []);

  useEffect(() => {
    setCurrentFrameIndex(0);
    setIsPlaying(Boolean(videoData && videoData.frameCount > 1));
  }, [videoData]);

  useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !videoData) {
      return;
    }

    canvas.width = videoData.width;
    canvas.height = videoData.height;

    const context = canvas.getContext('2d', { alpha: false });
    if (!context) {
      return;
    }

    const renderFrame = createFrameRenderer(
      context,
      videoData.width,
      videoData.height,
      videoData.channels,
    );
    renderFrame(videoData.frames, currentFrameIndex);
  }, [currentFrameIndex, videoData]);

  useEffect(() => {
    if (!videoData || !isPlaying || videoData.frameCount <= 1) {
      return;
    }

    let animationFrameId = 0;
    let lastTimestamp = performance.now();
    const frameDuration = 1000 / PREVIEW_FPS;

    const tick = (timestamp: number) => {
      const elapsed = timestamp - lastTimestamp;

      if (elapsed >= frameDuration) {
        const advance = Math.floor(elapsed / frameDuration);
        lastTimestamp += advance * frameDuration;
        setCurrentFrameIndex((current) => (current + advance) % videoData.frameCount);
      }

      animationFrameId = window.requestAnimationFrame(tick);
    };

    animationFrameId = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying, videoData]);

  async function handleSaveVideo() {
    if (!videoData || !selectedDemo || !source) {
      return;
    }

    if (!supportedMimeType) {
      setSaveError('This browser cannot save the preview as a WebM video.');
      return;
    }

    setSaveError(null);

    const filename = buildDownloadFilename(source.datasetName, selectedDemo, videoData.key);
    if (cachedDownloadRef.current?.selectionKey === selectionKey) {
      triggerDownload(cachedDownloadRef.current.url, filename);
      return;
    }

    setIsSaving(true);
    try {
      const blob = await encodeVideoToBlob(videoData, supportedMimeType);
      const url = URL.createObjectURL(blob);

      if (cachedDownloadRef.current) {
        URL.revokeObjectURL(cachedDownloadRef.current.url);
      }

      cachedDownloadRef.current = {
        selectionKey,
        url,
      };
      triggerDownload(url, filename);
    } catch (error: unknown) {
      setSaveError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Analysis</p>
          <h1 className={styles.title}>Video Converter</h1>
          <p className={styles.subtitle}>
            Preview RGB demo videos stored in the current HDF5 file and save them to disk only when needed.
          </p>
        </div>
      </header>

      {!fileUrl && !file && !fileLoading && <EmptyState openedFileCount={opened.length} />}

      {fileError && (
        <section className={styles.messageCard}>
          <p className={styles.errorText}>{fileError}</p>
        </section>
      )}

      {(fileLoading || sourceLoading) && (
        <section className={styles.messageCard}>
          <p>Loading video-converter data…</p>
        </section>
      )}

      {sourceError && (
        <section className={styles.messageCard}>
          <p className={styles.errorText}>{sourceError}</p>
        </section>
      )}

      {source && (
        <>
          <section className={styles.controlsCard}>
            <div className={styles.controlGrid}>
              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="video-demo-select">
                  Demo
                </label>
                <select
                  id="video-demo-select"
                  className={styles.select}
                  value={selectedDemo ?? ''}
                  onChange={(event) => {
                    const nextDemo = event.target.value;
                    startTransition(() => {
                      setSelectedDemo(nextDemo);
                    });
                  }}
                  disabled={demos.length === 0}
                >
                  {demos.length === 0 && <option value="">No demos available</option>}
                  {demos.map((demo) => (
                    <option key={demo.name} value={demo.name}>
                      {formatDemoOption(demo)}
                    </option>
                  ))}
                </select>
              </div>

              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="video-stream-select">
                  Video
                </label>
                <select
                  id="video-stream-select"
                  className={styles.select}
                  value={selectedVideoKey ?? ''}
                  onChange={(event) => {
                    const nextVideo = event.target.value as DemoVideoKey;
                    startTransition(() => {
                      setSelectedVideoKey(nextVideo || null);
                    });
                  }}
                  disabled={videoOptionsLoading || videoOptions.length === 0}
                >
                  {videoOptions.length === 0 && (
                    <option value="">
                      {videoOptionsLoading ? 'Loading videos…' : 'No supported videos found'}
                    </option>
                  )}
                  {videoOptions.map((video) => (
                    <option key={video.key} value={video.key}>
                      {video.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className={styles.statusRow}>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Demos:</span> {demos.length}
              </div>
              {selectionText && (
                <>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Selected:</span> {selectionText.demo}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Samples:</span> {selectionText.samples}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Source:</span> {selectionText.source}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Success:</span> {selectionText.success}
                  </div>
                </>
              )}
              {selectedVideoInfo && (
                <>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Frames:</span> {selectedVideoInfo.frameCount}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Resolution:</span> {selectedVideoInfo.width}×{selectedVideoInfo.height}
                  </div>
                </>
              )}
            </div>

          </section>

          {videoOptionsError && (
            <section className={styles.messageCard}>
              <p className={styles.errorText}>{videoOptionsError}</p>
            </section>
          )}

          <section className={styles.previewCard}>
            <div className={styles.previewHeader}>
              <div>
                <h2 className={styles.previewTitle}>
                  {selectedVideoInfo ? `${selectedVideoInfo.label} Preview` : 'Video Preview'}
                </h2>
                <p className={styles.previewMeta}>
                  {selectedVideoInfo
                    ? `${selectedVideoInfo.path} · ${PREVIEW_FPS} FPS preview`
                    : 'Choose a demo video to preview it.'}
                </p>
              </div>
              <button
                type="button"
                className={styles.saveBtn}
                onClick={() => {
                  void handleSaveVideo();
                }}
                disabled={!videoData || isSaving || !supportedMimeType}
              >
                <FiDownload aria-hidden />
                <span>{isSaving ? 'Saving…' : 'Save Video'}</span>
              </button>
            </div>

            {!supportedMimeType && (
              <p className={styles.hintText}>
                Saving is unavailable in this browser because WebM recording is not supported.
              </p>
            )}

            {saveError && <p className={styles.errorText}>{saveError}</p>}

            {videoOptions.length === 0 && !videoOptionsLoading && !videoOptionsError ? (
              <p className={styles.infoText}>
                No supported video datasets were found. Expected one or more (T, H, W, C) datasets under `obs/cameras/` or directly under `obs/`.
              </p>
            ) : videoLoading ? (
              <p className={styles.infoText}>Loading video frames…</p>
            ) : videoError ? (
              <p className={styles.errorText}>{videoError}</p>
            ) : videoData ? (
              <div className={styles.previewSurface}>
                <div className={styles.previewViewport}>
                  <canvas
                    ref={previewCanvasRef}
                    className={styles.previewCanvas}
                  />
                </div>
                <div className={styles.playerControls}>
                  <div className={styles.sliderBlock}>
                    <input
                      className={styles.timelineSlider}
                      type="range"
                      min={0}
                      max={Math.max(videoData.frameCount - 1, 0)}
                      step={1}
                      value={currentFrameIndex}
                      onChange={(event) => {
                        setCurrentFrameIndex(Number(event.target.value));
                      }}
                      disabled={videoData.frameCount <= 1}
                    />
                    <div className={styles.sliderMeta}>
                      <span>Frame {currentFrameLabel}/{videoData.frameCount}</span>
                      <span>{currentTimeSeconds.toFixed(2)}s / {totalTimeSeconds.toFixed(2)}s</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className={styles.playBtn}
                    onClick={() => {
                      setIsPlaying((current) => !current);
                    }}
                    disabled={videoData.frameCount <= 1}
                  >
                    {isPlaying ? 'Pause' : 'Play'}
                  </button>
                </div>
              </div>
            ) : (
              <p className={styles.infoText}>Select a video to preview it.</p>
            )}
          </section>
        </>
      )}

      {!source && !fileLoading && !sourceLoading && !fileError && !sourceError && fileUrl && (
        <section className={styles.messageCard}>
          <p>
            Select an opened file from the sidebar to inspect demo videos, or go back to the viewer.
          </p>
          <div className={styles.emptyActions}>
            <Link
              className={styles.openBtn}
              to={`/video-converter?${createSearchParams({ url: fileUrl }).toString()}`}
            >
              Retry
            </Link>
            <Link className={styles.openBtn} to="/">
              Open HDF5
            </Link>
          </div>
        </section>
      )}
    </div>
  );
}

export default VideoConverterPage;
