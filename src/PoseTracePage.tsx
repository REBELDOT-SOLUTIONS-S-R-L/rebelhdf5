import { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import { FiActivity } from 'react-icons/fi';
import { Link, createSearchParams, useSearchParams } from 'react-router-dom';

import { openPoseTraceSource, loadDemoRows } from './pose-trace/hdf5';
import Plot from './pose-trace/PlotlyChart';
import {
  build2DData,
  build2DLayout,
  build3DDataForStep,
  build3DLayout,
  buildEmptyLayout,
  get2DTraceGroups,
  getDefaultHidden3DTraceGroups,
  type PlotSceneCamera,
} from './pose-trace/plotConfig';
import type { DemoInfo, DemoRow, PoseTraceSource } from './pose-trace/types';
import { type H5File, useStore } from './stores';
import styles from './PoseTracePage.module.css';
import { resolveFileUrl } from './utils';

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

interface PlotLegendTraceState {
  legendgroup?: string;
}

interface PlotLegendClickEvent {
  curveNumber?: number;
  data?: PlotLegendTraceState[];
}

function usePrefersDarkMode(): boolean {
  const [prefersDarkMode, setPrefersDarkMode] = useState(() => {
    if (typeof window === 'undefined') {
      return false;
    }

    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const updatePreference = (event: MediaQueryListEvent) => {
      setPrefersDarkMode(event.matches);
    };

    setPrefersDarkMode(mediaQuery.matches);
    mediaQuery.addEventListener('change', updatePreference);
    return () => {
      mediaQuery.removeEventListener('change', updatePreference);
    };
  }, []);

  return prefersDarkMode;
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

function formatDemoOption(demo: DemoInfo): string {
  const parts = [demo.name];
  if (demo.num_samples != null) parts.push(`samples=${demo.num_samples}`);
  if (demo.success != null) parts.push(`success=${demo.success ? 1 : 0}`);
  if (demo.source_episode_index != null) parts.push(`source=${demo.source_episode_index}`);
  return parts.join(' | ');
}

function clonePlotSceneCamera(camera: PlotSceneCamera | null | undefined): PlotSceneCamera | null {
  if (!camera) {
    return null;
  }

  return {
    center: camera.center ? { ...camera.center } : undefined,
    eye: camera.eye ? { ...camera.eye } : undefined,
    up: camera.up ? { ...camera.up } : undefined,
    projection: camera.projection ? { ...camera.projection } : undefined,
  };
}

function PoseTraceCharts({
  rows,
  loading,
  themeKey,
}: {
  rows: DemoRow[];
  loading: boolean;
  themeKey: string;
}) {
  const hasData = rows.length > 0;
  const [selectedStepIndex, setSelectedStepIndex] = useState(0);
  const [hiddenTraceGroupsVersion, setHiddenTraceGroupsVersion] = useState(0);
  const sceneCameraRef = useRef<PlotSceneCamera | null>(null);
  const sceneCameraIdentityRef = useRef<string | null>(null);
  const hiddenTraceGroupsRef = useRef<Set<string>>(new Set());
  const emptyMessage = loading
    ? 'Loading pose-trace data…'
    : 'Select a demo to inspect its end-effector and garment trajectories.';
  const maxStepIndex = Math.max(rows.length - 1, 0);
  const currentStepIndex = Math.min(selectedStepIndex, maxStepIndex);
  const currentStepRow = rows[currentStepIndex];
  const currentStepLabel = currentStepRow?.episode_step ?? currentStepIndex;
  const threeDimensionalChartKey = `${rows[0]?.dataset_name ?? 'dataset'}-${rows[0]?.demo_name ?? 'demo'}-3d-${themeKey}`;
  const twoDimensionalGroups = useMemo(() => get2DTraceGroups(rows), [rows]);
  const defaultHiddenTraceGroups = useMemo(() => getDefaultHidden3DTraceGroups(rows), [rows]);

  if (sceneCameraIdentityRef.current !== threeDimensionalChartKey) {
    sceneCameraIdentityRef.current = threeDimensionalChartKey;
    sceneCameraRef.current = null;
    hiddenTraceGroupsRef.current = defaultHiddenTraceGroups;
  }

  const threeDimensionalData = useMemo(
    () => (hasData ? build3DDataForStep(rows, currentStepIndex, hiddenTraceGroupsRef.current) : []),
    [currentStepIndex, hasData, hiddenTraceGroupsVersion, rows, threeDimensionalChartKey],
  );

  useEffect(() => {
    setSelectedStepIndex(maxStepIndex);
  }, [maxStepIndex, rows]);

  return (
    <div className={styles.chartStack}>
      <section className={styles.chartCard}>
        <Plot
          key={threeDimensionalChartKey}
          data={threeDimensionalData}
          layout={
            hasData
              ? build3DLayout(rows, sceneCameraRef.current)
              : buildEmptyLayout('3D Pose Trace', emptyMessage, true)
          }
          onUpdate={(figure) => {
            const nextCamera = clonePlotSceneCamera(figure.layout?.scene?.camera);
            if (nextCamera) {
              sceneCameraRef.current = nextCamera;
            }
          }}
          onLegendClick={(event) => {
            const clickEvent = event as PlotLegendClickEvent;
            const clickedTrace = clickEvent.data?.[clickEvent.curveNumber ?? -1];
            if (typeof clickedTrace?.legendgroup !== 'string') {
              return false;
            }

            const nextHiddenTraceGroups = new Set(hiddenTraceGroupsRef.current);
            if (nextHiddenTraceGroups.has(clickedTrace.legendgroup)) {
              nextHiddenTraceGroups.delete(clickedTrace.legendgroup);
            } else {
              nextHiddenTraceGroups.add(clickedTrace.legendgroup);
            }

            hiddenTraceGroupsRef.current = nextHiddenTraceGroups;
            startTransition(() => {
              setHiddenTraceGroupsVersion((version) => version + 1);
            });

            return false;
          }}
          onLegendDoubleClick={() => {
            return false;
          }}
          useResizeHandler
          style={{ width: '100%' }}
          config={{ responsive: true }}
        />
        {hasData && (
          <div className={styles.stepSliderBlock}>
            <div className={styles.stepSliderHeader}>
              <label className={styles.stepSliderLabel} htmlFor="pose-trace-step-slider">
                3D step
              </label>
              <span className={styles.stepSliderValue}>
                {currentStepLabel} ({currentStepIndex + 1}/{rows.length})
              </span>
            </div>
            <input
              id="pose-trace-step-slider"
              className={styles.stepSlider}
              type="range"
              min={0}
              max={maxStepIndex}
              step={1}
              value={currentStepIndex}
              onChange={(event) => {
                const nextStepIndex = Number(event.target.value);
                startTransition(() => {
                  setSelectedStepIndex(nextStepIndex);
                });
              }}
              disabled={rows.length <= 1}
            />
          </div>
        )}
      </section>

      <section className={styles.splitCharts}>
        {twoDimensionalGroups.map((group) => (
          <div key={group.id} className={styles.chartCard}>
            <Plot
              key={`${rows[0]?.dataset_name ?? 'dataset'}-${rows[0]?.demo_name ?? 'demo'}-${group.id}-${themeKey}`}
              data={hasData ? build2DData(rows, group) : []}
              layout={
                hasData
                  ? build2DLayout(rows, group)
                  : buildEmptyLayout(
                      `${group.label} 2D Pose Trace`,
                      emptyMessage,
                    )
              }
              useResizeHandler
              style={{ width: '100%' }}
              config={{ responsive: true }}
            />
          </div>
        ))}
      </section>
    </div>
  );
}

function EmptyState({
  openedFileCount,
}: {
  openedFileCount: number;
}) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Pose Trace</h2>
      <p className={styles.emptyText}>
        Open an HDF5 file in myHDF5, then switch to this page to plot the pose-trace data.
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

function PoseTracePage() {
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const opened = useStore((state) => state.opened);
  const prefersDarkMode = usePrefersDarkMode();

  const { file, loading: fileLoading, error: fileError } = useResolvedFile(fileUrl);
  const {
    source,
    loading: sourceLoading,
    error: sourceError,
  } = usePoseTraceSource(file);

  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [rows, setRows] = useState<DemoRow[]>([]);
  const [rowsLoading, setRowsLoading] = useState(false);
  const [rowsError, setRowsError] = useState<string | null>(null);

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
      setRows([]);
      setRowsLoading(false);
      setRowsError(null);
      return;
    }

    let cancelled = false;
    setRowsLoading(true);
    setRowsError(null);

    Promise.resolve()
      .then(() => loadDemoRows(source, selectedDemo))
      .then((nextRows) => {
        if (cancelled) {
          return;
        }

        setRows(nextRows);
        setRowsLoading(false);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setRows([]);
        setRowsLoading(false);
        setRowsError(error instanceof Error ? error.message : String(error));
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDemo, source]);

  const selectionText = useMemo(() => {
    if (!selectedDemo || rows.length === 0) {
      return null;
    }

    const demo = demos.find((entry) => entry.name === selectedDemo);
    return {
      demo: selectedDemo,
      samples: demo?.num_samples ?? rows.length,
      source: demo?.source_episode_index ?? 'n/a',
      success: demo?.success != null ? Number(demo.success) : 'n/a',
    };
  }, [demos, rows, selectedDemo]);

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Analysis</p>
          <h1 className={styles.title}>Pose Trace</h1>
          <p className={styles.subtitle}>
            Plot end-effector and garment trajectories directly from the currently opened HDF5 file.
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
          <p>Loading pose-trace data…</p>
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
                <label className={styles.fieldLabel} htmlFor="demo-select">
                  Demo
                </label>
                <select
                  id="demo-select"
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
            </div>
          </section>

          {rowsError && (
            <section className={styles.messageCard}>
              <p className={styles.errorText}>{rowsError}</p>
            </section>
          )}

          <PoseTraceCharts
            rows={rows}
            loading={rowsLoading}
            themeKey={prefersDarkMode ? 'dark' : 'light'}
          />
        </>
      )}

      {!source && !fileLoading && !sourceLoading && !fileError && !sourceError && fileUrl && (
        <section className={styles.messageCard}>
          <p>
            Select an opened file from the sidebar to render pose traces, or go back to the viewer.
          </p>
          <div className={styles.emptyActions}>
            <Link
              className={styles.openBtn}
              to={`/pose-trace?${createSearchParams({ url: fileUrl }).toString()}`}
            >
              <FiActivity aria-hidden />
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

export default PoseTracePage;
