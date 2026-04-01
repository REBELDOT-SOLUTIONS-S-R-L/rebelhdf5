import { startTransition, useEffect, useMemo, useState } from 'react';
import { FiArrowUpRight, FiActivity } from 'react-icons/fi';
import { Link, createSearchParams, useSearchParams } from 'react-router-dom';

import { openPoseTraceSource, loadDemoRows } from './pose-trace/hdf5';
import Plot from './pose-trace/PlotlyChart';
import {
  build2DData,
  build2DLayout,
  build3DData,
  build3DLayout,
  buildEmptyLayout,
} from './pose-trace/plotConfig';
import type { DemoInfo, DemoRow, GraphMode, PoseTraceSource } from './pose-trace/types';
import { type H5File, useStore } from './stores';
import styles from './PoseTracePage.module.css';
import { getViewerLink, resolveFileUrl } from './utils';

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

function PoseTraceCharts({
  rows,
  graphMode,
  loading,
}: {
  rows: DemoRow[];
  graphMode: GraphMode;
  loading: boolean;
}) {
  const hasData = rows.length > 0;
  const emptyMessage = loading
    ? 'Loading pose-trace data…'
    : 'Select a demo to inspect its end-effector and garment trajectories.';

  return (
    <div className={styles.chartStack}>
      {(graphMode === 'both' || graphMode === '3d') && (
        <section className={styles.chartCard}>
          <Plot
            data={hasData ? build3DData(rows) : []}
            layout={
              hasData
                ? build3DLayout(rows)
                : buildEmptyLayout('3D Pose Trace', emptyMessage, true)
            }
            useResizeHandler
            style={{ width: '100%' }}
            config={{ responsive: true }}
          />
        </section>
      )}

      {(graphMode === 'both' || graphMode === '2d') && (
        <section className={styles.splitCharts}>
          {(['left', 'right'] as const).map((side) => (
            <div key={side} className={styles.chartCard}>
              <Plot
                key={`${rows[0]?.dataset_name ?? 'dataset'}-${rows[0]?.demo_name ?? 'demo'}-${side}`}
                data={hasData ? build2DData(rows, side) : []}
                layout={
                  hasData
                    ? build2DLayout(rows, side)
                    : buildEmptyLayout(
                        side === 'left'
                          ? 'Left EEF 2D Pose Trace'
                          : 'Right EEF 2D Pose Trace',
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
      )}
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

  const { file, loading: fileLoading, error: fileError } = useResolvedFile(fileUrl);
  const {
    source,
    loading: sourceLoading,
    error: sourceError,
  } = usePoseTraceSource(file);

  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [graphMode, setGraphMode] = useState<GraphMode>('both');
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
        {source && <p className={styles.datasetName}>{source.datasetName}</p>}
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

              <div className={styles.field}>
                <span className={styles.fieldLabel}>Graph Mode</span>
                <div className={styles.modeGroup} role="group" aria-label="Graph mode">
                  {(['both', '3d', '2d'] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      className={styles.modeButton}
                      data-active={graphMode === mode || undefined}
                      onClick={() => setGraphMode(mode)}
                    >
                      {mode === 'both' ? 'Both' : mode.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {file && (
                <Link
                  className={styles.viewerLink}
                  to={getViewerLink(file.url)}
                >
                  Open In Viewer
                  <FiArrowUpRight aria-hidden />
                </Link>
              )}
            </div>

            <div className={styles.statusRow}>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>File:</span>{' '}
                {file?.name ?? source.datasetName}
              </div>
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

          <PoseTraceCharts rows={rows} graphMode={graphMode} loading={rowsLoading} />
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
