import { startTransition, useEffect, useMemo, useState } from 'react';
import { FiActivity } from 'react-icons/fi';
import { createSearchParams, Link, useSearchParams } from 'react-router-dom';

import { formatUnknownError } from './error-utils';
import { loadDemoRows, openPoseTraceSource } from './pose-trace/hdf5';
import { type DemoRow, type PoseTraceSource } from './pose-trace/types';
import { PoseTraceCharts } from './PoseTraceCharts';
import { PoseTraceEmptyState } from './PoseTraceEmptyState';
import styles from './PoseTracePage.module.css';
import { formatDemoOption } from './PoseTracePage.utils';
import { type H5File, useStore } from './stores';
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

function noop(): void {
  return undefined;
}

function usePrefersDarkMode(): boolean {
  const [prefersDarkMode, setPrefersDarkMode] = useState(
    () => globalThis.matchMedia('(prefers-color-scheme: dark)').matches,
  );

  useEffect(() => {
    const mediaQuery = globalThis.matchMedia('(prefers-color-scheme: dark)');

    function updatePreference(event: MediaQueryListEvent) {
      setPrefersDarkMode(event.matches);
    }

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
      return undefined;
    }

    const openedFile = opened.find((file) => file.url === fileUrl);
    if (openedFile) {
      setState({ file: openedFile, loading: false, error: null });
      return undefined;
    }

    let cancelled = false;
    const resolvedFileUrl = fileUrl;
    setState({ file: null, loading: true, error: null });

    async function resolveFile() {
      try {
        const resolvedFile = await resolveFileUrl(resolvedFileUrl);
        if (cancelled) {
          return;
        }

        if (!resolvedFile) {
          setState({
            file: null,
            loading: false,
            error:
              'This file cannot be reopened automatically. Open it again from the home page.',
          });
          return;
        }

        openFiles([resolvedFile]);
        setState({ file: resolvedFile, loading: false, error: null });
      } catch (error: unknown) {
        if (!cancelled) {
          setState({
            file: null,
            loading: false,
            error: formatUnknownError(error),
          });
        }
      }
    }

    void resolveFile();

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
    let cleanup = noop;

    if (!file) {
      setState({ source: null, loading: false, error: null });
      return cleanup;
    }

    const resolvedFile = file;
    setState({ source: null, loading: true, error: null });

    async function openSource() {
      try {
        const source = await openPoseTraceSource(resolvedFile);
        if (cancelled) {
          source.cleanup();
          return;
        }

        ({ cleanup } = source);
        setState({ source, loading: false, error: null });
      } catch (error: unknown) {
        if (!cancelled) {
          setState({
            source: null,
            loading: false,
            error: formatUnknownError(error),
          });
        }
      }
    }

    void openSource();

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [file]);

  return state;
}

function PoseTracePage() {
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const opened = useStore((state) => state.opened);
  const prefersDarkMode = usePrefersDarkMode();
  const {
    file,
    loading: fileLoading,
    error: fileError,
  } = useResolvedFile(fileUrl);
  const {
    source,
    loading: sourceLoading,
    error: sourceError,
  } = usePoseTraceSource(file);
  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [rows, setRows] = useState<DemoRow[]>([]);
  const [rowsLoading, setRowsLoading] = useState(false);
  const [rowsError, setRowsError] = useState<string | null>(null);
  const demos = useMemo(() => source?.demos ?? [], [source]);

  useEffect(() => {
    if (demos.length === 0) {
      setSelectedDemo(null);
      return;
    }

    setSelectedDemo((current) =>
      current && demos.some((demo) => demo.name === current)
        ? current
        : demos[0].name,
    );
  }, [demos]);

  useEffect(() => {
    if (!source || !selectedDemo) {
      setRows([]);
      setRowsLoading(false);
      setRowsError(null);
      return undefined;
    }

    let cancelled = false;
    const resolvedSource = source;
    const resolvedDemo = selectedDemo;
    setRowsLoading(true);
    setRowsError(null);

    async function loadRows() {
      try {
        const nextRows = await loadDemoRows(resolvedSource, resolvedDemo);
        if (!cancelled) {
          setRows(nextRows);
          setRowsLoading(false);
        }
      } catch (error: unknown) {
        if (!cancelled) {
          setRows([]);
          setRowsLoading(false);
          setRowsError(formatUnknownError(error));
        }
      }
    }

    void loadRows();

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
      success: demo && demo.success !== null ? Number(demo.success) : 'n/a',
    };
  }, [demos, rows, selectedDemo]);

  function renderInitialEmptyState() {
    if (fileUrl || file || fileLoading) {
      return null;
    }

    return <PoseTraceEmptyState openedFileCount={opened.length} />;
  }

  function renderError(error: string | null) {
    if (!error) {
      return null;
    }

    return (
      <section className={styles.messageCard}>
        <p className={styles.errorText}>{error}</p>
      </section>
    );
  }

  function renderNoSourceState() {
    if (
      source ||
      fileLoading ||
      sourceLoading ||
      fileError ||
      sourceError ||
      !fileUrl
    ) {
      return null;
    }

    return (
      <section className={styles.messageCard}>
        <p>
          Select an opened file from the sidebar to render pose traces, or go
          back to the viewer.
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
    );
  }

  function renderSourceContent() {
    if (!source) {
      return null;
    }

    return (
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
                {demos.length === 0 && (
                  <option value="">No demos available</option>
                )}
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
                  <span className={styles.statusKey}>Selected:</span>{' '}
                  {selectionText.demo}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Samples:</span>{' '}
                  {selectionText.samples}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Source:</span>{' '}
                  {selectionText.source}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Success:</span>{' '}
                  {selectionText.success}
                </div>
              </>
            )}
          </div>
        </section>

        {renderError(rowsError)}

        <PoseTraceCharts
          rows={rows}
          loading={rowsLoading}
          themeKey={prefersDarkMode ? 'dark' : 'light'}
          articulation={source.articulation}
        />
      </>
    );
  }

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Analysis</p>
          <h1 className={styles.title}>Pose Trace</h1>
          <p className={styles.subtitle}>
            Plot end-effector and object trajectories directly from the
            currently opened HDF5 file.
          </p>
        </div>
      </header>

      {renderInitialEmptyState()}
      {renderError(fileError)}

      {(fileLoading || sourceLoading) && (
        <section className={styles.messageCard}>
          <p>Loading pose-trace data…</p>
        </section>
      )}

      {renderError(sourceError)}
      {renderSourceContent()}
      {renderNoSourceState()}
    </div>
  );
}

export default PoseTracePage;
