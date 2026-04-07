import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FiCheckCircle,
  FiChevronLeft,
  FiCloud,
  FiDownload,
  FiFolder,
  FiKey,
  FiLoader,
  FiPlay,
  FiUpload,
  FiXCircle,
} from 'react-icons/fi';

import { checkBackend, type PythonBackendStatus } from './python-backend';
import { useStore } from './stores';
import styles from './DatabricksPage.module.css';

const BASE_URL = 'http://localhost:4095';

const VOLUMES = [
  'workspace.default.mimicgen_annotated_hdf5_datasets',
  'workspace.default.assets',
  'workspace.default.finetune_lerobot_datasets',
  'workspace.default.hdf52lerobot_script_files_metrics',
  'workspace.default.hdf5_test',
  'workspace.default.hdf5datasets_lehome_many_clothes',
  'workspace.default.lerobot_datasets_lehome_many_clothes',
  'workspace.default.modality_files',
  'workspace.default.trained_models',
];

const PIPELINES = [
  { id: '501226726732882', name: 'AI-ROBOTICS-GENERATE-SYNTHETIC-DATA' },
  { id: '1091040782230484', name: 'AI-ROBOTICS-CONVERT-ALL-HDF5-TO-LEROBOT' },
  { id: '129636104529500', name: 'check-for-instances' },
  { id: '273456906261750', name: 'AI-ROBOTICS-PIPELINE-LEROBOT-UPLOAD' },
  { id: '449953930494301', name: 'AI-ROBOTICS-SSH-FINETUNE-GR00T-PIPELINE' },
];

interface JobRun {
  jobId: string;
  runId: string | null;
  name: string;
  state: string;
  resultState: string;
  message: string;
  polling: boolean;
}

interface VolumeFile {
  name: string;
  type: string;
  size: number;
}

// ---------------------------------------------------------------------------
// Secrets Section
// ---------------------------------------------------------------------------

function SecretsSection({ backendAvailable }: { backendAvailable: boolean }) {
  const [maxSteps, setMaxSteps] = useState('');
  const [saveSteps, setSaveSteps] = useState('');
  const [brevToken, setBrevToken] = useState('');
  const [instanceName, setInstanceName] = useState('');
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  async function handleSave() {
    const secrets: Record<string, string> = {};
    if (maxSteps) {
      secrets.max_steps = maxSteps;
    }
    if (saveSteps) {
      secrets.save_steps = saveSteps;
    }
    if (brevToken) {
      secrets.token = brevToken;
    }
    if (instanceName) {
      secrets.instance = instanceName;
    }

    if (Object.keys(secrets).length === 0) {
      setResult('No secrets to save.');
      return;
    }

    setSaving(true);
    setResult(null);

    try {
      const response = await fetch(`${BASE_URL}/api/databricks/put-secrets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ secrets, scope: 'brev' }),
      });

      const data: { allOk: boolean; results: { key: string; ok: boolean; error: string | null }[] } = await response.json();
      if (data.allOk) {
        setResult(`Saved ${Object.keys(secrets).length} secret(s).`);
        setMaxSteps('');
        setSaveSteps('');
        setBrevToken('');
        setInstanceName('');
      } else {
        const failed = data.results.filter((r) => !r.ok).map((r) => `${r.key}: ${String(r.error)}`);
        setResult(`Some secrets failed: ${failed.join(', ')}`);
      }
    } catch (error: unknown) {
      setResult(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className={styles.card}>
      <h2 className={styles.sectionTitle}><FiKey aria-hidden /> Secrets</h2>
      <p className={styles.sectionText}>Push secrets to the Databricks <code>brev</code> scope.</p>
      <div className={styles.secretsGrid}>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>max_steps</span>
          <input className={styles.input} type="text" value={maxSteps} onChange={(e) => { setMaxSteps(e.target.value); }} placeholder="e.g. 5000" />
        </label>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>save_steps</span>
          <input className={styles.input} type="text" value={saveSteps} onChange={(e) => { setSaveSteps(e.target.value); }} placeholder="e.g. 2500" />
        </label>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>token</span>
          <input className={styles.input} type="password" value={brevToken} onChange={(e) => { setBrevToken(e.target.value); }} placeholder="Brev API token" />
        </label>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>instance</span>
          <input className={styles.input} type="text" value={instanceName} onChange={(e) => { setInstanceName(e.target.value); }} placeholder="e.g. my-gpu-instance" />
        </label>
      </div>
      <div className={styles.actionRow}>
        <button type="button" className={styles.primaryBtn} onClick={() => { void handleSave(); }} disabled={saving || !backendAvailable}>
          {saving ? <FiLoader aria-hidden className={styles.spin} /> : <FiUpload aria-hidden />}
          <span>{saving ? 'Saving...' : 'Push Secrets'}</span>
        </button>
        {result && (
          <span className={result.startsWith('Saved') ? styles.successText : styles.resultText}>
            {result}
          </span>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Upload Dataset Section
// ---------------------------------------------------------------------------

function UploadSection({ backendAvailable }: { backendAvailable: boolean }) {
  const opened = useStore((state) => state.opened);
  const [selectedFile, setSelectedFile] = useState('');
  const [volume, setVolume] = useState('/Volumes/workspace/default/mimicgen_annotated_hdf5_datasets/');
  const [uploading, setUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [uploadLog, setUploadLog] = useState<string[]>([]);

  async function handleUpload() {
    if (!selectedFile) {
      return;
    }

    const h5File = opened.find((f) => f.url === selectedFile);
    if (!h5File) {
      return;
    }

    const filePath = h5File.name;
    setUploading(true);
    setUploadPercent(0);
    setUploadStatus('Starting upload...');
    setUploadLog([]);

    try {
      const response = await fetch(`${BASE_URL}/api/databricks/upload-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filePath, volume }),
      });

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No stream');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed.startsWith('data: ')) {
            continue;
          }

          try {
            const event: Record<string, unknown> = JSON.parse(trimmed.slice(6));
            if (event.type === 'progress') {
              setUploadPercent(event.percent as number);
              setUploadStatus(event.line as string);
            } else if (event.type === 'output' || event.type === 'start') {
              const line = (event.line ?? event.type) as string;
              setUploadLog((prev) => [...prev, line]);
              setUploadStatus(line);
            } else if (event.type === 'done') {
              setUploadPercent(100);
              setUploadStatus('Upload complete.');
            } else if (event.type === 'error') {
              setUploadStatus(`Error: ${event.message as string}`);
              setUploadLog((prev) => [...prev, `Error: ${event.message as string}`]);
            }
          } catch {
            // skip
          }
        }
      }
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      setUploadStatus(`Error: ${msg}`);
      setUploadLog((prev) => [...prev, `Error: ${msg}`]);
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className={styles.card}>
      <h2 className={styles.sectionTitle}><FiUpload aria-hidden /> Upload Dataset</h2>
      <p className={styles.sectionText}>Upload an opened dataset to a Databricks volume.</p>
      <div className={styles.controlGrid}>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Dataset</span>
          <select className={styles.input} value={selectedFile} onChange={(e) => { setSelectedFile(e.target.value); }}>
            <option value="">Select a dataset...</option>
            {opened.map((f) => (
              <option key={f.url} value={f.url}>{f.name}</option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Volume</span>
          <select className={styles.input} value={volume} onChange={(e) => { setVolume(e.target.value); }}>
            {VOLUMES.map((v) => (
              <option key={v} value={`/Volumes/${v.replaceAll('.', '/')}/`}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.actionRow}>
        <button type="button" className={styles.primaryBtn} onClick={() => { void handleUpload(); }} disabled={uploading || !selectedFile || !backendAvailable}>
          {uploading ? <FiLoader aria-hidden className={styles.spin} /> : <FiCloud aria-hidden />}
          <span>{uploading ? 'Uploading...' : 'Upload'}</span>
        </button>
      </div>
      {(uploading || uploadPercent > 0) && (
        <div>
          <div className={styles.uploadProgressRow}>
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: `${String(uploadPercent)}%` }} />
            </div>
            <span className={styles.uploadPercentText}>{uploadPercent}%</span>
          </div>
          {uploadStatus && <p className={styles.uploadStatusText}>{uploadStatus}</p>}
        </div>
      )}
      {uploadLog.length > 0 && (
        <pre className={styles.logBox}>{uploadLog.join('\n')}</pre>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Pipelines Section
// ---------------------------------------------------------------------------

function PipelinesSection({ backendAvailable }: { backendAvailable: boolean }) {
  const [runs, setRuns] = useState<JobRun[]>([]);
  const pollTimers = useRef(new Map<string, ReturnType<typeof setInterval>>());

  // Clean up poll timers on unmount.
  useEffect(() => {
    const timers = pollTimers.current;
    return () => {
      for (const timer of timers.values()) {
        clearInterval(timer);
      }
    };
  }, []);

  const pollStatus = useCallback((runId: string) => {
    const poll = async () => {
      try {
        const response = await fetch(`${BASE_URL}/api/databricks/job-status?run_id=${runId}`);
        const data: { lifeCycleState: string; resultState: string; stateMessage: string } = await response.json();

        setRuns((prev) => prev.map((r) =>
          r.runId === runId
            ? { ...r, state: data.lifeCycleState ?? '', resultState: data.resultState ?? '', message: data.stateMessage ?? '' }
            : r,
        ));

        const terminal = ['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'];
        if (terminal.includes(data.lifeCycleState)) {
          const timer = pollTimers.current.get(runId);
          if (timer) {
            clearInterval(timer);
            pollTimers.current.delete(runId);
          }
          setRuns((prev) => prev.map((r) => r.runId === runId ? { ...r, polling: false } : r));
        }
      } catch {
        // keep polling
      }
    };

    void poll();
    const timer = setInterval(() => { void poll(); }, 10_000);
    pollTimers.current.set(runId, timer);
  }, []);

  // Fetch active runs on mount.
  useEffect(() => {
    if (!backendAvailable) {
      return;
    }

    const jobIds = PIPELINES.map((p) => p.id).join(',');

    void fetch(`${BASE_URL}/api/databricks/active-runs?job_ids=${jobIds}`)
      .then(async (response) => {
        if (!response.ok) {
          return;
        }

        const data: { runs: { jobId: string; runId: string; runName: string; lifeCycleState: string; resultState: string; stateMessage: string }[] } = await response.json();

        const activeRuns: JobRun[] = data.runs.map((r) => ({
          jobId: r.jobId,
          runId: r.runId,
          name: r.runName || PIPELINES.find((p) => p.id === r.jobId)?.name || r.jobId,
          state: r.lifeCycleState,
          resultState: r.resultState,
          message: r.stateMessage,
          polling: !['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'].includes(r.lifeCycleState),
        }));

        setRuns(activeRuns);

        // Start polling non-terminal runs.
        for (const run of activeRuns) {
          if (run.polling && run.runId) {
            pollStatus(run.runId);
          }
        }
      })
      .catch(() => {
        // Ignore — just means no active runs or server issue.
      });
  }, [backendAvailable, pollStatus]);

  async function handleRun(jobId: string, name: string) {
    try {
      const response = await fetch(`${BASE_URL}/api/databricks/run-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId }),
      });

      const data: { ok: boolean; runId: string | null } = await response.json();
      const runId = data.runId ? String(data.runId) : null;

      const run: JobRun = {
        jobId,
        runId,
        name,
        state: runId ? 'PENDING' : 'UNKNOWN',
        resultState: '',
        message: runId ? 'Triggered' : 'No run ID returned',
        polling: Boolean(runId),
      };

      setRuns((prev) => [run, ...prev]);

      if (runId) {
        pollStatus(runId);
      }
    } catch (error: unknown) {
      setRuns((prev) => [{
        jobId,
        runId: null,
        name,
        state: 'ERROR',
        resultState: '',
        message: error instanceof Error ? error.message : String(error),
        polling: false,
      }, ...prev]);
    }
  }

  return (
    <section className={styles.card}>
      <h2 className={styles.sectionTitle}><FiPlay aria-hidden /> Pipelines</h2>
      <p className={styles.sectionText}>Trigger Databricks jobs and monitor their status.</p>
      <div className={styles.pipelineGrid}>
        {PIPELINES.map((p) => (
          <button
            key={p.id}
            type="button"
            className={styles.pipelineBtn}
            onClick={() => { void handleRun(p.id, p.name); }}
            disabled={!backendAvailable}
          >
            <FiPlay aria-hidden />
            <span>{p.name}</span>
          </button>
        ))}
      </div>
      {runs.length > 0 && (
        <div className={styles.runList}>
          {runs.map((run, i) => (
            <div key={run.runId ?? `${run.jobId}-${String(i)}`} className={styles.runItem}>
              {run.polling ? (
                <FiLoader aria-hidden className={styles.spin} />
              ) : run.resultState === 'SUCCESS' ? (
                <FiCheckCircle aria-hidden style={{ color: '#a9e4a2' }} />
              ) : run.state === 'ERROR' || run.resultState === 'FAILED' ? (
                <FiXCircle aria-hidden style={{ color: 'var(--color-danger)' }} />
              ) : (
                <FiCloud aria-hidden />
              )}
              <div className={styles.runInfo}>
                <strong>{run.name}</strong>
                <span>{run.state}{run.resultState ? ` / ${run.resultState}` : ''}</span>
                {run.message && <small>{run.message}</small>}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Volume Browser Section
// ---------------------------------------------------------------------------

function VolumeBrowserSection({ backendAvailable }: { backendAvailable: boolean }) {
  const [volume, setVolume] = useState(VOLUMES[VOLUMES.length - 1]); // trained_models default
  const [pathStack, setPathStack] = useState<string[]>([]);
  const [files, setFiles] = useState<VolumeFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadDst, setDownloadDst] = useState('/workspace/IsaacTools/downloads');
  const [downloading, setDownloading] = useState<string | null>(null);
  const [downloadLog, setDownloadLog] = useState<string[]>([]);

  const currentPath = pathStack.join('/');

  const loadFiles = useCallback(async (vol: string, subPath: string) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ volume: vol });
      if (subPath) {
        params.set('path', subPath);
      }
      const response = await fetch(`${BASE_URL}/api/databricks/volume-files?${params.toString()}`);
      if (!response.ok) {
        const body: { error?: string } = await response.json();
        throw new Error(body.error ?? `Status ${String(response.status)}`);
      }
      const data: { files: VolumeFile[] } = await response.json();
      setFiles(data.files);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (backendAvailable) {
      void loadFiles(volume, currentPath);
    }
  }, [backendAvailable, volume, currentPath, loadFiles]);

  function navigateInto(dirName: string) {
    setPathStack((prev) => [...prev, dirName.replace(/\/$/, '')]);
  }

  function navigateUp() {
    setPathStack((prev) => prev.slice(0, -1));
  }

  function changeVolume(newVolume: string) {
    setVolume(newVolume);
    setPathStack([]);
  }

  async function handleDownload(fileName: string) {
    const volumeRoot = `dbfs:/Volumes/${volume.replaceAll('.', '/')}`;
    const src = currentPath ? `${volumeRoot}/${currentPath}/${fileName}` : `${volumeRoot}/${fileName}`;
    const dstPath = `${downloadDst.replace(/\/$/, '')}/${fileName}`;

    setDownloading(fileName);
    setDownloadLog([]);

    try {
      const params = new URLSearchParams({ src, dst: dstPath });
      const response = await fetch(`${BASE_URL}/api/databricks/volume-download?${params.toString()}`);
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No stream');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed.startsWith('data: ')) {
            continue;
          }
          try {
            const event: Record<string, unknown> = JSON.parse(trimmed.slice(6));
            const line = (event.line ?? event.message ?? event.dst ?? event.type) as string;
            setDownloadLog((prev) => [...prev, line]);
          } catch {
            // skip
          }
        }
      }
    } catch (err: unknown) {
      setDownloadLog((prev) => [...prev, `Error: ${err instanceof Error ? err.message : String(err)}`]);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <section className={styles.card}>
      <h2 className={styles.sectionTitle}><FiDownload aria-hidden /> Volume Browser</h2>
      <p className={styles.sectionText}>Browse and download files from Databricks volumes.</p>
      <div className={styles.controlGrid}>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Volume</span>
          <select className={styles.input} value={volume} onChange={(e) => { changeVolume(e.target.value); }}>
            {VOLUMES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Download to</span>
          <input className={styles.input} type="text" value={downloadDst} onChange={(e) => { setDownloadDst(e.target.value); }} />
        </label>
      </div>

      <div className={styles.browserBar}>
        <button type="button" className={styles.backBtn} onClick={navigateUp} disabled={pathStack.length === 0}>
          <FiChevronLeft aria-hidden />
        </button>
        <code className={styles.breadcrumb}>
          /Volumes/{volume.replaceAll('.', '/')}/{currentPath}
        </code>
      </div>

      {loading && <p className={styles.infoText}>Loading...</p>}
      {error && <p className={styles.errorText}>{error}</p>}

      {!loading && files.length > 0 && (
        <div className={styles.fileList}>
          {files.map((f) => (
            <div key={f.name} className={styles.fileRow}>
              {f.type === 'DIRECTORY' ? (
                <button type="button" className={styles.fileLink} onClick={() => { navigateInto(f.name); }}>
                  <FiFolder aria-hidden />
                  <span>{f.name}</span>
                </button>
              ) : (
                <>
                  <span className={styles.fileName}>{f.name}</span>
                  {f.size > 0 && (
                    <small className={styles.fileSize}>
                      {f.size > 1_073_741_824
                        ? `${(f.size / 1_073_741_824).toFixed(1)} GB`
                        : f.size > 1_048_576
                          ? `${(f.size / 1_048_576).toFixed(1)} MB`
                          : `${(f.size / 1024).toFixed(1)} KB`}
                    </small>
                  )}
                  <button
                    type="button"
                    className={styles.dlBtn}
                    onClick={() => { void handleDownload(f.name); }}
                    disabled={Boolean(downloading) || !backendAvailable}
                  >
                    {downloading === f.name ? <FiLoader aria-hidden className={styles.spin} /> : <FiDownload aria-hidden />}
                  </button>
                </>
              )}
            </div>
          ))}
        </div>
      )}

      {!loading && files.length === 0 && !error && <p className={styles.infoText}>No files found.</p>}

      {downloadLog.length > 0 && (
        <pre className={styles.logBox}>{downloadLog.join('\n')}</pre>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

function DatabricksPage() {
  const [backend, setBackend] = useState<PythonBackendStatus>({ available: false, rootDir: null, version: null });

  useEffect(() => {
    let cancelled = false;
    void checkBackend().then((status) => {
      if (!cancelled) {
        setBackend(status);
      }
    });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Databricks</p>
          <h1 className={styles.title}>Databricks</h1>
          <p className={styles.subtitle}>
            Manage secrets, upload datasets, trigger pipelines, and download results.
          </p>
        </div>
        {!backend.available && (
          <p className={styles.errorText}>
            Python server not detected. Start with: <code>MERGE_SERVER_DIR=/path/to/data pnpm start</code>
          </p>
        )}
      </header>

      <SecretsSection backendAvailable={backend.available} />
      <UploadSection backendAvailable={backend.available} />
      <PipelinesSection backendAvailable={backend.available} />
      <VolumeBrowserSection backendAvailable={backend.available} />
    </div>
  );
}

export default DatabricksPage;
