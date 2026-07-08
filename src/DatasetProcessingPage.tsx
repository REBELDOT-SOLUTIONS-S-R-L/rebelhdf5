import { useEffect, useMemo, useRef, useState } from 'react';
import {
  FiChevronDown,
  FiChevronRight,
  FiDownload,
  FiFile,
  FiFolder,
  FiLoader,
  FiRefreshCw,
  FiServer,
} from 'react-icons/fi';
import { HiFolder } from 'react-icons/hi';
import { createSearchParams, Link, useSearchParams } from 'react-router-dom';

import styles from './DatasetProcessingPage.module.css';
import {
  buildBackendKeyInfos,
  buildDefaultOutputName,
  buildKeyTree,
  type KeyTreeNode,
  parseTaskRulesJson,
  sumKeyInfos,
} from './DatasetProcessingPage.utils';
import {
  useDatasetProcessingSources,
  useResolvedFile,
} from './dataset-selection';
import {
  getDatasetProcessingInfo,
  openPoseTraceSource,
  processDataset,
} from './pose-trace/hdf5';
import {
  type DatasetProcessingOperation,
  type DatasetProcessingProgress,
} from './pose-trace/types';
import {
  pollBackendStatus,
  type PythonBackendStatus,
  type PythonScanResult,
  runLeRobotConvert,
  runProcess,
  scanFiles,
} from './python-backend';
import { type H5File, useStore } from './stores';

interface ProcessResultState {
  fileName: string;
  demoCount: number;
  selectedKeyCount: number;
  outputPath?: string;
  outputType?: 'file' | 'directory';
  skippedDemoCount?: number;
  totalFrames?: number;
  taskCount?: number;
  downloadUrl?: string;
  downloadBlob?: Blob;
}

interface SourceOption {
  id: string;
  name: string;
  label: string;
  backendPath?: string;
}

const OPERATION_LABELS: Record<DatasetProcessingOperation, string> = {
  cut: 'Cut',
  merge: 'Merge',
  append: 'Append',
  lerobot: 'LeRobot',
};
const DATASET_PROCESSING_OPERATIONS: DatasetProcessingOperation[] = [
  'cut',
  'merge',
  'append',
  'lerobot',
];
const BACKEND_SOURCE_PREFIX = 'backend:';
const DEFAULT_LEROBOT_MODALITY_JSON =
  '/workspace/IsaacTools/ROBOTICS-lehome-challenge/configs/gr00t/modality.json';
const DEFAULT_LEROBOT_TASK = 'Complete the task';

function getBackendSourceId(path: string): string {
  return `${BACKEND_SOURCE_PREFIX}${path}`;
}

function getDesktopFilePath(file: File): string | undefined {
  try {
    return globalThis.rebelHdf5Desktop?.getPathForFile?.(file) || undefined;
  } catch {
    return undefined;
  }
}

function triggerDownloadBlob(fileName: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.click();
  globalThis.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
}

function triggerDownloadUrl(fileName: string, downloadUrl: string) {
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = fileName;
  link.click();
}

function KeyTreeNodeItem({
  node,
  depth,
  selectedKeys,
  collapsedGroups,
  onToggleKey,
  onToggleGroup,
  onToggleCollapse,
}: {
  node: KeyTreeNode;
  depth: number;
  selectedKeys: ReadonlySet<string>;
  collapsedGroups: ReadonlySet<string>;
  onToggleKey: (keyPath: string) => void;
  onToggleGroup: (keyPaths: readonly string[]) => void;
  onToggleCollapse: (groupPath: string) => void;
}) {
  if (node.keyInfo) {
    return (
      <label
        className={styles.treeLeaf}
        style={{ paddingLeft: `${depth * 1.1 + 0.75}rem` }}
        title={node.fullPath}
      >
        <input
          type="checkbox"
          checked={selectedKeys.has(node.fullPath)}
          onChange={() => {
            onToggleKey(node.fullPath);
          }}
        />
        <FiFile aria-hidden className={styles.treeLeafIcon} />
        <span className={styles.treeLeafLabel}>{node.name}</span>
        <small className={styles.treeMeta}>
          {node.keyInfo.availableInDemoCount}
        </small>
      </label>
    );
  }

  const selectedCount = node.leafKeyPaths.filter((keyPath) =>
    selectedKeys.has(keyPath),
  ).length;
  const allSelected = selectedCount === node.leafKeyPaths.length;
  const isCollapsed = collapsedGroups.has(node.fullPath);
  const FolderIcon = allSelected ? HiFolder : FiFolder;

  return (
    <div className={styles.treeBranch}>
      <div
        className={styles.treeBranchRow}
        style={{ paddingLeft: `${depth * 1.1}rem` }}
      >
        <button
          type="button"
          className={styles.treeBranchSelectBtn}
          title={`Toggle all keys under ${node.fullPath}`}
          aria-pressed={allSelected}
          onClick={() => {
            onToggleGroup(node.leafKeyPaths);
          }}
        >
          <FolderIcon aria-hidden className={styles.treeBranchIcon} />
        </button>
        <button
          type="button"
          className={styles.treeBranchLabel}
          title={node.fullPath}
          aria-expanded={!isCollapsed}
          onClick={() => {
            onToggleCollapse(node.fullPath);
          }}
        >
          {isCollapsed ? (
            <FiChevronRight aria-hidden className={styles.treeCollapseIcon} />
          ) : (
            <FiChevronDown aria-hidden className={styles.treeCollapseIcon} />
          )}
          <span className={styles.treeBranchName}>{node.name}</span>
        </button>
        <small className={styles.treeMeta}>
          {selectedCount}/{node.leafKeyPaths.length}
        </small>
      </div>
      {!isCollapsed && (
        <div className={styles.treeChildren}>
          {node.children.map((child) => (
            <KeyTreeNodeItem
              key={child.fullPath}
              node={child}
              depth={depth + 1}
              selectedKeys={selectedKeys}
              collapsedGroups={collapsedGroups}
              onToggleKey={onToggleKey}
              onToggleGroup={onToggleGroup}
              onToggleCollapse={onToggleCollapse}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function EmptyState({ openedFileCount }: { openedFileCount: number }) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Dataset Processing</h2>
      <p className={styles.emptyText}>
        Open one or more HDF5 files in rebelHDF5, then switch to this page to
        cut, merge, append, or convert datasets.
      </p>
      <div className={styles.emptyActions}>
        <Link className={styles.openBtn} to="/">
          Open HDF5
        </Link>
        {openedFileCount > 0 && (
          <span>
            {openedFileCount} opened file{openedFileCount === 1 ? '' : 's'}{' '}
            available in the sidebar.
          </span>
        )}
      </div>
    </div>
  );
}

function DatasetProcessingPage() {
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const opened = useStore((state) => state.opened);

  const {
    file,
    loading: fileLoading,
    error: fileError,
  } = useResolvedFile(fileUrl);
  const [operation, setOperation] = useState<DatasetProcessingOperation>('cut');
  const [primarySourceUrl, setPrimarySourceUrl] = useState<string | null>(null);
  const [mergeSourceUrls, setMergeSourceUrls] = useState<string[]>([]);
  const [appendSourceUrls, setAppendSourceUrls] = useState<string[]>([]);
  const [lerobotSourceUrls, setLerobotSourceUrls] = useState<string[]>([]);
  const [skipFailedDemos, setSkipFailedDemos] = useState(true);
  const [lerobotModalityJsonPath, setLerobotModalityJsonPath] = useState(
    DEFAULT_LEROBOT_MODALITY_JSON,
  );
  const [lerobotConversionConfigPath, setLerobotConversionConfigPath] =
    useState('');
  const [lerobotModalityPythonPath, setLerobotModalityPythonPath] =
    useState('');
  const [lerobotDefaultTask, setLerobotDefaultTask] =
    useState(DEFAULT_LEROBOT_TASK);
  const [lerobotTaskRulesText, setLerobotTaskRulesText] = useState('');
  const [cutStartDemoName, setCutStartDemoName] = useState<string | null>(null);
  const [cutEndDemoName, setCutEndDemoName] = useState<string | null>(null);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [collapsedGroupPaths, setCollapsedGroupPaths] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ProcessResultState | null>(null);
  const [progress, setProgress] = useState<DatasetProcessingProgress | null>(
    null,
  );

  // Python backend state.
  const [backend, setBackend] = useState<PythonBackendStatus>({
    available: false,
    rootDir: null,
    version: null,
  });
  const [useBackend, setUseBackend] = useState(false);

  const [backendScan, setBackendScan] = useState<PythonScanResult | null>(null);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const previousAvailableKeyPathsRef = useRef<string[]>([]);
  const backendAutoEnabledRef = useRef(false);

  // Detect Python backend on mount and keep retrying while Vite starts it.
  useEffect(() => {
    return pollBackendStatus((status) => {
      setBackend(status);
      if (status.available && !backendAutoEnabledRef.current) {
        backendAutoEnabledRef.current = true;
        setUseBackend(true);
      }
    });
  }, []);

  useEffect(() => {
    if (!useBackend || !backend.available) {
      setBackendScan(null);
      setBackendLoading(false);
    }
  }, [backend.available, useBackend]);

  useEffect(() => {
    if (operation === 'lerobot' && backend.available && !useBackend) {
      setUseBackend(true);
    }
  }, [backend.available, operation, useBackend]);

  const availableFiles = useMemo(() => {
    const byUrl = new Map<string, H5File>();
    if (file) {
      byUrl.set(file.url, file);
    }
    for (const openedFile of opened) {
      if (!byUrl.has(openedFile.url)) {
        byUrl.set(openedFile.url, openedFile);
      }
    }

    return [...byUrl.values()];
  }, [file, opened]);

  // Source options are always restricted to files the user has opened. In
  // backend mode we prefer explicit Electron filesystem paths, with a backend
  // unique-basename fallback for files opened before path metadata existed.
  const { sourceOptions, skippedNames } = useMemo<{
    sourceOptions: SourceOption[];
    skippedNames: string[];
  }>(() => {
    if (useBackend && backend.available) {
      const options: SourceOption[] = [];
      for (const f of availableFiles) {
        const backendPath = f.serverPath ?? f.name;
        options.push({
          id: getBackendSourceId(backendPath),
          name: f.name,
          label: f.name,
          backendPath,
        });
      }
      return { sourceOptions: options, skippedNames: [] };
    }

    return {
      sourceOptions: availableFiles.map((f) => ({
        id: f.url,
        name: f.name,
        label: f.name,
      })),
      skippedNames: [],
    };
  }, [availableFiles, backend.available, useBackend]);

  const sourceOptionMap = useMemo(
    () => new Map(sourceOptions.map((entry) => [entry.id, entry])),
    [sourceOptions],
  );

  useEffect(() => {
    if (sourceOptions.length === 0) {
      setPrimarySourceUrl(null);
      return;
    }

    const activeId =
      (!useBackend ? (fileUrl ?? file?.url) : null) ??
      sourceOptions[0]?.id ??
      null;
    setPrimarySourceUrl((current) =>
      current && sourceOptions.some((entry) => entry.id === current)
        ? current
        : activeId,
    );
  }, [file, fileUrl, sourceOptions, useBackend]);

  useEffect(() => {
    const availableIds = new Set(sourceOptions.map((entry) => entry.id));
    const activeId = primarySourceUrl ?? sourceOptions[0]?.id ?? null;
    const firstOther = sourceOptions.find((entry) => entry.id !== activeId)?.id;

    setMergeSourceUrls((current) => {
      const valid = current.filter(
        (url) => availableIds.has(url) && url !== activeId,
      );
      if (valid.length > 0) {
        return valid;
      }

      return firstOther ? [firstOther] : [];
    });

    setAppendSourceUrls((current) => {
      const valid = current.filter(
        (url) => availableIds.has(url) && url !== activeId,
      );
      if (valid.length > 0) {
        return valid;
      }

      return firstOther ? [firstOther] : [];
    });

    setLerobotSourceUrls((current) => {
      const valid = current.filter((url) => availableIds.has(url));
      if (valid.length > 0) {
        return valid;
      }

      return sourceOptions.map((entry) => entry.id);
    });
  }, [primarySourceUrl, sourceOptions]);

  const orderedSelectedSourceUrls = useMemo(() => {
    if (operation === 'cut') {
      return primarySourceUrl ? [primarySourceUrl] : [];
    }

    if (operation === 'lerobot') {
      return sourceOptions
        .map((entry) => entry.id)
        .filter((id) => lerobotSourceUrls.includes(id));
    }

    return sourceOptions
      .map((entry) => entry.id)
      .filter(
        (id) =>
          id === primarySourceUrl ||
          (operation === 'append'
            ? appendSourceUrls.includes(id)
            : mergeSourceUrls.includes(id)),
      );
  }, [
    appendSourceUrls,
    lerobotSourceUrls,
    mergeSourceUrls,
    operation,
    primarySourceUrl,
    sourceOptions,
  ]);
  const selectedSourceOptions = useMemo(
    () =>
      orderedSelectedSourceUrls
        .map((id) => sourceOptionMap.get(id))
        .filter((entry): entry is SourceOption => Boolean(entry)),
    [orderedSelectedSourceUrls, sourceOptionMap],
  );

  const resolveError = useMemo(() => {
    if (!useBackend || !backend.available) {
      return null;
    }

    const missingPaths = selectedSourceOptions
      .filter((entry) => !entry.backendPath)
      .map((entry) => entry.name);

    return missingPaths.length > 0
      ? `No local filesystem path for: ${missingPaths.join(', ')}. Reopen the file with the desktop file picker.`
      : null;
  }, [backend.available, selectedSourceOptions, useBackend]);

  const backendScanPaths = useMemo(() => {
    if (resolveError) {
      return [];
    }

    if (selectedSourceOptions.length === 0) {
      return [];
    }

    const paths = selectedSourceOptions.map((entry) => entry.backendPath);
    return paths.every((p): p is string => Boolean(p)) ? paths : [];
  }, [resolveError, selectedSourceOptions]);

  // Demo names for the primary source in backend mode.
  const backendPrimaryDemos = useMemo(() => {
    if (!useBackend || !backendScan || !primarySourceUrl) {
      return [];
    }

    const primaryOption = sourceOptionMap.get(primarySourceUrl);
    if (!primaryOption) {
      return [];
    }

    if (!primaryOption.backendPath) {
      return [];
    }

    const fileInfo = backendScan.files.find(
      (f) => f.path === primaryOption.backendPath,
    );
    return fileInfo?.demoNames ?? [];
  }, [backendScan, primarySourceUrl, sourceOptionMap, useBackend]);

  useEffect(() => {
    if (!useBackend || !backend.available || !backend.rootDir) {
      setBackendScan(null);
      setBackendLoading(false);
      return;
    }

    if (resolveError) {
      setBackendScan(null);
      setBackendError(resolveError);
      setBackendLoading(false);
      return;
    }

    if (backendScanPaths.length === 0) {
      setBackendScan(null);
      setBackendLoading(false);
      return;
    }

    let cancelled = false;
    setBackendLoading(true);
    setBackendError(null);

    void scanFiles(backendScanPaths)
      .then((result) => {
        if (!cancelled) {
          setBackendScan(result);
          setBackendLoading(false);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setBackendScan(null);
          setBackendError(
            error instanceof Error ? error.message : String(error),
          );
          setBackendLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    backend.available,
    backend.rootDir,
    backendScanPaths,
    resolveError,
    useBackend,
  ]);

  const sourceStates = useDatasetProcessingSources(
    availableFiles,
    orderedSelectedSourceUrls,
  );

  const primarySourceState = primarySourceUrl
    ? (sourceStates[primarySourceUrl] ?? null)
    : null;
  const primarySource = primarySourceState?.source ?? null;
  const primaryDemos = useBackend
    ? backendPrimaryDemos.map((name) => ({ name }))
    : (primarySource?.demos ?? []);

  useEffect(() => {
    if (primaryDemos.length === 0) {
      setCutStartDemoName(null);
      setCutEndDemoName(null);
      return;
    }

    setCutStartDemoName((current) =>
      current && primaryDemos.some((demo) => demo.name === current)
        ? current
        : primaryDemos[0].name,
    );
    setCutEndDemoName((current) =>
      current && primaryDemos.some((demo) => demo.name === current)
        ? current
        : primaryDemos[primaryDemos.length - 1].name,
    );
  }, [primaryDemos]);

  useEffect(() => {
    if (!cutStartDemoName || !cutEndDemoName) {
      return;
    }

    const startIndex = primaryDemos.findIndex(
      (demo) => demo.name === cutStartDemoName,
    );
    const endIndex = primaryDemos.findIndex(
      (demo) => demo.name === cutEndDemoName,
    );
    if (startIndex !== -1 && endIndex !== -1 && startIndex > endIndex) {
      setCutEndDemoName(cutStartDemoName);
    }
  }, [cutEndDemoName, cutStartDemoName, primaryDemos]);

  // Memoized so downstream memos/effects (e.g. availableKeyInfos → selectedKeys)
  // see a stable reference; otherwise a fresh array every render drives an
  // effect→setState→re-render loop.
  const selectedSourceStates = useMemo(
    () =>
      orderedSelectedSourceUrls.map((url) => sourceStates[url]).filter(Boolean),
    [orderedSelectedSourceUrls, sourceStates],
  );
  const selectedSourceFiles = selectedSourceStates.map((entry) => entry.file);
  const selectedSourceRefs =
    useBackend && backend.available
      ? selectedSourceOptions.map((entry) => ({ name: entry.name }))
      : selectedSourceFiles;
  const selectedSourceLoading = selectedSourceStates.some(
    (entry) => entry.loading,
  );
  const selectedSourceErrors = selectedSourceStates
    .filter((entry) => entry.error)
    .map((entry) => `${entry.file.name}: ${entry.error}`);
  const selectedSourcesReady =
    selectedSourceStates.length === orderedSelectedSourceUrls.length &&
    selectedSourceStates.every(
      (entry) => entry.source && entry.info && !entry.loading && !entry.error,
    );

  const availableKeyInfos = useMemo(() => {
    if (operation === 'lerobot') {
      return [];
    }

    if (useBackend && backendScan) {
      if (operation === 'cut') {
        const primaryOption = primarySourceUrl
          ? (sourceOptionMap.get(primarySourceUrl) ?? null)
          : null;
        const primaryPath = primaryOption?.backendPath ?? null;
        const primaryInfo =
          backendScan.files.find((entry) => entry.path === primaryPath) ?? null;
        return buildBackendKeyInfos(primaryInfo);
      }

      return sumKeyInfos(
        backendScan.files.flatMap((fileInfo) => buildBackendKeyInfos(fileInfo)),
      );
    }

    if (useBackend) {
      return [];
    }

    if (operation === 'cut') {
      return primarySourceState?.info?.keyPaths ?? [];
    }

    const readySources = selectedSourceStates.filter((entry) => entry.info);
    return sumKeyInfos(
      readySources.flatMap((entry) => entry.info?.keyPaths ?? []),
    );
  }, [
    backendScan,
    operation,
    primarySourceState,
    primarySourceUrl,
    selectedSourceStates,
    sourceOptionMap,
    useBackend,
  ]);

  const availableKeySet = useMemo(
    () => new Set(availableKeyInfos.map((keyInfo) => keyInfo.path)),
    [availableKeyInfos],
  );
  const baseKeyPaths = useMemo(
    () =>
      (primarySourceState?.info?.keyPaths ?? []).map((keyInfo) => keyInfo.path),
    [primarySourceState],
  );
  const keyTreeNodes = useMemo(
    () => buildKeyTree(availableKeyInfos),
    [availableKeyInfos],
  );
  const selectedKeySet = useMemo(() => new Set(selectedKeys), [selectedKeys]);
  const collapsedGroupSet = useMemo(
    () => new Set(collapsedGroupPaths),
    [collapsedGroupPaths],
  );

  useEffect(() => {
    const nextAvailableKeyPaths = availableKeyInfos.map(
      (keyInfo) => keyInfo.path,
    );
    const previousAvailableKeyPaths = previousAvailableKeyPathsRef.current;

    setSelectedKeys((current) => {
      const filtered = current.filter((key) => availableKeySet.has(key));
      const hadAllPreviousKeysSelected =
        previousAvailableKeyPaths.length > 0 &&
        current.length === previousAvailableKeyPaths.length &&
        previousAvailableKeyPaths.every((key) => current.includes(key));

      if (filtered.length > 0 && !hadAllPreviousKeysSelected) {
        return filtered;
      }

      return nextAvailableKeyPaths;
    });

    previousAvailableKeyPathsRef.current = nextAvailableKeyPaths;
  }, [availableKeyInfos, availableKeySet]);

  const cutDemoNames = useMemo(() => {
    if (!cutStartDemoName || !cutEndDemoName) {
      return [];
    }

    const startIndex = primaryDemos.findIndex(
      (demo) => demo.name === cutStartDemoName,
    );
    const endIndex = primaryDemos.findIndex(
      (demo) => demo.name === cutEndDemoName,
    );
    if (startIndex === -1 || endIndex === -1 || startIndex > endIndex) {
      return [];
    }

    return primaryDemos
      .slice(startIndex, endIndex + 1)
      .map((demo) => demo.name);
  }, [cutEndDemoName, cutStartDemoName, primaryDemos]);

  const defaultOutputName = useMemo(
    () => buildDefaultOutputName(operation, selectedSourceRefs, cutDemoNames),
    [cutDemoNames, operation, selectedSourceRefs],
  );
  const processingDescription = useMemo(() => {
    if (operation === 'lerobot') {
      return `Converting ${orderedSelectedSourceUrls.length} HDF5 dataset${orderedSelectedSourceUrls.length === 1 ? '' : 's'} to LeRobot v2.1 with GPU video encoding.`;
    }

    if (operation === 'merge') {
      return `Merging ${orderedSelectedSourceUrls.length} datasets into a new file. Video-heavy keys can take a while to copy.`;
    }

    if (operation === 'append') {
      return `Appending ${appendSourceUrls.length} dataset${appendSourceUrls.length === 1 ? '' : 's'} to the base dataset. Video-heavy keys can take a while to copy.`;
    }

    return `Cutting ${cutDemoNames.length} demo${cutDemoNames.length === 1 ? '' : 's'} into a new file. Video-heavy keys can take a while to copy.`;
  }, [
    appendSourceUrls.length,
    cutDemoNames.length,
    operation,
    orderedSelectedSourceUrls.length,
  ]);

  const canProcess = useMemo(() => {
    if (operation === 'lerobot') {
      return (
        useBackend &&
        backend.available &&
        Boolean(backendScan) &&
        orderedSelectedSourceUrls.length > 0 &&
        backendScanPaths.length === orderedSelectedSourceUrls.length &&
        !backendLoading &&
        !resolveError
      );
    }

    if (selectedKeys.length === 0) {
      return false;
    }

    // Python backend: need selected sources and not loading.
    if (useBackend && backend.available) {
      return (
        Boolean(backendScan) &&
        orderedSelectedSourceUrls.length > 0 &&
        backendScanPaths.length === orderedSelectedSourceUrls.length &&
        !backendLoading &&
        !resolveError
      );
    }

    if (selectedSourceLoading || !selectedSourcesReady) {
      return false;
    }

    if (operation === 'merge') {
      return Boolean(primarySourceUrl) && mergeSourceUrls.length > 0;
    }

    if (operation === 'append') {
      return Boolean(primarySourceUrl) && appendSourceUrls.length > 0;
    }

    return Boolean(primarySourceUrl) && cutDemoNames.length > 0;
  }, [
    appendSourceUrls.length,
    backend.available,
    backendLoading,
    cutDemoNames.length,
    backendScan,
    backendScanPaths.length,
    resolveError,
    mergeSourceUrls.length,
    operation,
    orderedSelectedSourceUrls.length,
    primarySourceUrl,
    selectedKeys.length,
    selectedSourceLoading,
    selectedSourcesReady,
    useBackend,
  ]);
  const resultResetKey = useMemo(
    () =>
      JSON.stringify({
        operation,
        primarySourceUrl,
        mergeSourceUrls,
        appendSourceUrls,
        lerobotSourceUrls,
        skipFailedDemos,
        lerobotModalityJsonPath,
        lerobotConversionConfigPath,
        lerobotModalityPythonPath,
        lerobotDefaultTask,
        lerobotTaskRulesText,
        cutStartDemoName,
        cutEndDemoName,
        selectedKeys,
      }),
    [
      appendSourceUrls,
      cutEndDemoName,
      cutStartDemoName,
      lerobotConversionConfigPath,
      lerobotDefaultTask,
      lerobotModalityJsonPath,
      lerobotModalityPythonPath,
      lerobotSourceUrls,
      lerobotTaskRulesText,
      mergeSourceUrls,
      operation,
      primarySourceUrl,
      selectedKeys,
      skipFailedDemos,
    ],
  );
  const hasDownloadReady = Boolean(
    lastResult?.downloadUrl || lastResult?.downloadBlob,
  );

  useEffect(() => {
    setLastResult(null);
  }, [resultResetKey]);

  async function handleProcess() {
    if (!canProcess) {
      return;
    }

    setProcessingError(null);
    setLastResult(null);
    setProgress(null);
    setIsProcessing(true);

    try {
      if (useBackend && backend.available) {
        // Python backend processing.
        if (backendScanPaths.length === 0 || resolveError) {
          throw new Error(
            resolveError ??
              'Could not resolve file paths on the Python backend.',
          );
        }

        if (operation === 'lerobot') {
          const taskRules = parseTaskRulesJson(lerobotTaskRulesText);
          const result = await runLeRobotConvert(
            {
              paths: backendScanPaths,
              outputName: defaultOutputName,
              skipFailed: skipFailedDemos,
              modalityJson: lerobotModalityJsonPath.trim() || undefined,
              conversionConfigJson:
                lerobotConversionConfigPath.trim() || undefined,
              modalityPython: lerobotModalityPythonPath.trim() || undefined,
              defaultTask: lerobotDefaultTask.trim() || undefined,
              taskRules,
            },
            { onProgress: setProgress },
          );

          setLastResult({
            fileName: result.fileName,
            demoCount: result.demoCount,
            selectedKeyCount: result.selectedKeyCount,
            outputPath: result.outputPath,
            outputType: result.outputType,
            skippedDemoCount: result.skippedDemoCount,
            totalFrames: result.totalFrames,
            taskCount: result.taskCount,
          });
          return;
        }

        const result = await runProcess(
          {
            paths: backendScanPaths,
            selectedKeys,
            outputName: defaultOutputName,
            operation,
            cutRange:
              operation === 'cut' && cutStartDemoName && cutEndDemoName
                ? {
                    startDemoName: cutStartDemoName,
                    endDemoName: cutEndDemoName,
                  }
                : undefined,
          },
          { onProgress: setProgress },
        );

        setLastResult({
          fileName: result.fileName,
          demoCount: result.demoCount,
          selectedKeyCount: result.selectedKeyCount,
          downloadUrl: result.downloadUrl,
        });
      } else {
        // WASM worker processing.
        const chunks: ArrayBuffer[] = [];
        const orderedSourceIds = selectedSourceStates
          .map((entry) => entry.source?.sourceId)
          .filter((id): id is string => Boolean(id));

        const result = await processDataset(
          {
            operation,
            orderedSourceIds,
            selectedKeys,
            fileName: defaultOutputName,
            cutRange:
              operation === 'cut' && cutStartDemoName && cutEndDemoName
                ? {
                    startDemoName: cutStartDemoName,
                    endDemoName: cutEndDemoName,
                  }
                : undefined,
          },
          {
            onProgress: setProgress,
            onChunk: (chunk) => {
              chunks.push(chunk);
            },
          },
        );

        setLastResult({
          fileName: result.fileName,
          demoCount: result.demoCount,
          selectedKeyCount: result.selectedKeyCount,
          downloadBlob: new Blob(chunks, { type: 'application/x-hdf5' }),
        });
      }
    } catch (error: unknown) {
      setProcessingError(
        error instanceof Error ? error.message : String(error),
      );
    } finally {
      setIsProcessing(false);
      setProgress(null);
    }
  }

  function handleDownload() {
    if (!lastResult) {
      return;
    }

    if (lastResult.downloadUrl) {
      triggerDownloadUrl(lastResult.fileName, lastResult.downloadUrl);
      return;
    }

    if (lastResult.downloadBlob) {
      triggerDownloadBlob(lastResult.fileName, lastResult.downloadBlob);
    }
  }

  function toggleSource(
    url: string,
    selectedUrls: string[],
    setter: (next: string[]) => void,
  ) {
    setter(
      selectedUrls.includes(url)
        ? selectedUrls.filter((currentUrl) => currentUrl !== url)
        : [...selectedUrls, url],
    );
  }

  function toggleKey(keyPath: string) {
    setSelectedKeys((current) =>
      current.includes(keyPath)
        ? current.filter((key) => key !== keyPath)
        : [...current, keyPath].sort((left, right) =>
            left.localeCompare(right),
          ),
    );
  }

  function toggleKeyGroup(keyPaths: readonly string[]) {
    setSelectedKeys((current) => {
      const next = new Set(current);
      const shouldClear = keyPaths.every((keyPath) => next.has(keyPath));

      for (const keyPath of keyPaths) {
        if (shouldClear) {
          next.delete(keyPath);
        } else {
          next.add(keyPath);
        }
      }

      return [...next].sort((left, right) => left.localeCompare(right));
    });
  }

  function toggleGroupCollapse(groupPath: string) {
    setCollapsedGroupPaths((current) =>
      current.includes(groupPath)
        ? current.filter((path) => path !== groupPath)
        : [...current, groupPath],
    );
  }

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Processing</p>
          <h1 className={styles.title}>Dataset Processing</h1>
          <p className={styles.subtitle}>
            Cut demos, merge multiple datasets, append one dataset to another,
            or convert HDF5 files to LeRobot v2.1. Processing leaves the
            original files unchanged.
          </p>
        </div>
      </header>

      {backend.available && (
        <section className={styles.controlsCard}>
          <div className={styles.backendHeader}>
            <FiServer aria-hidden />
            <div>
              <p className={styles.backendTitle}>Python Processing Server</p>
              <p className={styles.backendSubtitle}>
                Native processing — orders of magnitude faster for large files
                with video data.
                {backend.outputDir && (
                  <>
                    {' '}
                    Output directory: <code>{backend.outputDir}</code>
                  </>
                )}
              </p>
            </div>
            <label className={styles.backendToggle}>
              <input
                type="checkbox"
                checked={useBackend}
                onChange={(event) => {
                  setUseBackend(event.target.checked);
                }}
              />
              <span>{useBackend ? 'Active' : 'Off'}</span>
            </label>
          </div>
          {useBackend && backendError && (
            <p className={styles.errorText} style={{ marginTop: '0.75rem' }}>
              {backendError}
            </p>
          )}
          {useBackend && backendLoading && !backendError && (
            <p className={styles.infoText} style={{ marginTop: '0.75rem' }}>
              Scanning files…
            </p>
          )}
          {useBackend && skippedNames.length > 0 && (
            <p className={styles.infoText} style={{ marginTop: '0.75rem' }}>
              Hidden from backend processing because they were opened without a
              desktop filesystem path: {skippedNames.join(', ')}. Reopen them
              with the desktop file picker or turn the backend off to process
              them via WASM.
            </p>
          )}
          {useBackend && resolveError && (
            <p className={styles.errorText} style={{ marginTop: '0.75rem' }}>
              {resolveError}
            </p>
          )}
        </section>
      )}

      {!useBackend && !fileUrl && !file && !fileLoading && (
        <EmptyState openedFileCount={opened.length} />
      )}

      {!useBackend && fileError && (
        <section className={styles.messageCard}>
          <p className={styles.errorText}>{fileError}</p>
        </section>
      )}

      {!useBackend && fileLoading && (
        <section className={styles.messageCard}>
          <p>Loading dataset-processing context…</p>
        </section>
      )}

      {useBackend &&
        backend.available &&
        sourceOptions.length === 0 &&
        skippedNames.length === 0 && (
          <section className={styles.messageCard}>
            <p className={styles.infoText}>
              Open one or more HDF5 files from the home page to enable
              processing.
            </p>
          </section>
        )}

      {sourceOptions.length > 0 && (
        <>
          <section className={styles.controlsCard}>
            <div className={styles.controlGrid}>
              <div className={styles.field}>
                <label
                  className={styles.fieldLabel}
                  htmlFor="dataset-processing-operation"
                >
                  Operation
                </label>
                <select
                  id="dataset-processing-operation"
                  className={styles.select}
                  value={operation}
                  onChange={(event) => {
                    const nextOperation = event.target
                      .value as DatasetProcessingOperation;
                    setOperation(nextOperation);
                    if (nextOperation === 'lerobot' && backend.available) {
                      setUseBackend(true);
                    }
                  }}
                >
                  {DATASET_PROCESSING_OPERATIONS.map((value) => (
                    <option key={value} value={value}>
                      {OPERATION_LABELS[value]}
                    </option>
                  ))}
                </select>
              </div>

              {operation !== 'lerobot' && (
                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="dataset-processing-primary-source"
                  >
                    {operation === 'cut' ? 'Source Dataset' : 'Base Dataset'}
                  </label>
                  <select
                    id="dataset-processing-primary-source"
                    className={styles.select}
                    value={primarySourceUrl ?? ''}
                    onChange={(event) => {
                      const nextUrl = event.target.value;
                      setPrimarySourceUrl(nextUrl || null);
                    }}
                    disabled={sourceOptions.length === 0}
                  >
                    {sourceOptions.map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {entry.label}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {operation === 'cut' && (
              <div className={styles.controlGrid}>
                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="dataset-processing-cut-start"
                  >
                    Start Demo
                  </label>
                  <select
                    id="dataset-processing-cut-start"
                    className={styles.select}
                    value={cutStartDemoName ?? ''}
                    onChange={(event) => {
                      const nextDemo = event.target.value;
                      setCutStartDemoName(nextDemo || null);
                    }}
                    disabled={primaryDemos.length === 0}
                  >
                    {primaryDemos.map((demo) => (
                      <option key={demo.name} value={demo.name}>
                        {demo.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="dataset-processing-cut-end"
                  >
                    End Demo
                  </label>
                  <select
                    id="dataset-processing-cut-end"
                    className={styles.select}
                    value={cutEndDemoName ?? ''}
                    onChange={(event) => {
                      const nextDemo = event.target.value;
                      setCutEndDemoName(nextDemo || null);
                    }}
                    disabled={primaryDemos.length === 0}
                  >
                    {primaryDemos.map((demo) => (
                      <option key={demo.name} value={demo.name}>
                        {demo.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {operation === 'merge' && (
              <>
                <p className={styles.sectionLabel}>Datasets To Merge</p>
                <div className={styles.checkboxGrid}>
                  {sourceOptions
                    .filter((entry) => entry.id !== primarySourceUrl)
                    .map((entry) => (
                      <label key={entry.id} className={styles.checkboxItem}>
                        <input
                          type="checkbox"
                          checked={mergeSourceUrls.includes(entry.id)}
                          onChange={() => {
                            toggleSource(
                              entry.id,
                              mergeSourceUrls,
                              setMergeSourceUrls,
                            );
                          }}
                        />
                        <span>{entry.label}</span>
                      </label>
                    ))}
                </div>
              </>
            )}

            {operation === 'append' && (
              <>
                <p className={styles.sectionLabel}>Datasets To Append</p>
                <div className={styles.checkboxGrid}>
                  {sourceOptions
                    .filter((entry) => entry.id !== primarySourceUrl)
                    .map((entry) => (
                      <label key={entry.id} className={styles.checkboxItem}>
                        <input
                          type="checkbox"
                          checked={appendSourceUrls.includes(entry.id)}
                          onChange={() => {
                            toggleSource(
                              entry.id,
                              appendSourceUrls,
                              setAppendSourceUrls,
                            );
                          }}
                        />
                        <span>{entry.label}</span>
                      </label>
                    ))}
                </div>
              </>
            )}

            {operation === 'lerobot' && (
              <>
                <p className={styles.sectionLabel}>Datasets To Convert</p>
                <div className={styles.checkboxGrid}>
                  {sourceOptions.map((entry) => (
                    <label key={entry.id} className={styles.checkboxItem}>
                      <input
                        type="checkbox"
                        checked={lerobotSourceUrls.includes(entry.id)}
                        onChange={() => {
                          toggleSource(
                            entry.id,
                            lerobotSourceUrls,
                            setLerobotSourceUrls,
                          );
                        }}
                      />
                      <span>{entry.label}</span>
                    </label>
                  ))}
                </div>
                <label className={styles.inlineOption}>
                  <input
                    type="checkbox"
                    checked={skipFailedDemos}
                    onChange={(event) => {
                      setSkipFailedDemos(event.target.checked);
                    }}
                  />
                  <span>Skip demos whose success attribute is false</span>
                </label>
                <p className={styles.infoText}>
                  The converter writes LeRobot v2.1 parquet, metadata,
                  modality.json, and GPU-encoded MP4 videos.
                </p>
                <div className={styles.lerobotConfigGrid}>
                  <div className={styles.field}>
                    <label
                      className={styles.fieldLabel}
                      htmlFor="lerobot-modality-json-path"
                    >
                      Modality JSON
                    </label>
                    <input
                      id="lerobot-modality-json-path"
                      className={styles.select}
                      value={lerobotModalityJsonPath}
                      onChange={(event) => {
                        setLerobotModalityJsonPath(event.target.value);
                      }}
                    />
                    <input
                      aria-label="Select modality JSON"
                      className={styles.fileInput}
                      type="file"
                      accept=".json,application/json"
                      onChange={(event) => {
                        const pickedFile = event.currentTarget.files?.[0];
                        const pickedPath = pickedFile
                          ? getDesktopFilePath(pickedFile)
                          : undefined;
                        if (pickedPath) {
                          setLerobotModalityJsonPath(pickedPath);
                        }
                        event.currentTarget.value = '';
                      }}
                    />
                  </div>

                  <div className={styles.field}>
                    <label
                      className={styles.fieldLabel}
                      htmlFor="lerobot-conversion-config-path"
                    >
                      Conversion Config JSON
                    </label>
                    <input
                      id="lerobot-conversion-config-path"
                      className={styles.select}
                      placeholder="Optional"
                      value={lerobotConversionConfigPath}
                      onChange={(event) => {
                        setLerobotConversionConfigPath(event.target.value);
                      }}
                    />
                    <input
                      aria-label="Select conversion config JSON"
                      className={styles.fileInput}
                      type="file"
                      accept=".json,application/json"
                      onChange={(event) => {
                        const pickedFile = event.currentTarget.files?.[0];
                        const pickedPath = pickedFile
                          ? getDesktopFilePath(pickedFile)
                          : undefined;
                        if (pickedPath) {
                          setLerobotConversionConfigPath(pickedPath);
                        }
                        event.currentTarget.value = '';
                      }}
                    />
                  </div>

                  <div className={styles.field}>
                    <label
                      className={styles.fieldLabel}
                      htmlFor="lerobot-modality-python-path"
                    >
                      GR00T Python Config
                    </label>
                    <input
                      id="lerobot-modality-python-path"
                      className={styles.select}
                      placeholder="Optional, copied into meta/"
                      value={lerobotModalityPythonPath}
                      onChange={(event) => {
                        setLerobotModalityPythonPath(event.target.value);
                      }}
                    />
                    <input
                      aria-label="Select GR00T Python config"
                      className={styles.fileInput}
                      type="file"
                      accept=".py,text/x-python"
                      onChange={(event) => {
                        const pickedFile = event.currentTarget.files?.[0];
                        const pickedPath = pickedFile
                          ? getDesktopFilePath(pickedFile)
                          : undefined;
                        if (pickedPath) {
                          setLerobotModalityPythonPath(pickedPath);
                        }
                        event.currentTarget.value = '';
                      }}
                    />
                  </div>

                  <div className={styles.field}>
                    <label
                      className={styles.fieldLabel}
                      htmlFor="lerobot-default-task"
                    >
                      Default Task String
                    </label>
                    <input
                      id="lerobot-default-task"
                      className={styles.select}
                      value={lerobotDefaultTask}
                      onChange={(event) => {
                        setLerobotDefaultTask(event.target.value);
                      }}
                    />
                  </div>
                </div>

                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="lerobot-task-rules"
                  >
                    Task Rules JSON
                  </label>
                  <textarea
                    id="lerobot-task-rules"
                    className={styles.textarea}
                    placeholder={
                      'Optional: [{"match":"Top_Long","task":"Fold the long-sleeve top on the table"}]'
                    }
                    value={lerobotTaskRulesText}
                    onChange={(event) => {
                      setLerobotTaskRulesText(event.target.value);
                    }}
                  />
                </div>
              </>
            )}

            <div className={styles.statusRow}>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Opened:</span>{' '}
                {sourceOptions.length}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Selected Sources:</span>{' '}
                {orderedSelectedSourceUrls.length}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>
                  {operation === 'lerobot'
                    ? 'Output Format:'
                    : 'Selected Keys:'}
                </span>{' '}
                {operation === 'lerobot' ? 'LeRobot v2.1' : selectedKeys.length}
              </div>
              {operation === 'cut' && (
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Cut Demos:</span>{' '}
                  {cutDemoNames.length}
                </div>
              )}
            </div>
          </section>

          {selectedSourceErrors.length > 0 && (
            <section className={styles.messageCard}>
              {selectedSourceErrors.map((errorMessage) => (
                <p key={errorMessage} className={styles.errorText}>
                  {errorMessage}
                </p>
              ))}
            </section>
          )}
        </>
      )}

      {availableFiles.length > 0 && (
        <>
          {operation !== 'lerobot' && (
            <section className={styles.keysCard}>
              <div className={styles.keysHeader}>
                <div>
                  <h2 className={styles.sectionTitle}>Output Keys</h2>
                  <p className={styles.sectionText}>
                    {operation === 'cut'
                      ? 'Choose which demo-level dataset paths will be copied into the output file.'
                      : 'Choose which demo-level dataset paths will be copied into the output file. Merge and append expose the union of selected source keys, and keys missing in a given demo are skipped for that demo.'}
                  </p>
                </div>
                <div className={styles.keyActions}>
                  {(operation === 'merge' || operation === 'append') && (
                    <button
                      type="button"
                      className={styles.secondaryBtn}
                      onClick={() => {
                        setSelectedKeys(
                          baseKeyPaths.filter((keyPath) =>
                            availableKeySet.has(keyPath),
                          ),
                        );
                      }}
                      disabled={baseKeyPaths.length === 0}
                    >
                      Match Base Dataset
                    </button>
                  )}
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={() => {
                      setSelectedKeys(
                        availableKeyInfos.map((keyInfo) => keyInfo.path),
                      );
                    }}
                    disabled={availableKeyInfos.length === 0}
                  >
                    Select All
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={() => {
                      setSelectedKeys([]);
                    }}
                    disabled={availableKeyInfos.length === 0}
                  >
                    Clear All
                  </button>
                </div>
              </div>

              {selectedSourceLoading ? (
                <p className={styles.infoText}>
                  Loading source dataset structure…
                </p>
              ) : availableKeyInfos.length === 0 ? (
                <p className={styles.infoText}>
                  Select valid source datasets to inspect their available keys.
                </p>
              ) : (
                <div className={styles.keyTree}>
                  {keyTreeNodes.map((node) => (
                    <KeyTreeNodeItem
                      key={node.fullPath}
                      node={node}
                      depth={0}
                      selectedKeys={selectedKeySet}
                      collapsedGroups={collapsedGroupSet}
                      onToggleKey={toggleKey}
                      onToggleGroup={toggleKeyGroup}
                      onToggleCollapse={toggleGroupCollapse}
                    />
                  ))}
                </div>
              )}
            </section>
          )}

          <section className={styles.actionsCard}>
            <div>
              <h2 className={styles.sectionTitle}>Create Output</h2>
              <p className={styles.sectionText}>
                {operation === 'lerobot' ? (
                  <>
                    The LeRobot dataset directory will be created as{' '}
                    <code>{defaultOutputName}</code>.
                  </>
                ) : (
                  <>
                    The processed output will be created as{' '}
                    <code>{defaultOutputName}</code>.
                  </>
                )}
              </p>
            </div>

            {processingError && (
              <p className={styles.errorText}>{processingError}</p>
            )}
            {lastResult && (
              <p className={styles.successText}>
                {lastResult.outputType === 'directory' ? (
                  <>
                    Created {lastResult.fileName} with {lastResult.demoCount}{' '}
                    demos
                    {typeof lastResult.totalFrames === 'number' && (
                      <> and {lastResult.totalFrames} frames</>
                    )}
                    {typeof lastResult.taskCount === 'number' && (
                      <>
                        {' '}
                        across {lastResult.taskCount} task
                        {lastResult.taskCount === 1 ? '' : 's'}
                      </>
                    )}
                    .
                    {lastResult.outputPath && (
                      <>
                        {' '}
                        Output: <code>{lastResult.outputPath}</code>
                      </>
                    )}
                    {Boolean(lastResult.skippedDemoCount) && (
                      <>
                        {' '}
                        Skipped {lastResult.skippedDemoCount} incompatible demo
                        {lastResult.skippedDemoCount === 1 ? '' : 's'}.
                      </>
                    )}
                  </>
                ) : (
                  `Created ${lastResult.fileName} with ${lastResult.demoCount} demos and ${lastResult.selectedKeyCount} keys. Ready to download.`
                )}
              </p>
            )}
            {isProcessing && (
              <div
                className={styles.processingStatus}
                role="status"
                aria-live="polite"
              >
                <FiLoader aria-hidden className={styles.processingSpinner} />
                <div className={styles.processingCopy}>
                  <p className={styles.processingTitle}>
                    {progress?.phase === 'flushing'
                      ? 'Flushing output file…'
                      : progress?.phase === 'streaming'
                        ? 'Preparing download…'
                        : progress?.phase === 'encoding'
                          ? 'Encoding MP4 videos on GPU…'
                          : progress?.phase === 'converting'
                            ? 'Converting HDF5 demos…'
                            : progress?.phase === 'stats'
                              ? 'Aggregating LeRobot stats…'
                              : progress?.phase === 'metadata'
                                ? 'Writing LeRobot metadata…'
                                : 'Processing dataset operation…'}
                  </p>
                  {progress &&
                  ['copying', 'converting', 'encoding'].includes(
                    progress.phase,
                  ) ? (
                    <>
                      <p className={styles.processingText}>
                        {progress.phase === 'copying'
                          ? 'Copying'
                          : progress.phase === 'encoding'
                            ? 'Encoding'
                            : 'Converting'}{' '}
                        <strong>{progress.currentDemoName}</strong> from{' '}
                        <strong>{progress.currentSourceName}</strong>
                        {' — '}demo {progress.overallDemoIndex + 1} of{' '}
                        {progress.overallDemoCount}
                        {progress.datasetDetail && (
                          <>
                            {' — '}
                            <code>{progress.datasetDetail.path}</code>{' '}
                            {progress.datasetDetail.copiedRows}/
                            {progress.datasetDetail.totalRows} rows
                          </>
                        )}
                      </p>
                      <div className={styles.progressBar}>
                        <div
                          className={styles.progressFill}
                          style={{
                            width: `${((progress.overallDemoIndex + 1) / progress.overallDemoCount) * 100}%`,
                          }}
                        />
                      </div>
                    </>
                  ) : (
                    <p className={styles.processingText}>
                      {processingDescription}{' '}
                      {operation === 'lerobot'
                        ? 'The output directory path will be shown when conversion finishes.'
                        : 'The output will be ready to download when processing finishes.'}
                    </p>
                  )}
                </div>
              </div>
            )}

            {!canProcess && (
              <p className={styles.infoText}>
                {operation === 'lerobot'
                  ? backend.available
                    ? 'Select at least one backend-backed HDF5 file to convert to LeRobot v2.1.'
                    : 'LeRobot conversion requires the local Python processing server.'
                  : operation === 'merge'
                    ? 'Select at least two datasets and one key to create a merged output.'
                    : operation === 'append'
                      ? 'Select a base dataset, at least one dataset to append, and one key to create the output.'
                      : 'Select a source dataset, a valid demo range, and at least one key to create the output.'}
              </p>
            )}

            <div className={styles.actionRow}>
              <button
                type="button"
                className={styles.primaryAction}
                onClick={() => {
                  if (hasDownloadReady) {
                    handleDownload();
                    return;
                  }

                  void handleProcess();
                }}
                disabled={isProcessing || (!hasDownloadReady && !canProcess)}
              >
                {hasDownloadReady ? (
                  <FiDownload aria-hidden />
                ) : (
                  <FiFile aria-hidden />
                )}
                <span>
                  {isProcessing
                    ? operation === 'lerobot'
                      ? 'Converting…'
                      : 'Processing…'
                    : hasDownloadReady
                      ? 'Download'
                      : operation === 'lerobot'
                        ? 'Convert'
                        : 'Create'}
                </span>
              </button>
              {operation !== 'lerobot' && (
                <button
                  type="button"
                  className={styles.secondaryBtn}
                  onClick={() => {
                    setSelectedKeys(
                      availableKeyInfos.map((keyInfo) => keyInfo.path),
                    );
                    setProcessingError(null);
                    setLastResult(null);
                  }}
                  disabled={availableKeyInfos.length === 0}
                >
                  <FiRefreshCw aria-hidden />
                  <span>Reset Key Selection</span>
                </button>
              )}
            </div>
          </section>
        </>
      )}

      {!file && !fileLoading && fileUrl && availableFiles.length === 0 && (
        <section className={styles.messageCard}>
          <p>
            Select an opened file from the sidebar to process datasets, or go
            back to the viewer.
          </p>
          <div className={styles.emptyActions}>
            <Link
              className={styles.openBtn}
              to={`/dataset-processing?${createSearchParams({ url: fileUrl }).toString()}`}
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

export default DatasetProcessingPage;
