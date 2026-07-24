import { useEffect, useMemo, useRef, useState } from 'react';
import {
  FiDownload,
  FiFile,
  FiFolder,
  FiLoader,
  FiRefreshCw,
  FiServer,
} from 'react-icons/fi';
import { createSearchParams, Link, useSearchParams } from 'react-router-dom';

import {
  useDatasetProcessingSources,
  useResolvedFile,
} from './dataset-selection';
import { DatasetProcessingEmptyState } from './DatasetProcessingEmptyState';
import { DatasetProcessingKeyTreeNode } from './DatasetProcessingKeyTreeNode';
import styles from './DatasetProcessingPage.module.css';
import {
  buildBackendKeyInfos,
  buildDefaultOutputName,
  buildKeyTree,
  parseTaskRulesJson,
  sumKeyInfos,
} from './DatasetProcessingPage.utils';
import { formatUnknownError } from './error-utils';
import { processDataset } from './pose-trace/hdf5';
import {
  type DatasetProcessingOperation,
  type DatasetProcessingProgress,
} from './pose-trace/types';
import {
  type LeRobotOutputVersion,
  type LeRobotVideoCodec,
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

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}

function isString(value: string | undefined): value is string {
  return typeof value === 'string';
}

function toggleSourceSelection(
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

interface ProcessingReadiness {
  operation: DatasetProcessingOperation;
  useBackend: boolean;
  backendAvailable: boolean;
  hasBackendScan: boolean;
  selectedSourceCount: number;
  backendScanPathCount: number;
  hasModalityJson: boolean;
  backendLoading: boolean;
  hasResolveError: boolean;
  selectedKeyCount: number;
  selectedSourceLoading: boolean;
  selectedSourcesReady: boolean;
  hasPrimarySource: boolean;
  mergeSourceCount: number;
  appendSourceCount: number;
  cutDemoCount: number;
}

function hasReadyBackendSelection(state: ProcessingReadiness): boolean {
  return (
    state.hasBackendScan &&
    state.selectedSourceCount > 0 &&
    state.backendScanPathCount === state.selectedSourceCount &&
    !state.backendLoading &&
    !state.hasResolveError
  );
}

function hasRequiredOperationSources(state: ProcessingReadiness): boolean {
  if (state.operation === 'merge') {
    return state.hasPrimarySource && state.mergeSourceCount > 0;
  }

  if (state.operation === 'append') {
    return state.hasPrimarySource && state.appendSourceCount > 0;
  }

  return state.hasPrimarySource && state.cutDemoCount > 0;
}

function determineProcessingReadiness(state: ProcessingReadiness): boolean {
  if (state.operation === 'lerobot') {
    return (
      state.useBackend &&
      state.backendAvailable &&
      hasReadyBackendSelection(state) &&
      state.hasModalityJson
    );
  }

  if (state.selectedKeyCount === 0) {
    return false;
  }

  if (state.useBackend && state.backendAvailable) {
    return hasReadyBackendSelection(state);
  }

  if (state.selectedSourceLoading || !state.selectedSourcesReady) {
    return false;
  }

  return hasRequiredOperationSources(state);
}

function getProcessingTitle(
  progress: DatasetProcessingProgress | null,
  codec: LeRobotVideoCodec,
): string {
  switch (progress?.phase) {
    case 'flushing':
      return 'Flushing output file…';
    case 'streaming':
      return 'Preparing download…';
    case 'encoding':
      return `Encoding ${codec === 'h264' ? 'H.264' : 'AV1'} MP4 videos…`;
    case 'converting':
      return 'Converting HDF5 demos…';
    case 'stats':
      return 'Aggregating LeRobot stats…';
    case 'metadata':
      return 'Writing LeRobot metadata…';
    default:
      return 'Processing dataset operation…';
  }
}

function getReadinessMessage(
  operation: DatasetProcessingOperation,
  backendAvailable: boolean,
): string {
  if (operation === 'lerobot') {
    return backendAvailable
      ? 'Select at least one backend-backed HDF5 file and a required modality JSON file.'
      : 'LeRobot conversion requires the local Python processing server.';
  }

  if (operation === 'merge') {
    return 'Select at least two datasets and one key to create a merged output.';
  }

  if (operation === 'append') {
    return 'Select a base dataset, at least one dataset to append, and one key to create the output.';
  }

  return 'Select a source dataset, a valid demo range, and at least one key to create the output.';
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
  const [lerobotOutputVersion, setLerobotOutputVersion] =
    useState<LeRobotOutputVersion>('v3.0');
  const [lerobotVideoCodec, setLerobotVideoCodec] =
    useState<LeRobotVideoCodec>('h264');
  const [lerobotOutputDirectory, setLerobotOutputDirectory] = useState('');
  const [
    lerobotOutputDirectoryAuthorization,
    setLerobotOutputDirectoryAuthorization,
  ] = useState('');
  const [lerobotModalityJsonPath, setLerobotModalityJsonPath] = useState('');
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
  const [lerobotWarnings, setLerobotWarnings] = useState<string[]>([]);

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
      sourceOptions.at(0)?.id ??
      null;
    setPrimarySourceUrl((current) =>
      current && sourceOptions.some((entry) => entry.id === current)
        ? current
        : activeId,
    );
  }, [file, fileUrl, sourceOptions, useBackend]);

  useEffect(() => {
    const availableIds = new Set(sourceOptions.map((entry) => entry.id));
    const activeId = primarySourceUrl ?? sourceOptions.at(0)?.id ?? null;
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
        .filter(isDefined),
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

    const paths = selectedSourceOptions
      .map((entry) => entry.backendPath)
      .filter(isString);
    return paths.length === selectedSourceOptions.length ? paths : [];
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
      return undefined;
    }

    if (resolveError) {
      setBackendScan(null);
      setBackendError(resolveError);
      setBackendLoading(false);
      return undefined;
    }

    if (backendScanPaths.length === 0) {
      setBackendScan(null);
      setBackendLoading(false);
      return undefined;
    }

    let cancelled = false;
    setBackendLoading(true);
    setBackendError(null);

    async function scanBackendFiles() {
      try {
        const result = await scanFiles(backendScanPaths);
        if (!cancelled) {
          setBackendScan(result);
          setBackendLoading(false);
        }
      } catch (error: unknown) {
        if (!cancelled) {
          setBackendScan(null);
          setBackendError(formatUnknownError(error));
          setBackendLoading(false);
        }
      }
    }

    void scanBackendFiles();

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
  const primaryDemos = useMemo(
    () =>
      useBackend
        ? backendPrimaryDemos.map((name) => ({ name }))
        : (primarySource?.demos ?? []),
    [backendPrimaryDemos, primarySource, useBackend],
  );

  useEffect(() => {
    if (primaryDemos.length === 0) {
      setCutStartDemoName(null);
      setCutEndDemoName(null);
      return;
    }

    setCutStartDemoName((current) =>
      current && primaryDemos.some((demo) => demo.name === current)
        ? current
        : (primaryDemos.at(0)?.name ?? null),
    );
    setCutEndDemoName((current) =>
      current && primaryDemos.some((demo) => demo.name === current)
        ? current
        : (primaryDemos.at(-1)?.name ?? null),
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
      orderedSelectedSourceUrls
        .map((url) => sourceStates[url])
        .filter(isDefined),
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
    () =>
      buildDefaultOutputName(
        operation,
        selectedSourceRefs,
        cutDemoNames,
        lerobotOutputVersion,
      ),
    [cutDemoNames, lerobotOutputVersion, operation, selectedSourceRefs],
  );
  const processingDescription = useMemo(() => {
    if (operation === 'lerobot') {
      const codecDescription =
        lerobotVideoCodec === 'h264'
          ? 'H.264 (NVENC with automatic CPU fallback)'
          : 'smaller AV1';
      return `Converting ${orderedSelectedSourceUrls.length} HDF5 dataset${orderedSelectedSourceUrls.length === 1 ? '' : 's'} to LeRobot ${lerobotOutputVersion} with ${codecDescription} video.`;
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
    lerobotOutputVersion,
    lerobotVideoCodec,
    operation,
    orderedSelectedSourceUrls.length,
  ]);

  const canProcess = useMemo(
    () =>
      determineProcessingReadiness({
        operation,
        useBackend,
        backendAvailable: backend.available,
        hasBackendScan: Boolean(backendScan),
        selectedSourceCount: orderedSelectedSourceUrls.length,
        backendScanPathCount: backendScanPaths.length,
        hasModalityJson: lerobotModalityJsonPath.trim().length > 0,
        backendLoading,
        hasResolveError: Boolean(resolveError),
        selectedKeyCount: selectedKeys.length,
        selectedSourceLoading,
        selectedSourcesReady,
        hasPrimarySource: Boolean(primarySourceUrl),
        mergeSourceCount: mergeSourceUrls.length,
        appendSourceCount: appendSourceUrls.length,
        cutDemoCount: cutDemoNames.length,
      }),
    [
      appendSourceUrls.length,
      backend.available,
      backendLoading,
      cutDemoNames.length,
      backendScan,
      backendScanPaths.length,
      resolveError,
      mergeSourceUrls.length,
      lerobotModalityJsonPath,
      operation,
      orderedSelectedSourceUrls.length,
      primarySourceUrl,
      selectedKeys.length,
      selectedSourceLoading,
      selectedSourcesReady,
      useBackend,
    ],
  );
  const resultResetKey = useMemo(
    () =>
      JSON.stringify({
        operation,
        primarySourceUrl,
        mergeSourceUrls,
        appendSourceUrls,
        lerobotSourceUrls,
        skipFailedDemos,
        lerobotOutputVersion,
        lerobotVideoCodec,
        lerobotOutputDirectory,
        lerobotOutputDirectoryAuthorization,
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
      lerobotOutputDirectory,
      lerobotOutputDirectoryAuthorization,
      lerobotOutputVersion,
      lerobotSourceUrls,
      lerobotTaskRulesText,
      lerobotVideoCodec,
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
    setLerobotWarnings([]);
  }, [resultResetKey]);

  async function handleChooseLerobotOutputDirectory() {
    const chooseDirectory = globalThis.rebelHdf5Desktop?.chooseDirectory;
    if (!chooseDirectory) {
      return;
    }
    try {
      const selectedDirectory = await chooseDirectory(
        lerobotOutputDirectory.trim() || backend.outputDir,
      );
      if (selectedDirectory) {
        setLerobotOutputDirectory(selectedDirectory.path);
        setLerobotOutputDirectoryAuthorization(selectedDirectory.authorization);
      }
    } catch (error) {
      setProcessingError(
        error instanceof Error
          ? error.message
          : 'Could not open the output folder picker.',
      );
    }
  }

  function getSelectedCutRange() {
    if (operation !== 'cut' || !cutStartDemoName || !cutEndDemoName) {
      return undefined;
    }

    return {
      startDemoName: cutStartDemoName,
      endDemoName: cutEndDemoName,
    };
  }

  async function processLeRobotWithBackend() {
    const taskRules = parseTaskRulesJson(lerobotTaskRulesText);
    const result = await runLeRobotConvert(
      {
        paths: backendScanPaths,
        outputName: defaultOutputName,
        outputDirectory:
          lerobotOutputDirectory.trim() || backend.outputDir || undefined,
        outputDirectoryAuthorization:
          lerobotOutputDirectoryAuthorization || undefined,
        skipFailed: skipFailedDemos,
        modalityJson: lerobotModalityJsonPath.trim() || undefined,
        conversionConfigJson: lerobotConversionConfigPath.trim() || undefined,
        modalityPython: lerobotModalityPythonPath.trim() || undefined,
        defaultTask: lerobotDefaultTask.trim() || undefined,
        taskRules,
        outputVersion: lerobotOutputVersion,
        videoCodec: lerobotVideoCodec,
      },
      {
        onProgress: setProgress,
        onWarning: (message) => {
          setLerobotWarnings((current) =>
            current.includes(message) ? current : [...current, message],
          );
        },
      },
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
  }

  async function processWithBackend() {
    if (backendScanPaths.length === 0 || resolveError) {
      throw new Error(
        resolveError ?? 'Could not resolve file paths on the Python backend.',
      );
    }

    if (operation === 'lerobot') {
      await processLeRobotWithBackend();
      return;
    }

    const result = await runProcess(
      {
        paths: backendScanPaths,
        selectedKeys,
        outputName: defaultOutputName,
        operation,
        cutRange: getSelectedCutRange(),
      },
      { onProgress: setProgress },
    );

    setLastResult({
      fileName: result.fileName,
      demoCount: result.demoCount,
      selectedKeyCount: result.selectedKeyCount,
      downloadUrl: result.downloadUrl,
    });
  }

  async function processInBrowser() {
    const chunks: ArrayBuffer[] = [];
    const orderedSourceIds = selectedSourceStates
      .map((entry) => entry.source?.sourceId)
      .filter(isString);
    const result = await processDataset(
      {
        operation,
        orderedSourceIds,
        selectedKeys,
        fileName: defaultOutputName,
        cutRange: getSelectedCutRange(),
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

  async function handleProcess() {
    if (!canProcess) {
      return;
    }

    setProcessingError(null);
    setLastResult(null);
    setProgress(null);
    setLerobotWarnings([]);
    setIsProcessing(true);

    try {
      if (useBackend && backend.available) {
        await processWithBackend();
      } else {
        await processInBrowser();
      }
    } catch (error: unknown) {
      setProcessingError(formatUnknownError(error));
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

  function renderBackendPanel() {
    if (!backend.available) {
      return null;
    }

    return (
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
              aria-label="Use Python processing server"
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
            desktop filesystem path: {skippedNames.join(', ')}. Reopen them with
            the desktop file picker or turn the backend off to process them via
            WASM.
          </p>
        )}
        {useBackend && resolveError && (
          <p className={styles.errorText} style={{ marginTop: '0.75rem' }}>
            {resolveError}
          </p>
        )}
      </section>
    );
  }

  function renderInitialStatus() {
    if (!useBackend && !fileUrl && !file && !fileLoading) {
      return <DatasetProcessingEmptyState openedFileCount={opened.length} />;
    }

    if (!useBackend && fileError) {
      return (
        <section className={styles.messageCard}>
          <p className={styles.errorText}>{fileError}</p>
        </section>
      );
    }

    if (!useBackend && fileLoading) {
      return (
        <section className={styles.messageCard}>
          <p>Loading dataset-processing context…</p>
        </section>
      );
    }

    if (
      useBackend &&
      backend.available &&
      sourceOptions.length === 0 &&
      skippedNames.length === 0
    ) {
      return (
        <section className={styles.messageCard}>
          <p className={styles.infoText}>
            Open one or more HDF5 files from the home page to enable processing.
          </p>
        </section>
      );
    }

    return null;
  }

  function renderOutputLocationDescription() {
    if (operation !== 'lerobot') {
      return (
        <>
          The processed output will be created as{' '}
          <code>{defaultOutputName}</code>.
        </>
      );
    }

    const outputDirectory = lerobotOutputDirectory.trim() || backend.outputDir;
    return (
      <>
        The LeRobot dataset directory will be created as{' '}
        <code>{defaultOutputName}</code>
        {outputDirectory && (
          <>
            {' '}
            inside <code>{outputDirectory}</code>
          </>
        )}
        .
      </>
    );
  }

  function renderConversionWarnings() {
    if (lerobotWarnings.length === 0) {
      return null;
    }

    return (
      <div className={styles.warningText} role="status">
        <p>Conversion warnings:</p>
        <ul>
          {lerobotWarnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      </div>
    );
  }

  function renderLastResult() {
    if (!lastResult) {
      return null;
    }

    if (lastResult.outputType !== 'directory') {
      return (
        <p className={styles.successText}>
          Created {lastResult.fileName} with {lastResult.demoCount} demos and{' '}
          {lastResult.selectedKeyCount} keys. Ready to download.
        </p>
      );
    }

    return (
      <p className={styles.successText}>
        Created {lastResult.fileName} with {lastResult.demoCount} demos
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
      </p>
    );
  }

  function renderProcessingStatus() {
    if (!isProcessing) {
      return null;
    }

    const hasDetailedProgress =
      progress &&
      ['copying', 'converting', 'encoding'].includes(progress.phase);
    return (
      <div className={styles.processingStatus} role="status" aria-live="polite">
        <FiLoader aria-hidden className={styles.processingSpinner} />
        <div className={styles.processingCopy}>
          <p className={styles.processingTitle}>
            {getProcessingTitle(progress, lerobotVideoCodec)}
          </p>
          {hasDetailedProgress ? (
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
    );
  }

  function renderReadinessMessage() {
    if (canProcess) {
      return null;
    }

    return (
      <p className={styles.infoText}>
        {getReadinessMessage(operation, backend.available)}
      </p>
    );
  }

  function getPrimaryActionLabel() {
    if (isProcessing) {
      return operation === 'lerobot' ? 'Converting…' : 'Processing…';
    }

    if (hasDownloadReady) {
      return 'Download';
    }

    return operation === 'lerobot' ? 'Convert' : 'Create';
  }

  function renderOutputKeysSection() {
    if (operation === 'lerobot') {
      return null;
    }

    function renderKeyTree() {
      if (selectedSourceLoading) {
        return (
          <p className={styles.infoText}>Loading source dataset structure…</p>
        );
      }

      if (availableKeyInfos.length === 0) {
        return (
          <p className={styles.infoText}>
            Select valid source datasets to inspect their available keys.
          </p>
        );
      }

      return (
        <div className={styles.keyTree}>
          {keyTreeNodes.map((node) => (
            <DatasetProcessingKeyTreeNode
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
      );
    }

    return (
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

        {renderKeyTree()}
      </section>
    );
  }

  function renderNoFileState() {
    if (file || fileLoading || !fileUrl || availableFiles.length > 0) {
      return null;
    }

    return (
      <section className={styles.messageCard}>
        <p>
          Select an opened file from the sidebar to process datasets, or go back
          to the viewer.
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
    );
  }

  function renderControlStatus() {
    const value =
      operation === 'lerobot'
        ? `LeRobot ${lerobotOutputVersion}`
        : selectedKeys.length;

    return (
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
            {operation === 'lerobot' ? 'Output Format:' : 'Selected Keys:'}
          </span>{' '}
          {value}
        </div>
        {operation === 'cut' && (
          <div className={styles.statusItem}>
            <span className={styles.statusKey}>Cut Demos:</span>{' '}
            {cutDemoNames.length}
          </div>
        )}
      </div>
    );
  }

  function renderSelectedSourceErrors() {
    if (selectedSourceErrors.length === 0) {
      return null;
    }

    return (
      <section className={styles.messageCard}>
        {selectedSourceErrors.map((errorMessage) => (
          <p key={errorMessage} className={styles.errorText}>
            {errorMessage}
          </p>
        ))}
      </section>
    );
  }

  function renderProcessingError() {
    return processingError ? (
      <p className={styles.errorText}>{processingError}</p>
    ) : null;
  }

  function renderActionButtons() {
    return (
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
          <span>{getPrimaryActionLabel()}</span>
        </button>
        {operation !== 'lerobot' && (
          <button
            type="button"
            className={styles.secondaryBtn}
            onClick={() => {
              setSelectedKeys(availableKeyInfos.map((keyInfo) => keyInfo.path));
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
    );
  }

  function renderPage() {
    return (
      <div className={styles.root}>
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Processing</p>
            <h1 className={styles.title}>Dataset Processing</h1>
            <p className={styles.subtitle}>
              Cut demos, merge multiple datasets, append one dataset to another,
              or convert HDF5 files to LeRobot v2.1 or v3.0. Processing leaves
              the original files unchanged.
            </p>
          </div>
        </header>

        {renderBackendPanel()}
        {renderInitialStatus()}

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
                            aria-label={`Merge ${entry.label}`}
                            checked={mergeSourceUrls.includes(entry.id)}
                            onChange={() => {
                              toggleSourceSelection(
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
                            aria-label={`Append ${entry.label}`}
                            checked={appendSourceUrls.includes(entry.id)}
                            onChange={() => {
                              toggleSourceSelection(
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
                          aria-label={`Convert ${entry.label}`}
                          checked={lerobotSourceUrls.includes(entry.id)}
                          onChange={() => {
                            toggleSourceSelection(
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
                      aria-label="Skip failed demos"
                      checked={skipFailedDemos}
                      onChange={(event) => {
                        setSkipFailedDemos(event.target.checked);
                      }}
                    />
                    <span>Skip demos whose success attribute is false</span>
                  </label>
                  <p className={styles.infoText}>
                    {lerobotOutputVersion === 'v3.0'
                      ? 'V3 packs multiple episodes into larger Parquet and MP4 shards and reconstructs episodes from relational metadata.'
                      : 'V2.1 keeps one Parquet and MP4 file per episode with JSONL task and episode metadata.'}{' '}
                    {lerobotVideoCodec === 'h264'
                      ? 'H.264 prioritizes speed and falls back to CPU encoding when NVENC cannot initialize.'
                      : 'AV1 uses libsvtav1 for smaller output at the cost of slower encoding.'}{' '}
                    The selected modality JSON defines the state/action layout
                    and camera feature names in the output; use the conversion
                    config only when its fields need explicit HDF5 source paths
                    or a robot_type.
                  </p>
                  <div className={styles.lerobotConfigGrid}>
                    <div className={styles.field}>
                      <label
                        className={styles.fieldLabel}
                        htmlFor="lerobot-output-version"
                      >
                        Output Format
                      </label>
                      <select
                        id="lerobot-output-version"
                        className={styles.select}
                        value={lerobotOutputVersion}
                        onChange={(event) => {
                          setLerobotOutputVersion(
                            event.target.value as LeRobotOutputVersion,
                          );
                        }}
                      >
                        <option value="v3.0">LeRobot v3.0 (recommended)</option>
                        <option value="v2.1">LeRobot v2.1 (legacy)</option>
                      </select>
                    </div>

                    <div className={styles.field}>
                      <label
                        className={styles.fieldLabel}
                        htmlFor="lerobot-output-directory"
                      >
                        Output Parent Folder
                      </label>
                      <div className={styles.pathPickerRow}>
                        <input
                          id="lerobot-output-directory"
                          className={styles.select}
                          aria-label="LeRobot output parent folder"
                          placeholder="Choose a folder"
                          value={
                            lerobotOutputDirectory || backend.outputDir || ''
                          }
                          readOnly
                        />
                        {globalThis.rebelHdf5Desktop?.chooseDirectory && (
                          <button
                            type="button"
                            className={styles.secondaryBtn}
                            onClick={() => {
                              void handleChooseLerobotOutputDirectory();
                            }}
                          >
                            <FiFolder aria-hidden />
                            Browse
                          </button>
                        )}
                      </div>
                    </div>

                    <div className={styles.field}>
                      <label
                        className={styles.fieldLabel}
                        htmlFor="lerobot-video-codec"
                      >
                        Video Codec
                      </label>
                      <select
                        id="lerobot-video-codec"
                        className={styles.select}
                        value={lerobotVideoCodec}
                        onChange={(event) => {
                          setLerobotVideoCodec(
                            event.target.value as LeRobotVideoCodec,
                          );
                        }}
                      >
                        <option value="h264">H.264 (faster)</option>
                        <option value="av1">AV1 (smaller)</option>
                      </select>
                    </div>

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
                        aria-label="Modality JSON path"
                        placeholder="Required"
                        required
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
                          const input = event.currentTarget;
                          const pickedFile = input.files?.[0];
                          const pickedPath = pickedFile
                            ? getDesktopFilePath(pickedFile)
                            : undefined;
                          if (pickedPath) {
                            setLerobotModalityJsonPath(pickedPath);
                          }
                          input.value = '';
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
                        aria-label="Conversion config JSON path"
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
                          const input = event.currentTarget;
                          const pickedFile = input.files?.[0];
                          const pickedPath = pickedFile
                            ? getDesktopFilePath(pickedFile)
                            : undefined;
                          if (pickedPath) {
                            setLerobotConversionConfigPath(pickedPath);
                          }
                          input.value = '';
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
                        aria-label="GR00T Python config path"
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
                          const input = event.currentTarget;
                          const pickedFile = input.files?.[0];
                          const pickedPath = pickedFile
                            ? getDesktopFilePath(pickedFile)
                            : undefined;
                          if (pickedPath) {
                            setLerobotModalityPythonPath(pickedPath);
                          }
                          input.value = '';
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
                        aria-label="Default task string"
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
                      aria-label="Task rules JSON"
                      placeholder='Optional: [{"match":"Top_Long","task":"Fold the long-sleeve top on the table"}]'
                      value={lerobotTaskRulesText}
                      onChange={(event) => {
                        setLerobotTaskRulesText(event.target.value);
                      }}
                    />
                  </div>
                </>
              )}

              {renderControlStatus()}
            </section>

            {renderSelectedSourceErrors()}
          </>
        )}

        {availableFiles.length > 0 && (
          <>
            {renderOutputKeysSection()}

            <section className={styles.actionsCard}>
              <div>
                <h2 className={styles.sectionTitle}>Create Output</h2>
                <p className={styles.sectionText}>
                  {renderOutputLocationDescription()}
                </p>
              </div>

              {renderProcessingError()}
              {renderConversionWarnings()}
              {renderLastResult()}
              {renderProcessingStatus()}
              {renderReadinessMessage()}

              {renderActionButtons()}
            </section>
          </>
        )}

        {renderNoFileState()}
      </div>
    );
  }

  return renderPage();
}

export default DatasetProcessingPage;
