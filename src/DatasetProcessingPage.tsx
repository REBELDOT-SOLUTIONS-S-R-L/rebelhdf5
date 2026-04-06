import { useEffect, useMemo, useState } from 'react';
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
import { Link, createSearchParams, useSearchParams } from 'react-router-dom';

import {
  getDatasetProcessingInfo,
  openPoseTraceSource,
  processDataset,
} from './pose-trace/hdf5';
import type {
  DatasetProcessingKeyInfo,
  DatasetProcessingOperation,
  DatasetProcessingProgress,
  DatasetProcessingSourceInfo,
  PoseTraceSource,
} from './pose-trace/types';
import {
  checkBackend,
  listFiles,
  runProcess,
  scanFiles,
  type PythonFileEntry,
  type PythonBackendStatus,
  type PythonScanResult,
} from './python-backend';
import { FileService, type H5File, useStore } from './stores';
import styles from './DatasetProcessingPage.module.css';
import { resolveFileUrl } from './utils';

interface ResolvedFileState {
  file: H5File | null;
  loading: boolean;
  error: string | null;
}

interface ProcessingSourceState {
  file: H5File;
  source: PoseTraceSource | null;
  info: DatasetProcessingSourceInfo | null;
  loading: boolean;
  error: string | null;
}

interface ProcessResultState {
  fileName: string;
  demoCount: number;
  selectedKeyCount: number;
  downloadUrl?: string;
  downloadBlob?: Blob;
}

interface BackendSourceResolution {
  sourceUrl: string;
  file: H5File;
  backendPath: string;
  backendRelativePath?: string;
}

interface BackendSourceResolutionError {
  sourceUrl: string;
  file: H5File | null;
  error: string;
}

function isBackendSourceResolution(
  entry: BackendSourceResolution | BackendSourceResolutionError,
): entry is BackendSourceResolution {
  return 'backendPath' in entry;
}

interface KeyTreeNode {
  name: string;
  fullPath: string;
  children: KeyTreeNode[];
  keyInfo: DatasetProcessingKeyInfo | null;
  leafKeyPaths: string[];
}

const OPERATION_LABELS: Record<DatasetProcessingOperation, string> = {
  cut: 'Cut',
  merge: 'Merge',
  append: 'Append',
};

function stripExtension(filename: string): string {
  return filename.replace(/\.(hdf5|h5)$/i, '');
}

function triggerDownloadBlob(fileName: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.click();
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
}

function triggerDownloadUrl(fileName: string, downloadUrl: string) {
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = fileName;
  link.click();
}

function buildDefaultOutputName(
  operation: DatasetProcessingOperation,
  sourceFiles: H5File[],
  cutDemoNames: string[],
): string {
  if (operation === 'cut') {
    const baseName = stripExtension(sourceFiles[0]?.name ?? 'dataset');
    const firstDemo = cutDemoNames[0] ?? 'start';
    const lastDemo = cutDemoNames[cutDemoNames.length - 1] ?? 'end';
    return `${baseName}-cut-${firstDemo}-${lastDemo}.hdf5`;
  }

  if (operation === 'append') {
    const baseName = stripExtension(sourceFiles[0]?.name ?? 'dataset');
    const appendCount = Math.max(sourceFiles.length - 1, 0);
    return `${baseName}-append-${appendCount}-dataset${appendCount === 1 ? '' : 's'}.hdf5`;
  }

  return `merged-${sourceFiles.length}-datasets.hdf5`;
}

function sortKeyTreeNodes(nodes: KeyTreeNode[]): KeyTreeNode[] {
  return nodes
    .map((node) => ({
      ...node,
      children: sortKeyTreeNodes(node.children),
    }))
    .sort((left, right) => {
      const leftIsLeaf = Boolean(left.keyInfo);
      const rightIsLeaf = Boolean(right.keyInfo);
      if (leftIsLeaf !== rightIsLeaf) {
        return leftIsLeaf ? 1 : -1;
      }

      return left.name.localeCompare(right.name);
    });
}

function buildKeyTree(keyInfos: DatasetProcessingKeyInfo[]): KeyTreeNode[] {
  type MutableKeyTreeNode = KeyTreeNode & {
    childrenByName: Map<string, MutableKeyTreeNode>;
  };

  function createNode(
    name: string,
    fullPath: string,
    keyInfo: DatasetProcessingKeyInfo | null,
  ): MutableKeyTreeNode {
    return {
      name,
      fullPath,
      children: [],
      keyInfo,
      leafKeyPaths: keyInfo ? [fullPath] : [],
      childrenByName: new Map<string, MutableKeyTreeNode>(),
    };
  }

  function finalizeNode(node: MutableKeyTreeNode): KeyTreeNode {
    const children = [...node.childrenByName.values()].map(finalizeNode);
    return {
      name: node.name,
      fullPath: node.fullPath,
      keyInfo: node.keyInfo,
      children,
      leafKeyPaths: node.keyInfo ? [node.fullPath] : children.flatMap((child) => child.leafKeyPaths),
    };
  }

  const rootNodes = new Map<string, MutableKeyTreeNode>();

  for (const keyInfo of keyInfos) {
    const segments = keyInfo.path.split('/');
    let currentLevel = rootNodes;
    let currentPath = '';

    for (const [index, segment] of segments.entries()) {
      currentPath = currentPath ? `${currentPath}/${segment}` : segment;
      const isLeaf = index === segments.length - 1;
      const existing = currentLevel.get(segment);

      if (existing) {
        if (isLeaf) {
          existing.keyInfo = keyInfo;
        }
      } else {
        currentLevel.set(segment, createNode(segment, currentPath, isLeaf ? keyInfo : null));
      }

      const node = currentLevel.get(segment)!;
      currentLevel = node.childrenByName;
    }
  }

  return sortKeyTreeNodes([...rootNodes.values()].map(finalizeNode));
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

  const selectedCount = node.leafKeyPaths.filter((keyPath) => selectedKeys.has(keyPath)).length;
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
        <small className={styles.treeMeta}>{selectedCount}/{node.leafKeyPaths.length}</small>
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

function useDatasetProcessingSources(
  availableFiles: H5File[],
  selectedSourceUrls: string[],
): Record<string, ProcessingSourceState> {
  const [state, setState] = useState<Record<string, ProcessingSourceState>>({});

  const availableFileMap = useMemo(
    () => new Map(availableFiles.map((file) => [file.url, file])),
    [availableFiles],
  );
  const selectedUrls = useMemo(
    () => [...new Set(selectedSourceUrls)].filter((url) => availableFileMap.has(url)),
    [availableFileMap, selectedSourceUrls],
  );

  useEffect(() => {
    if (selectedUrls.length === 0) {
      setState({});
      return;
    }

    let cancelled = false;
    const cleanups: Array<() => void> = [];

    setState(
      Object.fromEntries(
        selectedUrls.map((url) => [
          url,
          {
            file: availableFileMap.get(url)!,
            source: null,
            info: null,
            loading: true,
            error: null,
          },
        ]),
      ),
    );

    void Promise.all(
      selectedUrls.map(async (url) => {
        const file = availableFileMap.get(url);
        if (!file) {
          return;
        }

        try {
          const source = await openPoseTraceSource(file);
          if (cancelled) {
            source.cleanup();
            return;
          }

          cleanups.push(source.cleanup);
          const info = await getDatasetProcessingInfo(source);
          if (cancelled) {
            source.cleanup();
            return;
          }

          setState((current) => ({
            ...current,
            [url]: {
              file,
              source,
              info,
              loading: false,
              error: null,
            },
          }));
        } catch (error: unknown) {
          if (cancelled) {
            return;
          }

          setState((current) => ({
            ...current,
            [url]: {
              file,
              source: null,
              info: null,
              loading: false,
              error: error instanceof Error ? error.message : String(error),
            },
          }));
        }
      }),
    );

    return () => {
      cancelled = true;
      cleanups.forEach((cleanup) => {
        cleanup();
      });
    };
  }, [availableFileMap, selectedUrls]);

  return state;
}

function EmptyState({
  openedFileCount,
}: {
  openedFileCount: number;
}) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Dataset Processing</h2>
      <p className={styles.emptyText}>
        Open one or more HDF5 files in rebelHDF5, then switch to this page to cut, merge, or append datasets.
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

function DatasetProcessingPage() {
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const opened = useStore((state) => state.opened);

  const { file, loading: fileLoading, error: fileError } = useResolvedFile(fileUrl);
  const [operation, setOperation] = useState<DatasetProcessingOperation>('cut');
  const [primarySourceUrl, setPrimarySourceUrl] = useState<string | null>(null);
  const [mergeSourceUrls, setMergeSourceUrls] = useState<string[]>([]);
  const [appendSourceUrls, setAppendSourceUrls] = useState<string[]>([]);
  const [cutStartDemoName, setCutStartDemoName] = useState<string | null>(null);
  const [cutEndDemoName, setCutEndDemoName] = useState<string | null>(null);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [collapsedGroupPaths, setCollapsedGroupPaths] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ProcessResultState | null>(null);
  const [progress, setProgress] = useState<DatasetProcessingProgress | null>(null);

  // Python backend state.
  const [backend, setBackend] = useState<PythonBackendStatus>({
    available: false,
    rootDir: null,
    version: null,
  });
  const [useBackend, setUseBackend] = useState(false);
  const [backendFiles, setBackendFiles] = useState<PythonFileEntry[] | null>(null);
  const [backendFilesLoading, setBackendFilesLoading] = useState(false);
  const [backendScan, setBackendScan] = useState<PythonScanResult | null>(null);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Detect Python backend on mount.
  useEffect(() => {
    let cancelled = false;
    void checkBackend().then((status) => {
      if (!cancelled) {
        setBackend(status);
        if (status.available) {
          setUseBackend(true);
        }
      }
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!useBackend || !backend.available || !backend.rootDir) {
      setBackendFiles(null);
      setBackendFilesLoading(false);
      setBackendScan(null);
      setBackendLoading(false);
      return;
    }

    let cancelled = false;
    setBackendFilesLoading(true);
    setBackendError(null);
    setBackendFiles(null);

    void listFiles(backend.rootDir, true)
      .then((result) => {
        if (!cancelled) {
          setBackendFiles(result.files);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setBackendError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setBackendFilesLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [backend.available, backend.rootDir, useBackend]);

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

  // Unified source option IDs.
  const sourceOptions = useMemo(
    () => availableFiles.map((f) => ({ id: f.url, name: f.name })),
    [availableFiles],
  );

  useEffect(() => {
    if (sourceOptions.length === 0) {
      setPrimarySourceUrl(null);
      return;
    }

    const activeId = (!useBackend ? (fileUrl ?? file?.url) : null) ?? sourceOptions[0]?.id ?? null;
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
      const valid = current.filter((url) => availableIds.has(url) && url !== activeId);
      if (valid.length > 0) {
        return valid;
      }

      return firstOther ? [firstOther] : [];
    });

    setAppendSourceUrls((current) => {
      const valid = current.filter((url) => availableIds.has(url) && url !== activeId);
      if (valid.length > 0) {
        return valid;
      }

      return firstOther ? [firstOther] : [];
    });
  }, [primarySourceUrl, sourceOptions]);

  const orderedSelectedSourceUrls = useMemo(() => {
    if (operation === 'cut') {
      return primarySourceUrl ? [primarySourceUrl] : [];
    }

    return sourceOptions
      .map((entry) => entry.id)
      .filter((id) =>
        id === primarySourceUrl
          || (operation === 'append' ? appendSourceUrls.includes(id) : mergeSourceUrls.includes(id)),
      );
  }, [appendSourceUrls, mergeSourceUrls, operation, primarySourceUrl, sourceOptions]);

  const backendSourceResolutions = useMemo(() => {
    if (!useBackend || !backend.available || !backend.rootDir || !backendFiles) {
      return [] as Array<BackendSourceResolution | BackendSourceResolutionError>;
    }

    return orderedSelectedSourceUrls.map((sourceUrl) => {
      const file = availableFiles.find((entry) => entry.url === sourceUrl) ?? null;
      if (!file) {
        return {
          sourceUrl,
          file,
          error: 'The selected source is no longer available in this session.',
        } satisfies BackendSourceResolutionError;
      }

      if (file.service !== FileService.Local) {
        return {
          sourceUrl,
          file,
          error: `${file.name} is not a local file. The Python backend can only process datasets that exist on disk under the backend server directory.`,
        } satisfies BackendSourceResolutionError;
      }

      const exactMatches = backendFiles.filter(
        (entry) => entry.name === file.name && entry.size === file.file.size,
      );
      if (exactMatches.length === 1) {
        return {
          sourceUrl,
          file,
          backendPath: exactMatches[0].path,
          backendRelativePath: exactMatches[0].relativePath,
        } satisfies BackendSourceResolution;
      }

      const sameNameMatches = backendFiles.filter((entry) => entry.name === file.name);
      if (exactMatches.length === 0) {
        if (sameNameMatches.length > 0) {
          return {
            sourceUrl,
            file,
            error: `Found ${file.name} on the Python backend, but none of the candidates match the local file size (${file.file.size} bytes).`,
          } satisfies BackendSourceResolutionError;
        }

        return {
          sourceUrl,
          file,
          error: `Could not find ${file.name} on the Python backend.`,
        } satisfies BackendSourceResolutionError;
      }

      return {
        sourceUrl,
        file,
        error: `Multiple backend files match ${file.name}: ${exactMatches
          .map((entry) => entry.relativePath ?? entry.path)
          .join(', ')}. Remove the duplicate or open a unique file.`,
      } satisfies BackendSourceResolutionError;
    });
  }, [availableFiles, backend.available, backend.rootDir, backendFiles, orderedSelectedSourceUrls, useBackend]);

  const backendSourceResolutionError = backendSourceResolutions.find(
    (entry): entry is BackendSourceResolutionError => 'error' in entry,
  )?.error ?? null;
  const backendScanPaths = useMemo(
    () => backendSourceResolutionError
      ? []
      : backendSourceResolutions.filter(isBackendSourceResolution).map((entry) => entry.backendPath),
    [backendSourceResolutionError, backendSourceResolutions],
  );

  // Demo names for the primary source in backend mode.
  const backendPrimaryDemos = useMemo(() => {
    if (!useBackend || !backendScan || !primarySourceUrl) {
      return [];
    }

    const primaryResolution = backendSourceResolutions.find(
      (entry): entry is BackendSourceResolution =>
        'backendPath' in entry && entry.sourceUrl === primarySourceUrl,
    );
    if (!primaryResolution) {
      return [];
    }

    const fileInfo = backendScan.files.find((f) => f.path === primaryResolution.backendPath);
    return fileInfo?.demoNames ?? [];
  }, [backendScan, backendSourceResolutions, primarySourceUrl, useBackend]);

  useEffect(() => {
    if (!useBackend || !backend.available || !backend.rootDir) {
      setBackendScan(null);
      setBackendLoading(false);
      return;
    }

    if (backendFiles === null) {
      setBackendScan(null);
      return;
    }

    if (backendSourceResolutionError) {
      setBackendScan(null);
      setBackendError(backendSourceResolutionError);
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
          setSelectedKeys((current) => current.length > 0 ? current : result.commonKeys);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setBackendScan(null);
          setBackendError(error instanceof Error ? error.message : String(error));
          setBackendLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [
    backend.available,
    backend.rootDir,
    backendFiles,
    backendScanPaths,
    backendSourceResolutionError,
    useBackend,
  ]);

  const sourceStates = useDatasetProcessingSources(availableFiles, orderedSelectedSourceUrls);

  const primarySourceState = primarySourceUrl ? sourceStates[primarySourceUrl] ?? null : null;
  const primarySource = primarySourceState?.source ?? null;
  const primaryDemos = useBackend
    ? backendPrimaryDemos.map((name) => ({ name }))
    : primarySource?.demos ?? [];

  useEffect(() => {
    if (primaryDemos.length === 0) {
      setCutStartDemoName(null);
      setCutEndDemoName(null);
      return;
    }

    setCutStartDemoName((current) =>
      current && primaryDemos.some((demo) => demo.name === current) ? current : primaryDemos[0].name,
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

    const startIndex = primaryDemos.findIndex((demo) => demo.name === cutStartDemoName);
    const endIndex = primaryDemos.findIndex((demo) => demo.name === cutEndDemoName);
    if (startIndex >= 0 && endIndex >= 0 && startIndex > endIndex) {
      setCutEndDemoName(cutStartDemoName);
    }
  }, [cutEndDemoName, cutStartDemoName, primaryDemos]);

  const selectedSourceStates = orderedSelectedSourceUrls
    .map((url) => sourceStates[url])
    .filter((entry): entry is ProcessingSourceState => Boolean(entry));
  const selectedSourceFiles = selectedSourceStates.map((entry) => entry.file);
  const selectedSourceLoading = selectedSourceStates.some((entry) => entry.loading);
  const selectedSourceErrors = selectedSourceStates
    .filter((entry) => entry.error)
    .map((entry) => `${entry.file.name}: ${entry.error}`);
  const selectedSourcesReady = selectedSourceStates.length === orderedSelectedSourceUrls.length
    && selectedSourceStates.every((entry) => entry.source && entry.info && !entry.loading && !entry.error);

  const availableKeyInfos = useMemo(() => {
    // When using the Python backend, derive keys from its scan result.
    if (useBackend && backendScan) {
      const totalDemos = backendScan.files.reduce((sum, f) => sum + f.demoCount, 0);
      return backendScan.commonKeys.map((path) => ({
        path,
        availableInDemoCount: totalDemos,
      }));
    }

    if (useBackend) {
      return [];
    }

    const readySources = selectedSourceStates.filter((entry) => entry.info);
    if (readySources.length === 0) {
      return [];
    }

    // Inner join: only include keys present in every selected source.
    const firstKeyPaths = new Set(
      (readySources[0].info?.keyPaths ?? []).map((keyInfo) => keyInfo.path),
    );

    const commonPaths = new Set(
      [...firstKeyPaths].filter((path) =>
        readySources.every((entry) =>
          entry.info?.keyPaths.some((keyInfo) => keyInfo.path === path),
        ),
      ),
    );

    // Use the highest demo count across sources for display.
    const keyMap = new Map<string, DatasetProcessingKeyInfo>();
    for (const entry of readySources) {
      for (const keyInfo of entry.info?.keyPaths ?? []) {
        if (!commonPaths.has(keyInfo.path)) {
          continue;
        }

        const existing = keyMap.get(keyInfo.path);
        if (!existing || existing.availableInDemoCount < keyInfo.availableInDemoCount) {
          keyMap.set(keyInfo.path, keyInfo);
        }
      }
    }

    return [...keyMap.values()].sort((left, right) => left.path.localeCompare(right.path));
  }, [backendScan, selectedSourceStates, useBackend]);

  const availableKeySet = useMemo(
    () => new Set(availableKeyInfos.map((keyInfo) => keyInfo.path)),
    [availableKeyInfos],
  );
  const baseKeyPaths = useMemo(
    () => (primarySourceState?.info?.keyPaths ?? []).map((keyInfo) => keyInfo.path),
    [primarySourceState],
  );
  const keyTreeNodes = useMemo(() => buildKeyTree(availableKeyInfos), [availableKeyInfos]);
  const selectedKeySet = useMemo(() => new Set(selectedKeys), [selectedKeys]);
  const collapsedGroupSet = useMemo(() => new Set(collapsedGroupPaths), [collapsedGroupPaths]);

  useEffect(() => {
    setSelectedKeys((current) => {
      const filtered = current.filter((key) => availableKeySet.has(key));
      if (filtered.length > 0) {
        return filtered;
      }

      return availableKeyInfos.map((keyInfo) => keyInfo.path);
    });
  }, [availableKeyInfos, availableKeySet]);

  const cutDemoNames = useMemo(() => {
    if (!cutStartDemoName || !cutEndDemoName) {
      return [];
    }

    const startIndex = primaryDemos.findIndex((demo) => demo.name === cutStartDemoName);
    const endIndex = primaryDemos.findIndex((demo) => demo.name === cutEndDemoName);
    if (startIndex < 0 || endIndex < 0 || startIndex > endIndex) {
      return [];
    }

    return primaryDemos.slice(startIndex, endIndex + 1).map((demo) => demo.name);
  }, [cutEndDemoName, cutStartDemoName, primaryDemos]);

  const defaultOutputName = useMemo(
    () => buildDefaultOutputName(operation, selectedSourceFiles, cutDemoNames),
    [cutDemoNames, operation, selectedSourceFiles],
  );
  const processingDescription = useMemo(() => {
    if (operation === 'merge') {
      return `Merging ${orderedSelectedSourceUrls.length} datasets into a new file. Video-heavy keys can take a while to copy.`;
    }

    if (operation === 'append') {
      return `Appending ${appendSourceUrls.length} dataset${appendSourceUrls.length === 1 ? '' : 's'} to the base dataset. Video-heavy keys can take a while to copy.`;
    }

    return `Cutting ${cutDemoNames.length} demo${cutDemoNames.length === 1 ? '' : 's'} into a new file. Video-heavy keys can take a while to copy.`;
  }, [appendSourceUrls.length, cutDemoNames.length, operation, orderedSelectedSourceUrls.length]);

  const canProcess = useMemo(() => {
    if (selectedKeys.length === 0) {
      return false;
    }

    // Python backend: need selected sources and not loading.
    if (useBackend && backend.available) {
      return Boolean(backendScan)
        && orderedSelectedSourceUrls.length > 0
        && backendScanPaths.length === orderedSelectedSourceUrls.length
        && !backendLoading
        && !backendFilesLoading
        && !backendSourceResolutionError;
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
    backendFilesLoading,
    cutDemoNames.length,
    backendScan,
    backendScanPaths.length,
    backendSourceResolutionError,
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
    () => JSON.stringify({
      operation,
      primarySourceUrl,
      mergeSourceUrls,
      appendSourceUrls,
      cutStartDemoName,
      cutEndDemoName,
      selectedKeys,
    }),
    [
      appendSourceUrls,
      cutEndDemoName,
      cutStartDemoName,
      mergeSourceUrls,
      operation,
      primarySourceUrl,
      selectedKeys,
    ],
  );
  const hasDownloadReady = Boolean(lastResult?.downloadUrl || lastResult?.downloadBlob);

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
        const resolvedPaths = backendSourceResolutions
          .filter(isBackendSourceResolution)
          .map((entry) => entry.backendPath);
        if (resolvedPaths.length !== orderedSelectedSourceUrls.length) {
          throw new Error(backendSourceResolutionError
            ?? 'Could not resolve every selected source on the Python backend.');
        }

        const result = await runProcess(
          {
            paths: resolvedPaths,
            selectedKeys,
            outputName: defaultOutputName,
            operation,
            cutRange: operation === 'cut' && cutStartDemoName && cutEndDemoName
              ? { startDemoName: cutStartDemoName, endDemoName: cutEndDemoName }
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
        const orderedSourceIds = selectedSourceStates.map((entry) => entry.source?.sourceId).filter(
          (sourceId): sourceId is string => Boolean(sourceId),
        );

        const result = await processDataset(
          {
            operation,
            orderedSourceIds,
            selectedKeys,
            fileName: defaultOutputName,
            cutRange: operation === 'cut' && cutStartDemoName && cutEndDemoName
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
      setProcessingError(error instanceof Error ? error.message : String(error));
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

  function toggleSource(url: string, selectedUrls: string[], setter: (next: string[]) => void) {
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
        : [...current, keyPath].sort((left, right) => left.localeCompare(right)),
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
            Cut demos, merge multiple datasets, or append one dataset to another. Processing always creates a new HDF5 file and leaves the original files unchanged.
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
                Native processing — orders of magnitude faster for large files with video data.
                {backend.rootDir && <> Serving <code>{backend.rootDir}</code></>}
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
            <p className={styles.errorText} style={{ marginTop: '0.75rem' }}>{backendError}</p>
          )}
          {useBackend && (backendFilesLoading || backendLoading) && !backendError && (
            <p className={styles.infoText} style={{ marginTop: '0.75rem' }}>Scanning files…</p>
          )}
        </section>
      )}

      {!useBackend && !fileUrl && !file && !fileLoading && <EmptyState openedFileCount={opened.length} />}

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

      {availableFiles.length > 0 && (
        <>
          <section className={styles.controlsCard}>
            <div className={styles.controlGrid}>
              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="dataset-processing-operation">
                  Operation
                </label>
                <select
                  id="dataset-processing-operation"
                  className={styles.select}
                  value={operation}
                  onChange={(event) => {
                    const nextOperation = event.target.value as DatasetProcessingOperation;
                    setOperation(nextOperation);
                  }}
                >
                  {(['cut', 'merge', 'append'] as DatasetProcessingOperation[]).map((value) => (
                    <option key={value} value={value}>
                      {OPERATION_LABELS[value]}
                    </option>
                  ))}
                </select>
              </div>

              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="dataset-processing-primary-source">
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
                      {entry.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {operation === 'cut' && (
              <div className={styles.controlGrid}>
                <div className={styles.field}>
                  <label className={styles.fieldLabel} htmlFor="dataset-processing-cut-start">
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
                  <label className={styles.fieldLabel} htmlFor="dataset-processing-cut-end">
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
                          toggleSource(entry.id, mergeSourceUrls, setMergeSourceUrls);
                        }}
                      />
                      <span>{entry.name}</span>
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
                            toggleSource(entry.id, appendSourceUrls, setAppendSourceUrls);
                          }}
                        />
                        <span>{entry.name}</span>
                      </label>
                    ))}
                </div>
              </>
            )}

            <div className={styles.statusRow}>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Opened:</span> {availableFiles.length}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Selected Sources:</span> {orderedSelectedSourceUrls.length}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Selected Keys:</span> {selectedKeys.length}
              </div>
              {operation === 'cut' && (
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Cut Demos:</span> {cutDemoNames.length}
                </div>
              )}
            </div>
          </section>

          {selectedSourceErrors.length > 0 && (
            <section className={styles.messageCard}>
              {selectedSourceErrors.map((errorMessage) => (
                <p key={errorMessage} className={styles.errorText}>{errorMessage}</p>
              ))}
            </section>
          )}
        </>
      )}

      {availableFiles.length > 0 && (
        <>
          <section className={styles.keysCard}>
            <div className={styles.keysHeader}>
              <div>
                <h2 className={styles.sectionTitle}>Output Keys</h2>
                <p className={styles.sectionText}>
                  Choose which demo-level dataset paths will be copied into the output file.
                </p>
              </div>
              <div className={styles.keyActions}>
                {(operation === 'merge' || operation === 'append') && (
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={() => {
                      setSelectedKeys(baseKeyPaths.filter((keyPath) => availableKeySet.has(keyPath)));
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
                    setSelectedKeys(availableKeyInfos.map((keyInfo) => keyInfo.path));
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
              <p className={styles.infoText}>Loading source dataset structure…</p>
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

          <section className={styles.actionsCard}>
            <div>
              <h2 className={styles.sectionTitle}>Create Output</h2>
              <p className={styles.sectionText}>
                The processed output will be created as <code>{defaultOutputName}</code>.
              </p>
            </div>

            {processingError && <p className={styles.errorText}>{processingError}</p>}
            {lastResult && (
              <p className={styles.successText}>
                Created {lastResult.fileName} with {lastResult.demoCount} demos and {lastResult.selectedKeyCount} keys. Ready to download.
              </p>
            )}
            {isProcessing && (
              <div className={styles.processingStatus} role="status" aria-live="polite">
                <FiLoader aria-hidden className={styles.processingSpinner} />
                <div className={styles.processingCopy}>
                  <p className={styles.processingTitle}>
                    {progress?.phase === 'flushing'
                      ? 'Flushing output file…'
                      : progress?.phase === 'streaming'
                        ? 'Preparing download…'
                        : 'Processing dataset operation…'}
                  </p>
                  {progress?.phase === 'copying' ? (
                    <>
                      <p className={styles.processingText}>
                        Copying <strong>{progress.currentDemoName}</strong> from{' '}
                        <strong>{progress.currentSourceName}</strong>
                        {' — '}demo {progress.overallDemoIndex + 1} of {progress.overallDemoCount}
                        {progress.datasetDetail && (
                          <>
                            {' — '}<code>{progress.datasetDetail.path}</code>
                            {' '}{progress.datasetDetail.copiedRows}/{progress.datasetDetail.totalRows} rows
                          </>
                        )}
                      </p>
                      <div className={styles.progressBar}>
                        <div
                          className={styles.progressFill}
                          style={{ width: `${((progress.overallDemoIndex + 1) / progress.overallDemoCount) * 100}%` }}
                        />
                      </div>
                    </>
                  ) : (
                    <p className={styles.processingText}>
                      {processingDescription} The output will be ready to download when processing finishes.
                    </p>
                  )}
                </div>
              </div>
            )}

            {!canProcess && (
              <p className={styles.infoText}>
                {operation === 'merge'
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
                {hasDownloadReady ? <FiDownload aria-hidden /> : <FiFile aria-hidden />}
                <span>
                  {isProcessing ? 'Processing…' : hasDownloadReady ? 'Download' : 'Create'}
                </span>
              </button>
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
            </div>
          </section>
        </>
      )}

      {!file && !fileLoading && fileUrl && availableFiles.length === 0 && (
        <section className={styles.messageCard}>
          <p>
            Select an opened file from the sidebar to process datasets, or go back to the viewer.
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
