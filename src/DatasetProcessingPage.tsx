import { useEffect, useMemo, useState } from 'react';
import { FiDownload, FiFile, FiFolder, FiRefreshCw } from 'react-icons/fi';
import { Link, createSearchParams, useSearchParams } from 'react-router-dom';

import {
  getDatasetProcessingInfo,
  openPoseTraceSource,
  processDataset,
} from './pose-trace/hdf5';
import type {
  DatasetProcessingKeyInfo,
  DatasetProcessingOperation,
  DatasetProcessingSourceInfo,
  PoseTraceSource,
} from './pose-trace/types';
import { type H5File, useStore } from './stores';
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
}

interface KeyTreeNode {
  name: string;
  fullPath: string;
  children: KeyTreeNode[];
  keyInfo: DatasetProcessingKeyInfo | null;
}

const OPERATION_LABELS: Record<DatasetProcessingOperation, string> = {
  cut: 'Cut',
  merge: 'Merge',
  append: 'Append',
};

function stripExtension(filename: string): string {
  return filename.replace(/\.(hdf5|h5)$/i, '');
}

function triggerDownload(fileName: string, buffer: ArrayBuffer) {
  const blob = new Blob([buffer], { type: 'application/x-hdf5' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.click();
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
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
      childrenByName: new Map<string, MutableKeyTreeNode>(),
    };
  }

  function finalizeNode(node: MutableKeyTreeNode): KeyTreeNode {
    return {
      name: node.name,
      fullPath: node.fullPath,
      keyInfo: node.keyInfo,
      children: [...node.childrenByName.values()].map(finalizeNode),
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
  onToggleKey,
}: {
  node: KeyTreeNode;
  depth: number;
  selectedKeys: ReadonlySet<string>;
  onToggleKey: (keyPath: string) => void;
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

  return (
    <div className={styles.treeBranch}>
      <div
        className={styles.treeBranchLabel}
        style={{ paddingLeft: `${depth * 1.1}rem` }}
        title={node.fullPath}
      >
        <FiFolder aria-hidden className={styles.treeBranchIcon} />
        <span>{node.name}</span>
      </div>
      <div className={styles.treeChildren}>
        {node.children.map((child) => (
          <KeyTreeNodeItem
            key={child.fullPath}
            node={child}
            depth={depth + 1}
            selectedKeys={selectedKeys}
            onToggleKey={onToggleKey}
          />
        ))}
      </div>
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
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ProcessResultState | null>(null);

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

  useEffect(() => {
    if (availableFiles.length === 0) {
      setPrimarySourceUrl(null);
      return;
    }

    const activeUrl = fileUrl ?? file?.url ?? availableFiles[0]?.url ?? null;
    setPrimarySourceUrl((current) =>
      current && availableFiles.some((entry) => entry.url === current)
        ? current
        : activeUrl,
    );
  }, [availableFiles, file, fileUrl]);

  useEffect(() => {
    const availableUrls = new Set(availableFiles.map((entry) => entry.url));
    const activeUrl = primarySourceUrl ?? availableFiles[0]?.url ?? null;
    const firstOther = availableFiles.find((entry) => entry.url !== activeUrl)?.url;

    setMergeSourceUrls((current) => {
      const valid = current.filter((url) => availableUrls.has(url) && url !== activeUrl);
      if (valid.length > 0) {
        return valid;
      }

      return firstOther ? [firstOther] : [];
    });

    setAppendSourceUrls((current) => {
      const valid = current.filter((url) => availableUrls.has(url) && url !== activeUrl);
      if (valid.length > 0) {
        return valid;
      }

      return firstOther ? [firstOther] : [];
    });
  }, [availableFiles, primarySourceUrl]);

  const orderedSelectedSourceUrls = useMemo(() => {
    if (operation === 'cut') {
      return primarySourceUrl ? [primarySourceUrl] : [];
    }

    return availableFiles
      .map((entry) => entry.url)
      .filter((url) =>
        url === primarySourceUrl
          || (operation === 'append' ? appendSourceUrls.includes(url) : mergeSourceUrls.includes(url)),
      );
  }, [appendSourceUrls, availableFiles, mergeSourceUrls, operation, primarySourceUrl]);

  const sourceStates = useDatasetProcessingSources(availableFiles, orderedSelectedSourceUrls);

  const primarySourceState = primarySourceUrl ? sourceStates[primarySourceUrl] ?? null : null;
  const primarySource = primarySourceState?.source ?? null;
  const primaryDemos = primarySource?.demos ?? [];

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
    const keyMap = new Map<string, DatasetProcessingKeyInfo>();

    for (const entry of selectedSourceStates) {
      for (const keyInfo of entry.info?.keyPaths ?? []) {
        const existing = keyMap.get(keyInfo.path);
        if (!existing || existing.availableInDemoCount < keyInfo.availableInDemoCount) {
          keyMap.set(keyInfo.path, keyInfo);
        }
      }
    }

    return [...keyMap.values()].sort((left, right) => left.path.localeCompare(right.path));
  }, [selectedSourceStates]);

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

  const canProcess = useMemo(() => {
    if (selectedSourceLoading || !selectedSourcesReady || selectedKeys.length === 0) {
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
    cutDemoNames.length,
    mergeSourceUrls.length,
    operation,
    primarySourceUrl,
    selectedKeys.length,
    selectedSourceLoading,
    selectedSourcesReady,
  ]);

  async function handleProcess() {
    if (!canProcess) {
      return;
    }

    setProcessingError(null);
    setLastResult(null);
    setIsProcessing(true);

    try {
      const orderedSourceIds = selectedSourceStates.map((entry) => entry.source?.sourceId).filter(
        (sourceId): sourceId is string => Boolean(sourceId),
      );

      const result = await processDataset({
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
      });

      triggerDownload(result.fileName, result.fileBuffer);
      setLastResult({
        fileName: result.fileName,
        demoCount: result.demoCount,
        selectedKeyCount: result.selectedKeyCount,
      });
    } catch (error: unknown) {
      setProcessingError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsProcessing(false);
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

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Processing</p>
          <h1 className={styles.title}>Dataset Processing</h1>
          <p className={styles.subtitle}>
            Cut demos, merge multiple datasets, or append one dataset to another. Processing always creates a new HDF5 download and leaves the original files unchanged.
          </p>
        </div>
      </header>

      {!fileUrl && !file && !fileLoading && <EmptyState openedFileCount={opened.length} />}

      {fileError && (
        <section className={styles.messageCard}>
          <p className={styles.errorText}>{fileError}</p>
        </section>
      )}

      {fileLoading && (
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
                  disabled={availableFiles.length === 0}
                >
                  {availableFiles.map((entry) => (
                    <option key={entry.url} value={entry.url}>
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
                  {availableFiles
                    .filter((entry) => entry.url !== primarySourceUrl)
                    .map((entry) => (
                    <label key={entry.url} className={styles.checkboxItem}>
                      <input
                        type="checkbox"
                        checked={mergeSourceUrls.includes(entry.url)}
                        onChange={() => {
                          toggleSource(entry.url, mergeSourceUrls, setMergeSourceUrls);
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
                  {availableFiles
                    .filter((entry) => entry.url !== primarySourceUrl)
                    .map((entry) => (
                      <label key={entry.url} className={styles.checkboxItem}>
                        <input
                          type="checkbox"
                          checked={appendSourceUrls.includes(entry.url)}
                          onChange={() => {
                            toggleSource(entry.url, appendSourceUrls, setAppendSourceUrls);
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
                    onToggleKey={toggleKey}
                  />
                ))}
              </div>
            )}
          </section>

          <section className={styles.actionsCard}>
            <div>
              <h2 className={styles.sectionTitle}>Create Output</h2>
              <p className={styles.sectionText}>
                The processed output will download as <code>{defaultOutputName}</code>.
              </p>
            </div>

            {processingError && <p className={styles.errorText}>{processingError}</p>}
            {lastResult && (
              <p className={styles.successText}>
                Downloaded {lastResult.fileName} with {lastResult.demoCount} demos and {lastResult.selectedKeyCount} keys.
              </p>
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
                  void handleProcess();
                }}
                disabled={!canProcess || isProcessing}
              >
                <FiDownload aria-hidden />
                <span>{isProcessing ? 'Processing…' : 'Create & Download'}</span>
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
