import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import {
  buildKeyTree,
  useDatasetProcessingSources,
  useResolvedFile,
} from './dataset-selection';
import { DatasetComparisonEmptyState } from './DatasetComparisonEmptyState';
import { DatasetComparisonKeyTreeNode } from './DatasetComparisonKeyTreeNode';
import styles from './DatasetComparisonPage.module.css';
import { DatasetComparisonValuePanel } from './DatasetComparisonValuePanel';
import { formatUnknownError } from './error-utils';
import { loadDatasetComparisonValues } from './pose-trace/hdf5';
import {
  type DatasetComparisonValuesResult,
  type DatasetProcessingKeyInfo,
} from './pose-trace/types';
import { type H5File, useStore } from './stores';

interface SourceOption {
  id: string;
  label: string;
}

interface ComparisonResultState {
  left: DatasetComparisonValuesResult | null;
  right: DatasetComparisonValuesResult | null;
  loading: boolean;
  error: string | null;
}

function commonKeyInfos(
  leftKeys: readonly DatasetProcessingKeyInfo[],
  rightKeys: readonly DatasetProcessingKeyInfo[],
): DatasetProcessingKeyInfo[] {
  const rightByPath = new Map(
    rightKeys.map((keyInfo) => [keyInfo.path, keyInfo]),
  );

  return leftKeys
    .filter((leftKey) => rightByPath.has(leftKey.path))
    .map((leftKey) => ({
      path: leftKey.path,
      availableInDemoCount: Math.min(
        leftKey.availableInDemoCount,
        rightByPath.get(leftKey.path)?.availableInDemoCount ?? 0,
      ),
    }))
    .sort((left, right) => left.path.localeCompare(right.path));
}

function DatasetComparisonPage() {
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const opened = useStore((state) => state.opened);
  const {
    file,
    loading: fileLoading,
    error: fileError,
  } = useResolvedFile(fileUrl);

  const [leftSourceUrl, setLeftSourceUrl] = useState<string | null>(null);
  const [rightSourceUrl, setRightSourceUrl] = useState<string | null>(null);
  const [leftDemoName, setLeftDemoName] = useState<string | null>(null);
  const [rightDemoName, setRightDemoName] = useState<string | null>(null);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [collapsedGroupPaths, setCollapsedGroupPaths] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ComparisonResultState>({
    left: null,
    right: null,
    loading: false,
    error: null,
  });

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

  const sourceOptions = useMemo<SourceOption[]>(
    () =>
      availableFiles.map((availableFile) => ({
        id: availableFile.url,
        label: availableFile.name,
      })),
    [availableFiles],
  );
  const sourceOptionMap = useMemo(
    () => new Map(sourceOptions.map((entry) => [entry.id, entry])),
    [sourceOptions],
  );

  useEffect(() => {
    if (sourceOptions.length === 0) {
      setLeftSourceUrl(null);
      return;
    }

    const activeId =
      sourceOptions.find((entry) => entry.id === (fileUrl ?? file?.url))?.id ??
      sourceOptions[0].id;
    setLeftSourceUrl((current) =>
      current && sourceOptions.some((entry) => entry.id === current)
        ? current
        : activeId,
    );
  }, [file, fileUrl, sourceOptions]);

  useEffect(() => {
    if (sourceOptions.length === 0) {
      setRightSourceUrl(null);
      return;
    }

    if (!leftSourceUrl) {
      setRightSourceUrl(null);
      return;
    }

    const fallback =
      sourceOptions.find((entry) => entry.id !== leftSourceUrl)?.id ??
      sourceOptions.find((entry) => entry.id === leftSourceUrl)?.id ??
      sourceOptions[0].id;
    setRightSourceUrl((current) =>
      current && sourceOptions.some((entry) => entry.id === current)
        ? current
        : fallback,
    );
  }, [leftSourceUrl, sourceOptions]);

  const selectedSourceUrls = useMemo(
    () =>
      [leftSourceUrl, rightSourceUrl].filter(
        (url): url is string => url !== null,
      ),
    [leftSourceUrl, rightSourceUrl],
  );
  const sourceStates = useDatasetProcessingSources(
    availableFiles,
    selectedSourceUrls,
  );
  const leftSourceState = leftSourceUrl
    ? (sourceStates[leftSourceUrl] ?? null)
    : null;
  const rightSourceState = rightSourceUrl
    ? (sourceStates[rightSourceUrl] ?? null)
    : null;
  const selectedSourceStates = selectedSourceUrls
    .map((url) => sourceStates[url])
    .filter(Boolean);
  const selectedSourceLoading = selectedSourceStates.some(
    (entry) => entry.loading,
  );
  const selectedSourceErrors = selectedSourceStates
    .filter((entry) => entry.error)
    .map((entry) => `${entry.file.name}: ${entry.error}`);

  const leftDemos = useMemo(
    () => (leftSourceState?.source?.demos ?? []).map((demo) => demo.name),
    [leftSourceState],
  );
  const rightDemos = useMemo(
    () => (rightSourceState?.source?.demos ?? []).map((demo) => demo.name),
    [rightSourceState],
  );

  useEffect(() => {
    if (leftDemos.length === 0) {
      setLeftDemoName(null);
      return;
    }

    setLeftDemoName((current) =>
      current && leftDemos.includes(current) ? current : leftDemos[0],
    );
  }, [leftDemos]);

  useEffect(() => {
    if (rightDemos.length === 0) {
      setRightDemoName(null);
      return;
    }

    setRightDemoName((current) =>
      current && rightDemos.includes(current) ? current : rightDemos[0],
    );
  }, [rightDemos]);

  const availableKeyInfos = useMemo(
    () =>
      commonKeyInfos(
        leftSourceState?.info?.keyPaths ?? [],
        rightSourceState?.info?.keyPaths ?? [],
      ),
    [leftSourceState, rightSourceState],
  );
  const availableKeySet = useMemo(
    () => new Set(availableKeyInfos.map((keyInfo) => keyInfo.path)),
    [availableKeyInfos],
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
    setSelectedKeys((current) => {
      const filtered = current.filter((keyPath) =>
        availableKeySet.has(keyPath),
      );
      if (filtered.length > 0) {
        return filtered;
      }

      return availableKeyInfos[0] ? [availableKeyInfos[0].path] : [];
    });
  }, [availableKeyInfos, availableKeySet]);

  useEffect(() => {
    const leftSource = leftSourceState?.source ?? null;
    const rightSource = rightSourceState?.source ?? null;

    if (
      !leftSource ||
      !rightSource ||
      !leftDemoName ||
      !rightDemoName ||
      selectedKeys.length === 0
    ) {
      setComparison({ left: null, right: null, loading: false, error: null });
      return undefined;
    }

    let cancelled = false;
    const resolvedLeftSource = leftSource;
    const resolvedRightSource = rightSource;
    const resolvedLeftDemoName = leftDemoName;
    const resolvedRightDemoName = rightDemoName;
    const resolvedSelectedKeys = selectedKeys;
    setComparison((current) => ({
      left: current.left,
      right: current.right,
      loading: true,
      error: null,
    }));

    async function loadComparisonValues() {
      try {
        const [left, right] = await Promise.all([
          loadDatasetComparisonValues(
            resolvedLeftSource,
            resolvedLeftDemoName,
            resolvedSelectedKeys,
          ),
          loadDatasetComparisonValues(
            resolvedRightSource,
            resolvedRightDemoName,
            resolvedSelectedKeys,
          ),
        ]);
        if (!cancelled) {
          setComparison({ left, right, loading: false, error: null });
        }
      } catch (error: unknown) {
        if (!cancelled) {
          setComparison({
            left: null,
            right: null,
            loading: false,
            error: formatUnknownError(error),
          });
        }
      }
    }

    void loadComparisonValues();

    return () => {
      cancelled = true;
    };
  }, [
    leftDemoName,
    leftSourceState,
    rightDemoName,
    rightSourceState,
    selectedKeys,
  ]);

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

  const leftValuesByKey = useMemo(
    () =>
      new Map(
        (comparison.left?.values ?? []).map((entry) => [entry.keyPath, entry]),
      ),
    [comparison.left],
  );
  const rightValuesByKey = useMemo(
    () =>
      new Map(
        (comparison.right?.values ?? []).map((entry) => [entry.keyPath, entry]),
      ),
    [comparison.right],
  );
  const leftOption = leftSourceUrl
    ? (sourceOptionMap.get(leftSourceUrl) ?? null)
    : null;
  const rightOption = rightSourceUrl
    ? (sourceOptionMap.get(rightSourceUrl) ?? null)
    : null;

  function renderRawData() {
    if (selectedKeys.length === 0) {
      return (
        <p className={styles.infoText}>
          Select one or more common keys to compare raw values.
        </p>
      );
    }

    if (comparison.loading) {
      return <p className={styles.infoText}>Loading raw comparison data…</p>;
    }

    if (comparison.error) {
      return <p className={styles.errorText}>{comparison.error}</p>;
    }

    return (
      <div className={styles.comparisonList}>
        {selectedKeys.map((keyPath) => (
          <article key={keyPath} className={styles.comparisonItem}>
            <h3 className={styles.comparisonKey} title={keyPath}>
              {keyPath}
            </h3>
            <div className={styles.valueGrid}>
              <DatasetComparisonValuePanel
                title={leftOption?.label ?? 'Left Dataset'}
                result={leftValuesByKey.get(keyPath)}
              />
              <DatasetComparisonValuePanel
                title={rightOption?.label ?? 'Right Dataset'}
                result={rightValuesByKey.get(keyPath)}
              />
            </div>
          </article>
        ))}
      </div>
    );
  }

  function renderPage() {
    return (
      <div className={styles.root}>
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Comparison</p>
            <h1 className={styles.title}>Dataset Comparison</h1>
            <p className={styles.subtitle}>
              Select datasets and demos for each side, choose common keys, then
              inspect raw values side by side.
            </p>
          </div>
        </header>

        {!fileUrl && availableFiles.length === 0 && !fileLoading && (
          <DatasetComparisonEmptyState openedFileCount={opened.length} />
        )}

        {fileError && (
          <section className={styles.messageCard}>
            <p className={styles.errorText}>{fileError}</p>
          </section>
        )}

        {fileLoading && (
          <section className={styles.messageCard}>
            <p className={styles.infoText}>
              Loading dataset-comparison context…
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
                    htmlFor="dataset-comparison-left-source"
                  >
                    Left Dataset
                  </label>
                  <select
                    id="dataset-comparison-left-source"
                    className={styles.select}
                    value={leftSourceUrl ?? ''}
                    onChange={(event) => {
                      setLeftSourceUrl(event.target.value || null);
                    }}
                  >
                    {sourceOptions.map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {entry.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="dataset-comparison-right-source"
                  >
                    Right Dataset
                  </label>
                  <select
                    id="dataset-comparison-right-source"
                    className={styles.select}
                    value={rightSourceUrl ?? ''}
                    onChange={(event) => {
                      setRightSourceUrl(event.target.value || null);
                    }}
                  >
                    {sourceOptions.map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {entry.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="dataset-comparison-left-demo"
                  >
                    Left Demo
                  </label>
                  <select
                    id="dataset-comparison-left-demo"
                    className={styles.select}
                    value={leftDemoName ?? ''}
                    onChange={(event) => {
                      setLeftDemoName(event.target.value || null);
                    }}
                    disabled={leftDemos.length === 0}
                  >
                    {leftDemos.length === 0 && (
                      <option value="">No demos available</option>
                    )}
                    {leftDemos.map((demoName) => (
                      <option key={demoName} value={demoName}>
                        {demoName}
                      </option>
                    ))}
                  </select>
                </div>

                <div className={styles.field}>
                  <label
                    className={styles.fieldLabel}
                    htmlFor="dataset-comparison-right-demo"
                  >
                    Right Demo
                  </label>
                  <select
                    id="dataset-comparison-right-demo"
                    className={styles.select}
                    value={rightDemoName ?? ''}
                    onChange={(event) => {
                      setRightDemoName(event.target.value || null);
                    }}
                    disabled={rightDemos.length === 0}
                  >
                    {rightDemos.length === 0 && (
                      <option value="">No demos available</option>
                    )}
                    {rightDemos.map((demoName) => (
                      <option key={demoName} value={demoName}>
                        {demoName}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className={styles.statusRow}>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Opened:</span>{' '}
                  {sourceOptions.length}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Left Demos:</span>{' '}
                  {leftDemos.length}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Right Demos:</span>{' '}
                  {rightDemos.length}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Common Keys:</span>{' '}
                  {availableKeyInfos.length}
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusKey}>Selected Keys:</span>{' '}
                  {selectedKeys.length}
                </div>
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

            <section className={styles.keysCard}>
              <div className={styles.keysHeader}>
                <div>
                  <h2 className={styles.sectionTitle}>Common Keys</h2>
                  <p className={styles.sectionText}>
                    Choose demo-level dataset paths present in both selected
                    datasets.
                  </p>
                </div>
                <div className={styles.keyActions}>
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
              ) : leftDemos.length === 0 || rightDemos.length === 0 ? (
                <p className={styles.infoText}>
                  Select datasets that expose demos on both sides.
                </p>
              ) : availableKeyInfos.length === 0 ? (
                <p className={styles.infoText}>
                  The selected datasets do not expose common demo-level keys.
                </p>
              ) : (
                <div className={styles.keyTree}>
                  {keyTreeNodes.map((node) => (
                    <DatasetComparisonKeyTreeNode
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

            <section className={styles.rawCard}>
              <div>
                <h2 className={styles.sectionTitle}>Raw Data</h2>
                <p className={styles.sectionText}>
                  Values are shown for {leftDemoName ?? 'the left demo'} and{' '}
                  {rightDemoName ?? 'the right demo'} using viewer-style matrix
                  slices.
                </p>
              </div>

              {renderRawData()}
            </section>
          </>
        )}
      </div>
    );
  }

  return renderPage();
}

export default DatasetComparisonPage;
