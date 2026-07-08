import { useEffect, useMemo, useState } from 'react';

import {
  getDatasetProcessingInfo,
  openPoseTraceSource,
} from './pose-trace/hdf5';
import {
  type DatasetProcessingKeyInfo,
  type DatasetProcessingSourceInfo,
  type PoseTraceSource,
} from './pose-trace/types';
import { type PythonScanResult } from './python-backend';
import { type H5File, useStore } from './stores';
import { resolveFileUrl } from './utils';

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  if (
    typeof error === 'number' ||
    typeof error === 'boolean' ||
    typeof error === 'bigint'
  ) {
    return `${error}`;
  }

  try {
    return JSON.stringify(error) || 'Unknown error';
  } catch {
    return 'Unknown error';
  }
}

export interface ResolvedFileState {
  file: H5File | null;
  loading: boolean;
  error: string | null;
}

export interface ProcessingSourceState {
  file: H5File;
  source: PoseTraceSource | null;
  info: DatasetProcessingSourceInfo | null;
  loading: boolean;
  error: string | null;
}

export interface KeyTreeNode {
  name: string;
  fullPath: string;
  children: KeyTreeNode[];
  keyInfo: DatasetProcessingKeyInfo | null;
  leafKeyPaths: string[];
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

export function buildKeyTree(
  keyInfos: DatasetProcessingKeyInfo[],
): KeyTreeNode[] {
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
      leafKeyPaths: node.keyInfo
        ? [node.fullPath]
        : children.flatMap((child) => child.leafKeyPaths),
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
      const node =
        existing ?? createNode(segment, currentPath, isLeaf ? keyInfo : null);

      if (existing) {
        if (isLeaf) {
          existing.keyInfo = keyInfo;
        }
      } else {
        currentLevel.set(segment, node);
      }

      currentLevel = node.childrenByName;
    }
  }

  return sortKeyTreeNodes([...rootNodes.values()].map(finalizeNode));
}

export function sumKeyInfos(
  keyInfos: readonly DatasetProcessingKeyInfo[],
): DatasetProcessingKeyInfo[] {
  const keyMap = new Map<string, DatasetProcessingKeyInfo>();

  for (const keyInfo of keyInfos) {
    const existing = keyMap.get(keyInfo.path);
    if (existing) {
      existing.availableInDemoCount += keyInfo.availableInDemoCount;
      continue;
    }

    keyMap.set(keyInfo.path, { ...keyInfo });
  }

  return [...keyMap.values()].sort((left, right) =>
    left.path.localeCompare(right.path),
  );
}

export function buildBackendKeyInfos(
  fileInfo: PythonScanResult['files'][number] | null,
): DatasetProcessingKeyInfo[] {
  if (!fileInfo) {
    return [];
  }

  if (fileInfo.keyCounts) {
    return Object.entries(fileInfo.keyCounts)
      .map(([path, availableInDemoCount]) => ({
        path,
        availableInDemoCount,
      }))
      .sort((left, right) => left.path.localeCompare(right.path));
  }

  return fileInfo.keys
    .map((path) => ({
      path,
      availableInDemoCount: fileInfo.demoCount,
    }))
    .sort((left, right) => left.path.localeCompare(right.path));
}

export function useResolvedFile(fileUrl: string | null): ResolvedFileState {
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
    const resolvedUrl = fileUrl;
    setState({ file: null, loading: true, error: null });

    async function resolveFile() {
      try {
        const resolvedFile = await resolveFileUrl(resolvedUrl);
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

export function useDatasetProcessingSources(
  availableFiles: H5File[],
  selectedSourceUrls: string[],
): Record<string, ProcessingSourceState> {
  const [state, setState] = useState<Record<string, ProcessingSourceState>>({});

  const availableFileMap = useMemo(
    () => new Map(availableFiles.map((file) => [file.url, file])),
    [availableFiles],
  );
  const selectedUrls = useMemo(
    () =>
      [...new Set(selectedSourceUrls)].filter((url) =>
        availableFileMap.has(url),
      ),
    [availableFileMap, selectedSourceUrls],
  );

  useEffect(() => {
    if (selectedUrls.length === 0) {
      setState({});
      return undefined;
    }

    let cancelled = false;
    const cleanups: (() => void)[] = [];

    setState(
      Object.fromEntries(
        selectedUrls.flatMap((url) => {
          const file = availableFileMap.get(url);
          return file
            ? [
                [
                  url,
                  {
                    file,
                    source: null,
                    info: null,
                    loading: true,
                    error: null,
                  },
                ],
              ]
            : [];
        }),
      ),
    );

    async function loadSources() {
      await Promise.all(
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
                error: formatUnknownError(error),
              },
            }));
          }
        }),
      );
    }

    void loadSources();

    return () => {
      cancelled = true;
      cleanups.forEach((cleanup) => {
        cleanup();
      });
    };
  }, [availableFileMap, selectedUrls]);

  return state;
}
