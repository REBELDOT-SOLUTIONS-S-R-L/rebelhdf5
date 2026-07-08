import {
  type DatasetProcessingKeyInfo,
  type DatasetProcessingOperation,
} from './pose-trace/types';
import { type PythonScanResult } from './python-backend';

export interface KeyTreeNode {
  name: string;
  fullPath: string;
  children: KeyTreeNode[];
  keyInfo: DatasetProcessingKeyInfo | null;
  leafKeyPaths: string[];
}

/** Strip a trailing `.h5`/`.hdf5` extension (case-insensitive) from a name. */
export function stripExtension(filename: string): string {
  return filename.replace(/\.(h5|hdf5)$/iu, '');
}

/**
 * Parse the free-text LeRobot task-rules field. Returns `undefined` for blank
 * input, throws for anything that is not a JSON array of plain objects.
 */
export function parseTaskRulesJson(
  text: string,
): Array<Record<string, unknown>> | undefined {
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return undefined;
  }

  const parsed = JSON.parse(trimmed) as unknown;
  if (
    !Array.isArray(parsed) ||
    !parsed.every(
      (entry) => entry && typeof entry === 'object' && !Array.isArray(entry),
    )
  ) {
    throw new Error('Task rules must be a JSON array of objects.');
  }

  return parsed as Array<Record<string, unknown>>;
}

/** Build the default output name suggested for a given processing operation. */
export function buildDefaultOutputName(
  operation: DatasetProcessingOperation,
  sourceFiles: { name: string }[],
  cutDemoNames: string[],
): string {
  if (operation === 'lerobot') {
    if (sourceFiles.length === 1) {
      return `${stripExtension(sourceFiles[0].name)}-lerobot-v21`;
    }

    return `lerobot-v21-${sourceFiles.length}-datasets`;
  }

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

/** Recursively sort tree nodes: groups before leaves, then alphabetically. */
export function sortKeyTreeNodes(nodes: KeyTreeNode[]): KeyTreeNode[] {
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

/** Build a nested tree from flat, slash-separated key paths. */
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

      if (existing) {
        if (isLeaf) {
          existing.keyInfo = keyInfo;
        }
      } else {
        currentLevel.set(
          segment,
          createNode(segment, currentPath, isLeaf ? keyInfo : null),
        );
      }

      const node = currentLevel.get(segment)!;
      currentLevel = node.childrenByName;
    }
  }

  return sortKeyTreeNodes([...rootNodes.values()].map(finalizeNode));
}

/** Merge duplicate key paths, summing their per-demo availability counts. */
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

/** Convert a backend scan file entry into sorted key-info records. */
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
